# pumml_encoder.py
#
# CNN encoder that compresses the three PUMML input image channels into a
# fixed-dimensional embedding vector used as the context Y for the CFM.
#
# Input  : (batch, 3, 40, 40)
#   channel 0 : ch_charged_lv   fine grid  40x40
#   channel 1 : ch_charged_pu   fine grid  40x40
#   channel 2 : ch_neutral_all  coarse 10x10 upsampled to 40x40
#
# Output : (batch, embed_dim)   default embed_dim = 128
#
# Architecture:
#   Conv1  : 10 filters, 6x6, stride 2, pad 2  -> (10, 20, 20)  ReLU
#   Conv2  : 10 filters, 6x6, stride 2, pad 2  -> (10,  9,  9)  ReLU
#   Flatten: 10*9*9 = 810
#   FC1    : 810 -> 256                                          ReLU
#   FC2    : 256 -> embed_dim                                    ReLU


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PummlEncoder(nn.Module):
    """
    CNN encoder for the three PUMML input channels.

    Parameters
    ----------
    n_filters  : int  number of conv filters per layer (default 10, as in paper)
    filter_size: int  conv kernel size (default 6, as in paper)
    stride     : int  conv stride (default 2, as in paper)
    embed_dim  : int  output embedding dimension (default 128)
    dropout    : float dropout probability in FC layers (default 0.0)
    """

    def __init__(
        self,
        n_filters:   int   = 10,
        filter_size: int   = 6,
        stride:      int   = 2,
        embed_dim:   int   = 128,
        dropout:     float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # convolutional backbone (PUMML architecture)
        # padding=2 keeps the same zero-pad-by-2 logic as the paper
        self.conv1 = nn.Conv2d(3,         n_filters, filter_size, stride=stride, padding=2)
        self.conv2 = nn.Conv2d(n_filters, n_filters, filter_size, stride=stride, padding=2)

        # Compute the flattened size after the two conv layers dynamically
        # by doing a dummy forward pass through the conv layers only.
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 40, 40)
            dummy = F.relu(self.conv1(dummy))
            dummy = F.relu(self.conv2(dummy))
            self._flat_dim = dummy.flatten(start_dim=1).shape[1]

        print(f"[PummlEncoder] conv output flat dim: {self._flat_dim}")

        # fully-connected head
        self.fc1     = nn.Linear(self._flat_dim, 256)
        self.fc2     = nn.Linear(256, embed_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, 3, 40, 40)

        Returns
        -------
        embedding : (batch, embed_dim)
        """
        x = F.relu(self.conv1(x))   # (B, 10, 20, 20)
        x = F.relu(self.conv2(x))   # (B, 10,  9,  9)
        x = x.flatten(start_dim=1)  # (B, 810)
        x = self.dropout(F.relu(self.fc1(x)))  # (B, 256)
        x = F.relu(self.fc2(x))                # (B, embed_dim)
        return x



# Image pre-processing shared between training and inference
def preprocess_images(data: dict, start: int, end: int) -> torch.Tensor:
    """
    Load and preprocess a slice of jet_images.npz into a (N, 3, 40, 40) tensor.

    Steps:
      1. Load ch_charged_lv and ch_charged_pu  (already 40x40).
      2. Load ch_neutral_all (10x10), upsample to 40x40 with nearest-neighbour
         and divide by 16 to preserve total pT (paper Section 2).
      3. Stack into a 3-channel tensor.

    Parameters
    ----------
    data  : dict  loaded from np.load(jet_images.npz)
    start : int   first jet index
    end   : int   one-past-last jet index

    Returns
    -------
    inputs : torch.FloatTensor  (N, 3, 40, 40)
    """
    end = min(end, len(data["ch_neutral_lv"]))

    ch_lv = torch.tensor(
        data["ch_charged_lv"][start:end].astype(np.float32)
    )   # (N, 40, 40)

    ch_pu = torch.tensor(
        data["ch_charged_pu"][start:end].astype(np.float32)
    )   # (N, 40, 40)

    n_all_raw = torch.tensor(
        data["ch_neutral_all"][start:end].astype(np.float32)
    ).unsqueeze(1)   # (N, 1, 10, 10)

    # Upsample neutral channel: 10x10 → 40x40, divide by 16
    n_all = F.interpolate(n_all_raw, size=(40, 40), mode="nearest") / 16.0
    n_all = n_all.squeeze(1)   # (N, 40, 40)

    return torch.stack([ch_lv, ch_pu, n_all], dim=1)   # (N, 3, 40, 40)


def preprocess_targets(data: dict, start: int, end: int) -> torch.Tensor:
    """
    Load ch_neutral_lv and flatten it to (N, 100).

    Parameters
    ----------
    data  : dict  loaded from np.load(jet_images.npz)
    start : int
    end   : int

    Returns
    -------
    targets : torch.FloatTensor  (N, 100)
    """
    end = min(end, len(data["ch_neutral_lv"]))
    t   = data["ch_neutral_lv"][start:end].astype(np.float32)  # (N, 10, 10)
    return torch.tensor(t.reshape(len(t), -1))                  # (N, 100)
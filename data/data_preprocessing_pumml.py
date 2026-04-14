# data_preprocessing_pumml.py
#
# preprocessing for the PUMML image-based CFM.
#
# Role mapping:
#   X  (what the flow generates) = ch_neutral_lv flattened  (N, 100)
#   Y  (conditioning context)    = CNN embedding             (N, 128)
#                                  produced by PummlEncoder
#
# The encoder is run once on the full dataset before training so that
# the CFM training loop only ever sees flat (X, Y) pairs — exactly
# like the scalar pipeline in data_preprocessing.py.
#
# Later, when adding scalar context (jetR, jetArea, etc.)
# just concatenate them onto Y after the CNN embedding:
#   Y = concat([cnn_embedding (128,), scalar_context (k,)])  -> (128+k,)
# and update context_dim in the YAML accordingly.

import os
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from pumml_encoder import PummlEncoder, preprocess_images, preprocess_targets


# Dimensions
TARGET_DIM  = 100   # 10x10  neutral LV image flattened
CONTEXT_DIM = 128   # CNN embedding dim (must match PummlEncoder embed_dim)

# Scaler paths (written next to this file)
_HERE        = os.path.dirname(os.path.abspath(__file__))
MEAN_X_PATH  = os.path.join(_HERE, "pumml_mean_x.npy")
SCALE_X_PATH = os.path.join(_HERE, "pumml_scale_x.npy")



# Internal helpers
def _log1p_t(arr: np.ndarray) -> np.ndarray:
    return np.log1p(np.clip(arr, 0.0, None))


def _remove_invalid(X: np.ndarray, Y: np.ndarray):
    bad = (
        np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1) |
        np.isnan(Y).any(axis=1) | np.isinf(Y).any(axis=1)
    )
    if bad.sum():
        print(f"[pumml preproc] Dropping {bad.sum()} invalid rows.")
    return X[~bad], Y[~bad]


def _init_or_load_scaler_x(X: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    if os.path.exists(MEAN_X_PATH) and os.path.exists(SCALE_X_PATH):
        scaler.mean_          = np.load(MEAN_X_PATH)
        scaler.scale_         = np.load(SCALE_X_PATH)
        scaler.var_           = scaler.scale_ ** 2
        scaler.n_features_in_ = len(scaler.mean_)
        print("[pumml preproc] Loaded X scaler from disk.")
    else:
        scaler.fit(X)
        np.save(MEAN_X_PATH,  scaler.mean_)
        np.save(SCALE_X_PATH, scaler.scale_)
        print("[pumml preproc] Fitted and saved X scaler.")
    return scaler


@torch.no_grad()
def _encode(encoder: PummlEncoder, images: torch.Tensor,
            batch_size: int = 512,
            device: torch.device = torch.device("cpu")) -> np.ndarray:
    """Run the CNN encoder over all images in batches -> (N, embed_dim)."""
    encoder.eval()
    encoder.to(device)
    parts = []
    for i in range(0, len(images), batch_size):
        parts.append(encoder(images[i : i + batch_size].to(device)).cpu().numpy())
    return np.concatenate(parts, axis=0)



# public preprocessor classes
class PummlTrainPreprocessor:
    """
    Preprocessor for the training split of jet_images.npz.

    Runs the CNN encoder once to produce Y embeddings, then applies
    log1p + StandardScaler to X (neutral LV pixels).

    Parameters
    ------
    data_kwargs : dict
        dataset_path : str   path to jet_images.npz
        N_train      : int   number of training jets
        standardize  : bool  apply StandardScaler to X (default True)
        log1p        : bool  apply log1p to X pixel values (default True)
        encode_batch : int   CNN encoding batch size (default 512)
    encoder  : PummlEncoder
    device   : torch.device
    scaler_x : pre-fitted StandardScaler (optional, pass when resuming)
    """

    def __init__(self, data_kwargs, encoder: PummlEncoder,
                 device=torch.device("cpu"), scaler_x=None):

        self.standardize = data_kwargs.get("standardize", True)
        self.log1p_flag  = data_kwargs.get("log1p", True)
        encode_batch     = data_kwargs.get("encode_batch", 512)

        raw  = dict(np.load(data_kwargs["dataset_path"]))
        N    = data_kwargs["N_train"]

        print("[pumml preproc] Encoding training images...")
        imgs = preprocess_images(raw, 0, N)           # (N, 3, 40, 40)
        X    = preprocess_targets(raw, 0, N).numpy()  # (N, 100)
        Y    = _encode(encoder, imgs, encode_batch, device)  # (N, 128)

        if self.log1p_flag:
            X = _log1p_t(X)

        X, Y = _remove_invalid(X, Y)

        print(f"[pumml preproc] Train jets : {len(X)}")
        print(f"[pumml preproc] X (target) : {X.shape}")
        print(f"[pumml preproc] Y (context): {Y.shape}")

        if self.standardize:
            self.scaler_x = scaler_x or _init_or_load_scaler_x(X)
            X = self.scaler_x.transform(X)
        else:
            self.scaler_x = None

        self.X = X.astype(np.float32)
        self.Y = Y.astype(np.float32)

    def get_dataset(self):
        return self.X, self.Y

    def get_scaler(self):
        return self.scaler_x

    def invert_standardize(self, X: np.ndarray) -> np.ndarray:
        if self.standardize and self.scaler_x is not None:
            X = self.scaler_x.inverse_transform(X)
        if self.log1p_flag:
            X = np.expm1(np.clip(X, 0.0, None))
        return np.clip(X, 0.0, None)


class PummlTestPreprocessor:
    #preprocessor for the test split. Uses the scaler fitted on training data.
    def __init__(self, data_kwargs, encoder: PummlEncoder,
                 device=torch.device("cpu"), scaler_x=None):

        self.standardize = data_kwargs.get("standardize", True)
        self.log1p_flag  = data_kwargs.get("log1p", True)
        encode_batch     = data_kwargs.get("encode_batch", 512)
        self.scaler_x    = scaler_x

        raw  = dict(np.load(data_kwargs["dataset_path"]))
        N_tr = data_kwargs["N_train"]
        N_te = data_kwargs["N_test"]

        print("[pumml preproc] Encoding test images...")
        imgs = preprocess_images(raw, N_tr, N_tr + N_te)
        X    = preprocess_targets(raw, N_tr, N_tr + N_te).numpy()

        # Physical copies before any transform — used for validation plots
        self.X_physical   = np.clip(X.copy(), 0.0, None)  # (N, 100) raw pT
        self.imgs_physical = imgs                           # (N, 3, 40, 40)

        Y = _encode(encoder, imgs, encode_batch, device)

        if self.log1p_flag:
            X = _log1p_t(X)

        X, Y = _remove_invalid(X, Y)
        print(f"[pumml preproc] Test  jets : {len(X)}")

        if self.standardize and scaler_x is not None:
            X = scaler_x.transform(X)

        self.X = X.astype(np.float32)
        self.Y = Y.astype(np.float32)

    def get_dataset(self):
        return self.X, self.Y

    def get_scaler(self):
        return self.scaler_x

    def invert_standardize(self, X: np.ndarray) -> np.ndarray:
        if self.standardize and self.scaler_x is not None:
            X = self.scaler_x.inverse_transform(X)
        if self.log1p_flag:
            X = np.expm1(np.clip(X, 0.0, None))
        return np.clip(X, 0.0, None)
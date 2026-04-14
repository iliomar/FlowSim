# data/data_preprocessing.py
# Preprocesses jet datasets for training and testing the CFM model.
#
# Dataset column description from the new jet-generation pipeline:
# 0:  pt_gen
# 1:  eta_gen
# 2:  phi_gen
# 3:  m_gen
# 4:  flavour
# 5:  btag
# 6:  recoPt
# 7:  recoPhi
# 8:  recoEta
# 9:  muon_pT
# 10: recoNConst
# 11: nef
# 12: nhf
# 13: cef
# 14: chf
# 15: qgl
# 16: jetId
# 17: ncharged
# 18: nneutral
# 19: ctag
# 20: nSV
# 21: recoMass
# 22: jetR
# 23: algoCode
# 24: jetArea
# eventually add the 3 channels of the images...but not now

import os
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Directory where this file lives.
PATH = os.path.dirname(__file__)

# Paths to save/load the fitted scaler parameters.
MEAN_X_PATH = os.path.join(PATH, "mean_x.npy")
SCALE_X_PATH = os.path.join(PATH, "scale_x.npy")
MEAN_Y_PATH = os.path.join(PATH, "mean_y.npy")
SCALE_Y_PATH = os.path.join(PATH, "scale_y.npy")

# Fixed flavour categories for stable train/test one-hot encoding.
# This now includes flavour 6, which appeared in your ttbar dataset.
FLAVOUR_CATEGORIES = np.array([0, 1, 2, 3, 4, 5, 6, 21])

# Reconstructed features used as model inputs X.
# These are the features the flow model learns to generate/reconstruct
# conditioned on the context vector Y.
X_COLUMNS = (
    5,   # btag
    6,   # recoPt
    7,   # recoPhi
    8,   # recoEta
    10,  # recoNConst
    11,  # nef
    12,  # nhf
    13,  # cef
    14,  # chf
    15,  # qgl
    16,  # jetId
    17,  # ncharged
    18,  # nneutral
    19,  # ctag
    20,  # nSV
    21,  # recoMass
)

# Context features used as conditioning vector Y before one-hot encoding.
Y_COLUMNS = (
    0,   # pt_gen
    1,   # eta_gen
    2,   # phi_gen
    3,   # m_gen
    4,   # flavour
    9,   # muon_pT
    22,  # jetR
    24,  # jetArea
)

# Indices of the continuous Y variables after flavour is removed and one-hot
# encoding is appended:
# [pt_gen, eta_gen, phi_gen, m_gen, muon_pT, jetR, jetArea, flavour_ohe...]
Y_CONTINUOUS_SLICE = slice(0, 7)


def _load_dataset(dataset_path):
    """Load a dataset from a NumPy .npy file."""
    data = np.load(dataset_path)
    if data.ndim != 2:
        raise ValueError(
            f"Expected a 2D dataset, but got shape {data.shape} from {dataset_path}"
        )
    if data.shape[1] < 25:
        raise ValueError(
            f"Expected at least 25 columns, but got shape {data.shape} from {dataset_path}"
        )
    return data


def _build_xy(data_slice):
    """
    Build the X and Y arrays from a selected slice of the full dataset.

    X contains reconstructed variables.
    Y contains generator-level/context variables.
    """
    X = data_slice[:, X_COLUMNS].copy()
    Y = data_slice[:, Y_COLUMNS].copy()
    return X, Y


def _apply_ratios(X, Y):
    """
    Replace selected reconstructed variables with ratios.

    X column layout:
    0:  btag
    1:  recoPt
    2:  recoPhi
    3:  recoEta
    4:  recoNConst
    5:  nef
    6:  nhf
    7:  cef
    8:  chf
    9:  qgl
    10: jetId
    11: ncharged
    12: nneutral
    13: ctag
    14: nSV
    15: recoMass

    Y column layout before OHE:
    0: pt_gen
    1: eta_gen
    2: phi_gen
    3: m_gen
    4: flavour
    5: muon_pT
    6: jetR
    7: jetArea
    """
    # Use pT ratio instead of absolute reconstructed pT.
    X[:, 1] = X[:, 1] / Y[:, 0]

    # Use mass ratio instead of absolute reconstructed mass.
    X[:, 15] = X[:, 15] / Y[:, 3]

    return X, Y


def _remove_invalid_rows(X, Y):
    """
    Remove rows that contain NaNs or infinities, typically from divisions by zero.
    """
    invalid_mask = np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1)
    X = X[~invalid_mask]
    Y = Y[~invalid_mask]
    return X, Y


def _sanitize_flavour(Y):
    """
    Force flavour labels to be non-negative.

    Some generators may store anti-particles with negative IDs; for the purpose
    of flavour conditioning we keep only the absolute label.
    """
    Y[:, 4] = np.abs(Y[:, 4])
    return Y


def _smear_discrete_like_features(X):
    """
    Apply a small uniform smearing to count-like variables.

    This helps the network learn smoother distributions for integer-valued
    observables. The smearing amplitude is ±0.5.
    """
    # recoNConst
    X[:, 4] = X[:, 4] + 0.5 * np.random.uniform(-1.0, 1.0, len(X[:, 4]))

    # ncharged
    X[:, 11] = X[:, 11] + 0.5 * np.random.uniform(-1.0, 1.0, len(X[:, 11]))

    # nneutral
    X[:, 12] = X[:, 12] + 0.5 * np.random.uniform(-1.0, 1.0, len(X[:, 12]))

    # nSV
    X[:, 14] = X[:, 14] + 0.5 * np.random.uniform(-1.0, 1.0, len(X[:, 14]))

    return X


def _apply_flavour_ohe(Y, flavour_ohe):
    """
    Apply one-hot encoding to the flavour column if requested.

    Before OHE, Y columns are:
    0: pt_gen
    1: eta_gen
    2: phi_gen
    3: m_gen
    4: flavour
    5: muon_pT
    6: jetR
    7: jetArea

    After OHE, Y becomes:
    [pt_gen, eta_gen, phi_gen, m_gen, muon_pT, jetR, jetArea, flavour_one_hot...]
    """
    if not flavour_ohe:
        return Y

    encoder = OneHotEncoder(
        categories=[FLAVOUR_CATEGORIES],
        sparse_output=False,
        handle_unknown="ignore",
    )

    flavour_encoded = encoder.fit_transform(Y[:, 4].reshape(-1, 1))

    # Keep continuous context variables and append one-hot flavour encoding.
    Y = np.hstack((Y[:, 0:4], Y[:, 5:8], flavour_encoded))
    return Y


def _initialize_or_load_scalers(standardize, scaler_x, scaler_y, X, Y, flavour_ohe, tag="train"):
    """
    Create/load the X and Y scalers.

    If saved scaler statistics exist on disk, load them.
    Otherwise, fit scalers from the provided data and save them.
    """
    if not standardize:
        return scaler_x, scaler_y

    if scaler_x is None and scaler_y is None:
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()

        saved_scalers_exist = (
            os.path.exists(MEAN_X_PATH)
            and os.path.exists(SCALE_X_PATH)
            and os.path.exists(MEAN_Y_PATH)
            and os.path.exists(SCALE_Y_PATH)
        )

        if saved_scalers_exist:
            scaler_x.mean_ = np.load(MEAN_X_PATH)
            scaler_x.scale_ = np.load(SCALE_X_PATH)
            scaler_y.mean_ = np.load(MEAN_Y_PATH)
            scaler_y.scale_ = np.load(SCALE_Y_PATH)

            # These attributes are also used internally by scikit-learn checks.
            scaler_x.n_features_in_ = len(scaler_x.mean_)
            scaler_y.n_features_in_ = len(scaler_y.mean_)
            scaler_x.var_ = scaler_x.scale_ ** 2
            scaler_y.var_ = scaler_y.scale_ ** 2
        else:
            scaler_x.fit(X)

            if flavour_ohe:
                # Standardize only the continuous Y variables, not the one-hot bits.
                scaler_y.fit(Y[:, Y_CONTINUOUS_SLICE])
            else:
                scaler_y.fit(Y)

            print(f"{tag} saving")

            np.save(MEAN_X_PATH, scaler_x.mean_)
            np.save(SCALE_X_PATH, scaler_x.scale_)
            np.save(MEAN_Y_PATH, scaler_y.mean_)
            np.save(SCALE_Y_PATH, scaler_y.scale_)

    return scaler_x, scaler_y


def _apply_standardization(X, Y, standardize, scaler_x, scaler_y, flavour_ohe):
    """
    Standardize X and Y using the provided scalers.
    """
    if not standardize:
        return X, Y

    X = scaler_x.transform(X)

    if flavour_ohe:
        # Standardize only the continuous Y block.
        Y[:, Y_CONTINUOUS_SLICE] = scaler_y.transform(Y[:, Y_CONTINUOUS_SLICE])
    else:
        Y = scaler_y.transform(Y)

    return X, Y


class TrainDataPreprocessor:
    """
    Preprocessor for the training split.

    This class:
    1. loads the dataset,
    2. selects X and Y columns,
    3. applies derived ratios,
    4. removes invalid rows,
    5. sanitizes flavour labels,
    6. smears selected integer-like observables,
    7. optionally one-hot encodes flavour,
    8. optionally standardizes X and Y.
    """

    def __init__(self, data_kwargs, scaler_x=None, scaler_y=None):
        self.data = _load_dataset(data_kwargs["dataset_path"])
        self.standardize = data_kwargs["standardize"]
        self.flavour_ohe = data_kwargs.get("flavour_ohe", False)
        self.N_train = data_kwargs["N_train"]
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y

        print(self.data.shape)

        # Select the training slice.
        train_slice = self.data[: self.N_train]

        # Build raw X and Y matrices.
        self.X, self.Y = _build_xy(train_slice)

        # Convert selected features into ratios.
        self.X, self.Y = _apply_ratios(self.X, self.Y)

        # Remove rows with NaNs or infinities.
        self.X, self.Y = _remove_invalid_rows(self.X, self.Y)

        # Ensure flavour labels are non-negative.
        self.Y = _sanitize_flavour(self.Y)

        # Add small smearing to count-like variables.
        self.X = _smear_discrete_like_features(self.X)

        # Helpful debug prints.
        print("Unique flavour labels in training set:", np.unique(self.Y[:, 4]))
        print("Y shape before OHE:", self.Y.shape)

        # Apply flavour one-hot encoding if requested.
        self.Y = _apply_flavour_ohe(self.Y, self.flavour_ohe)

        # Initialize or load scalers.
        self.scaler_x, self.scaler_y = _initialize_or_load_scalers(
            standardize=self.standardize,
            scaler_x=self.scaler_x,
            scaler_y=self.scaler_y,
            X=self.X,
            Y=self.Y,
            flavour_ohe=self.flavour_ohe,
            tag="train",
        )

        # Standardize the processed arrays.
        self.X, self.Y = _apply_standardization(
            X=self.X,
            Y=self.Y,
            standardize=self.standardize,
            scaler_x=self.scaler_x,
            scaler_y=self.scaler_y,
            flavour_ohe=self.flavour_ohe,
        )

    def get_scaler(self):
        """Return the fitted X and Y scalers."""
        return self.scaler_x, self.scaler_y

    def __len__(self):
        """Return the number of rows in the original loaded dataset."""
        return len(self.data)

    def get_dataset(self):
        """Return the processed X and Y arrays."""
        return self.X, self.Y

    def invert_standardize(self, X, Y):
        """
        Undo the standardization transformation.

        This is useful when bringing generated samples back to physical units.
        """
        X = self.scaler_x.inverse_transform(X)

        if self.flavour_ohe:
            Y[:, Y_CONTINUOUS_SLICE] = self.scaler_y.inverse_transform(
                Y[:, Y_CONTINUOUS_SLICE]
            )
        else:
            Y = self.scaler_y.inverse_transform(Y)

        return X, Y


class TestDataPreprocessor:
    """
    Preprocessor for the test split.

    This mirrors the training preprocessor, but uses the test slice of the data.
    """

    def __init__(self, data_kwargs, scaler_x=None, scaler_y=None):
        self.data = _load_dataset(data_kwargs["dataset_path"])
        self.standardize = data_kwargs["standardize"]
        self.flavour_ohe = data_kwargs.get("flavour_ohe", False)
        self.N_train = data_kwargs["N_train"]
        self.N_test = data_kwargs["N_test"]
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y

        print(self.data.shape)

        # Select the test slice.
        test_slice = self.data[self.N_train : self.N_train + self.N_test]

        # Build raw X and Y matrices.
        self.X, self.Y = _build_xy(test_slice)

        # Convert selected features into ratios.
        self.X, self.Y = _apply_ratios(self.X, self.Y)

        # Remove rows with NaNs or infinities.
        self.X, self.Y = _remove_invalid_rows(self.X, self.Y)

        # Ensure flavour labels are non-negative.
        self.Y = _sanitize_flavour(self.Y)

        # Add small smearing to count-like variables.
        self.X = _smear_discrete_like_features(self.X)

        # Helpful debug prints.
        print("Unique flavour labels in test set:", np.unique(self.Y[:, 4]))
        print("Y shape before OHE:", self.Y.shape)

        # Apply flavour one-hot encoding if requested.
        self.Y = _apply_flavour_ohe(self.Y, self.flavour_ohe)

        # Initialize or load scalers.
        self.scaler_x, self.scaler_y = _initialize_or_load_scalers(
            standardize=self.standardize,
            scaler_x=self.scaler_x,
            scaler_y=self.scaler_y,
            X=self.X,
            Y=self.Y,
            flavour_ohe=self.flavour_ohe,
            tag="test",
        )

        # Standardize the processed arrays.
        self.X, self.Y = _apply_standardization(
            X=self.X,
            Y=self.Y,
            standardize=self.standardize,
            scaler_x=self.scaler_x,
            scaler_y=self.scaler_y,
            flavour_ohe=self.flavour_ohe,
        )

    def get_scaler(self):
        """Return the fitted X and Y scalers."""
        return self.scaler_x, self.scaler_y

    def __len__(self):
        """Return the number of rows in the original loaded dataset."""
        return len(self.data)

    def get_dataset(self):
        """Return the processed X and Y arrays."""
        return self.X, self.Y

    def invert_standardize(self, X, Y):
        """
        Undo the standardization transformation.

        This is useful when bringing generated samples back to physical units.
        """
        X = self.scaler_x.inverse_transform(X)

        if self.flavour_ohe:
            print("test invert flavour ohe")
            Y[:, Y_CONTINUOUS_SLICE] = self.scaler_y.inverse_transform(
                Y[:, Y_CONTINUOUS_SLICE]
            )
        else:
            Y = self.scaler_y.inverse_transform(Y)

        return X, Y


if __name__ == "__main__":
    data_kwargs = {
        "dataset_path": "/scratchnvme/cattafe/gen_ttbar_400k.npy",
        "standardize": True,
        "flavour_ohe": True,
        "N_train": 500000,
    }

    train_dataset = TrainDataPreprocessor(data_kwargs)
    X, Y = train_dataset.get_dataset()

    print("Processed X shape:", X.shape)
    print("Processed Y shape:", Y.shape)
    print("First 5 rows of X:")
    print(X[:5])
    print("First 5 rows of Y:")
    print(Y[:5])
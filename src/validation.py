# validation.py
# Validate generated jet samples against real test samples.
#
# This module compares:
#   - generated reconstructed features (samples)
#   - real reconstructed test features (X_test_cpu)
# conditioned on:
#   - physical compact context variables (Y_test_cpu)
#
# Expected input shapes:
#   samples   : (N, 16)
#   X_test_cpu: (N, 16)
#   Y_test_cpu: (N, 8)
#
# X feature layout:
#   0:  btag
#   1:  recoPtoverPt
#   2:  recoPhi
#   3:  recoEta
#   4:  recoNConst
#   5:  nef
#   6:  nhf
#   7:  cef
#   8:  chf
#   9:  qgl
#   10: jetId
#   11: ncharged
#   12: nneutral
#   13: ctag
#   14: nSV
#   15: recoMassOverM
#
# Y compact physical layout:
#   0: pt_gen
#   1: eta_gen
#   2: phi_gen
#   3: m_gen
#   4: flavour
#   5: muon_pT
#   6: jetR
#   7: jetArea

import os
import csv
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Feature names used for plots and summary files
# -----------------------------------------------------------------------------

X_FEATURE_NAMES = [
    "btag",
    "recoPtoverPt",
    "recoPhi",
    "recoEta",
    "recoNConst",
    "nef",
    "nhf",
    "cef",
    "chf",
    "qgl",
    "jetId",
    "ncharged",
    "nneutral",
    "ctag",
    "nSV",
    "recoMassOverM",
]

Y_FEATURE_NAMES = [
    "pt_gen",
    "eta_gen",
    "phi_gen",
    "m_gen",
    "flavour",
    "muon_pT",
    "jetR",
    "jetArea",
]


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def _ensure_dir(path):
    """Create a directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def _safe_combined_range(a, b, feature_name, percentile_low=0.5, percentile_high=99.5):
    """
    Build a robust plotting range from two arrays.

    For highly peaked or outlier-prone distributions, using percentiles gives
    more readable histograms than raw min/max.

    Parameters
    ----------
    a, b : np.ndarray
        Arrays to compare.
    feature_name : str
        Name of the feature, used to handle angular or discrete variables.
    percentile_low, percentile_high : float
        Percentile clipping range.

    Returns
    -------
    tuple[float, float]
        Plotting range.
    """
    combined = np.concatenate([a, b])
    combined = combined[np.isfinite(combined)]

    if len(combined) == 0:
        return (0.0, 1.0)

    # Special treatment for angular variables.
    if "Phi" in feature_name or feature_name == "phi_gen":
        return (-np.pi, np.pi)

    # Special treatment for bounded fractions / taggers.
    if feature_name in ["btag", "nef", "nhf", "cef", "chf", "qgl", "ctag"]:
        return (min(0.0, np.min(combined)), max(1.0, np.max(combined)))

    low = np.percentile(combined, percentile_low)
    high = np.percentile(combined, percentile_high)

    if not np.isfinite(low) or not np.isfinite(high) or low == high:
        low = np.min(combined)
        high = np.max(combined)

    if low == high:
        low -= 0.5
        high += 0.5

    return (low, high)


def _choose_bins(feature_name, data):
    """
    Choose histogram binning based on variable type.

    Discrete-like variables get integer-friendly binning.
    Continuous variables use a standard number of bins.
    """
    discrete_features = {"recoNConst", "ncharged", "nneutral", "nSV", "jetId"}

    if feature_name in discrete_features:
        finite = data[np.isfinite(data)]
        if len(finite) == 0:
            return 20

        min_val = int(np.floor(np.min(finite)))
        max_val = int(np.ceil(np.max(finite)))

        if min_val == max_val:
            min_val -= 1
            max_val += 1

        return np.arange(min_val - 0.5, max_val + 1.5, 1.0)

    return 60


def _ks_statistic(x, y):
    """
    Compute a simple two-sample KS statistic without SciPy.

    Parameters
    ----------
    x, y : np.ndarray
        1D samples.

    Returns
    -------
    float
        KS statistic in [0, 1].
    """
    x = np.sort(x[np.isfinite(x)])
    y = np.sort(y[np.isfinite(y)])

    if len(x) == 0 or len(y) == 0:
        return np.nan

    all_values = np.sort(np.unique(np.concatenate([x, y])))

    cdf_x = np.searchsorted(x, all_values, side="right") / len(x)
    cdf_y = np.searchsorted(y, all_values, side="right") / len(y)

    return np.max(np.abs(cdf_x - cdf_y))


def _summary_stats(real, fake):
    """
    Compute summary metrics comparing one real feature to one generated feature.

    Returns a dictionary of scalar summary values.
    """
    real = real[np.isfinite(real)]
    fake = fake[np.isfinite(fake)]

    if len(real) == 0 or len(fake) == 0:
        return {
            "mean_real": np.nan,
            "mean_fake": np.nan,
            "std_real": np.nan,
            "std_fake": np.nan,
            "median_real": np.nan,
            "median_fake": np.nan,
            "mean_abs_diff": np.nan,
            "ks_statistic": np.nan,
        }

    return {
        "mean_real": float(np.mean(real)),
        "mean_fake": float(np.mean(fake)),
        "std_real": float(np.std(real)),
        "std_fake": float(np.std(fake)),
        "median_real": float(np.median(real)),
        "median_fake": float(np.median(fake)),
        "mean_abs_diff": float(np.mean(np.abs(real - fake[: len(real)]))) if len(real) == len(fake) else np.nan,
        "ks_statistic": float(_ks_statistic(real, fake)),
    }


def _save_summary_csv(samples, X_test_cpu, save_path):
    """
    Save per-feature summary metrics to CSV.
    """
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "feature",
            "mean_real",
            "mean_fake",
            "std_real",
            "std_fake",
            "median_real",
            "median_fake",
            "mean_abs_diff",
            "ks_statistic",
        ])

        for i, feature_name in enumerate(X_FEATURE_NAMES):
            stats = _summary_stats(X_test_cpu[:, i], samples[:, i])
            writer.writerow([
                feature_name,
                stats["mean_real"],
                stats["mean_fake"],
                stats["std_real"],
                stats["std_fake"],
                stats["median_real"],
                stats["median_fake"],
                stats["mean_abs_diff"],
                stats["ks_statistic"],
            ])


# -----------------------------------------------------------------------------
# Plotting functions
# -----------------------------------------------------------------------------

def _plot_feature_histogram(real, fake, feature_name, output_path):
    """
    Plot a histogram comparison for a single reconstructed feature.
    """
    plot_range = _safe_combined_range(real, fake, feature_name)
    bins = _choose_bins(feature_name, np.concatenate([real, fake]))

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.hist(
        real,
        bins=bins,
        range=plot_range if not isinstance(bins, np.ndarray) else None,
        density=True,
        histtype="step",
        linewidth=2,
        label="Real",
    )
    ax.hist(
        fake,
        bins=bins,
        range=plot_range if not isinstance(bins, np.ndarray) else None,
        density=True,
        histtype="step",
        linewidth=2,
        label="Generated",
    )

    ax.set_xlabel(feature_name)
    ax.set_ylabel("Density")
    ax.set_title(f"{feature_name}: real vs generated")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    return fig


def _plot_all_feature_histograms(samples, X_test_cpu, save_dir, epoch, writer=None):
    """
    Plot and save histogram comparisons for all reconstructed features.
    """
    feature_dir = os.path.join(save_dir, f"epoch_{epoch:04d}", "feature_hists")
    _ensure_dir(feature_dir)

    for i, feature_name in enumerate(X_FEATURE_NAMES):
        fig = _plot_feature_histogram(
            real=X_test_cpu[:, i],
            fake=samples[:, i],
            feature_name=feature_name,
            output_path=os.path.join(feature_dir, f"{i:02d}_{feature_name}.png"),
        )

        if writer is not None:
            writer.add_figure(f"validation/{feature_name}", fig, global_step=epoch)

        plt.close(fig)


def _plot_correlation_matrix(data, feature_names, title, output_path):
    """
    Plot a correlation matrix for a feature array.
    """
    corr = np.corrcoef(data, rowvar=False)

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr, vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(np.arange(len(feature_names)))
    ax.set_yticks(np.arange(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=90)
    ax.set_yticklabels(feature_names)
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Correlation")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    return fig


def _plot_real_vs_generated_correlations(samples, X_test_cpu, save_dir, epoch, writer=None):
    """
    Plot and save correlation matrices for real and generated reconstructed features.
    """
    corr_dir = os.path.join(save_dir, f"epoch_{epoch:04d}", "correlations")
    _ensure_dir(corr_dir)

    fig_real = _plot_correlation_matrix(
        X_test_cpu,
        X_FEATURE_NAMES,
        "Real X correlation matrix",
        os.path.join(corr_dir, "real_correlation_matrix.png"),
    )
    if writer is not None:
        writer.add_figure("validation/real_correlation_matrix", fig_real, global_step=epoch)
    plt.close(fig_real)

    fig_fake = _plot_correlation_matrix(
        samples,
        X_FEATURE_NAMES,
        "Generated X correlation matrix",
        os.path.join(corr_dir, "generated_correlation_matrix.png"),
    )
    if writer is not None:
        writer.add_figure("validation/generated_correlation_matrix", fig_fake, global_step=epoch)
    plt.close(fig_fake)


def _plot_conditioned_feature_means(samples, X_test_cpu, Y_test_cpu, save_dir, epoch, writer=None):
    """
    Compare mean reconstructed features as a function of flavour.

    This is useful because the generated jets are conditioned on Y, and flavour
    is one of the physical context variables.
    """
    flavour_dir = os.path.join(save_dir, f"epoch_{epoch:04d}", "flavour_conditioning")
    _ensure_dir(flavour_dir)

    flavours = np.unique(Y_test_cpu[:, 4])
    flavours = np.sort(flavours)

    # Only keep reasonable flavour values.
    flavours = flavours[np.isfinite(flavours)]

    if len(flavours) == 0:
        return

    for feat_idx, feature_name in enumerate(X_FEATURE_NAMES):
        real_means = []
        fake_means = []
        used_flavours = []

        for flav in flavours:
            mask = Y_test_cpu[:, 4] == flav
            if np.sum(mask) < 5:
                continue

            used_flavours.append(flav)
            real_means.append(np.mean(X_test_cpu[mask, feat_idx]))
            fake_means.append(np.mean(samples[mask, feat_idx]))

        if len(used_flavours) == 0:
            continue

        fig, ax = plt.subplots(figsize=(7, 5))
        x = np.arange(len(used_flavours))

        ax.plot(x, real_means, marker="o", label="Real")
        ax.plot(x, fake_means, marker="o", label="Generated")

        ax.set_xticks(x)
        ax.set_xticklabels([str(int(f)) if float(f).is_integer() else str(f) for f in used_flavours])
        ax.set_xlabel("Flavour")
        ax.set_ylabel(f"Mean {feature_name}")
        ax.set_title(f"{feature_name} mean vs flavour")
        ax.legend()
        ax.grid(alpha=0.25)

        fig.tight_layout()
        fig.savefig(
            os.path.join(flavour_dir, f"{feat_idx:02d}_{feature_name}_vs_flavour.png"),
            dpi=200,
        )

        if writer is not None:
            writer.add_figure(
                f"validation_flavour/{feature_name}_vs_flavour",
                fig,
                global_step=epoch,
            )

        plt.close(fig)


def _plot_context_distributions(Y_test_cpu, save_dir, epoch, writer=None):
    """
    Plot distributions of the compact physical context variables.

    These are not compared against generated values because the generator is
    conditioned on them; the goal here is simply to inspect the conditioning set.
    """
    ctx_dir = os.path.join(save_dir, f"epoch_{epoch:04d}", "context")
    _ensure_dir(ctx_dir)

    for i, feature_name in enumerate(Y_FEATURE_NAMES):
        values = Y_test_cpu[:, i]
        values = values[np.isfinite(values)]

        if len(values) == 0:
            continue

        bins = 60
        hist_range = None

        if feature_name == "flavour":
            unique_vals = np.unique(values.astype(int))
            bins = np.arange(unique_vals.min() - 0.5, unique_vals.max() + 1.5, 1.0)
        elif "phi" in feature_name.lower():
            hist_range = (-np.pi, np.pi)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(values, bins=bins, range=hist_range, density=True, histtype="step", linewidth=2)
        ax.set_xlabel(feature_name)
        ax.set_ylabel("Density")
        ax.set_title(f"Context distribution: {feature_name}")
        ax.grid(alpha=0.25)

        fig.tight_layout()
        fig.savefig(os.path.join(ctx_dir, f"{i:02d}_{feature_name}.png"), dpi=200)

        if writer is not None:
            writer.add_figure(f"context/{feature_name}", fig, global_step=epoch)

        plt.close(fig)


# -----------------------------------------------------------------------------
# Main validation entry point
# -----------------------------------------------------------------------------

def validate(samples, X_test_cpu, Y_test_cpu, save_dir, epoch, writer=None):
    """
    Main validation function.

    Parameters
    ----------
    samples : np.ndarray
        Generated reconstructed features, shape (N, 16).
    X_test_cpu : np.ndarray
        Real reconstructed test features, shape (N, 16).
    Y_test_cpu : np.ndarray
        Compact physical context variables, shape (N, 8).
    save_dir : str
        Directory where validation outputs will be saved.
    epoch : int
        Current training epoch.
    writer : SummaryWriter or None
        TensorBoard writer. If provided, figures will also be logged.

    Output
    ------
    Creates:
      save_dir/
        epoch_XXXX/
          feature_hists/
          correlations/
          flavour_conditioning/
          context/
          summary_metrics.csv
    """
    # Basic shape checks.
    if samples.ndim != 2:
        raise ValueError(f"'samples' must be 2D, got shape {samples.shape}")
    if X_test_cpu.ndim != 2:
        raise ValueError(f"'X_test_cpu' must be 2D, got shape {X_test_cpu.shape}")
    if Y_test_cpu.ndim != 2:
        raise ValueError(f"'Y_test_cpu' must be 2D, got shape {Y_test_cpu.shape}")

    if samples.shape[1] != len(X_FEATURE_NAMES):
        raise ValueError(
            f"'samples' must have {len(X_FEATURE_NAMES)} columns, got {samples.shape[1]}"
        )
    if X_test_cpu.shape[1] != len(X_FEATURE_NAMES):
        raise ValueError(
            f"'X_test_cpu' must have {len(X_FEATURE_NAMES)} columns, got {X_test_cpu.shape[1]}"
        )
    if Y_test_cpu.shape[1] != len(Y_FEATURE_NAMES):
        raise ValueError(
            f"'Y_test_cpu' must have {len(Y_FEATURE_NAMES)} columns, got {Y_test_cpu.shape[1]}"
        )

    # Ensure lengths agree as much as possible.
    n = min(len(samples), len(X_test_cpu), len(Y_test_cpu))
    samples = samples[:n]
    X_test_cpu = X_test_cpu[:n]
    Y_test_cpu = Y_test_cpu[:n]

    epoch_dir = os.path.join(save_dir, f"epoch_{epoch:04d}")
    _ensure_dir(epoch_dir)

    print(f"[validation] samples shape   : {samples.shape}")
    print(f"[validation] X_test_cpu shape: {X_test_cpu.shape}")
    print(f"[validation] Y_test_cpu shape: {Y_test_cpu.shape}")
    print(f"[validation] saving to       : {epoch_dir}")

    # Save summary metrics.
    _save_summary_csv(
        samples=samples,
        X_test_cpu=X_test_cpu,
        save_path=os.path.join(epoch_dir, "summary_metrics.csv"),
    )

    # Plot all reconstructed-feature histograms.
    _plot_all_feature_histograms(
        samples=samples,
        X_test_cpu=X_test_cpu,
        save_dir=save_dir,
        epoch=epoch,
        writer=writer,
    )

    # Plot real and generated correlation matrices.
    _plot_real_vs_generated_correlations(
        samples=samples,
        X_test_cpu=X_test_cpu,
        save_dir=save_dir,
        epoch=epoch,
        writer=writer,
    )

    # Plot feature means vs flavour.
    _plot_conditioned_feature_means(
        samples=samples,
        X_test_cpu=X_test_cpu,
        Y_test_cpu=Y_test_cpu,
        save_dir=save_dir,
        epoch=epoch,
        writer=writer,
    )

    # Plot context distributions for inspection.
    _plot_context_distributions(
        Y_test_cpu=Y_test_cpu,
        save_dir=save_dir,
        epoch=epoch,
        writer=writer,
    )

    print("[validation] done.")
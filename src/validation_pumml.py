# validation_pumml.py
#
# Compares generated neutral-LV images against the true ones and
# shows the three conditioning channels for reference.
#
# Expected shapes:
#   samples          (N, 100)     generated neutral LV (flattened 10x10), physical pT
#   X_test_physical  (N, 100)     true    neutral LV (flattened 10x10), physical pT
#   imgs_test        (N, 3, 40, 40)  three input channels, physical pT

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

N_NEUTRAL = 10
N_CHARGED = 40


def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def _ks(x, y):
    x = np.sort(x[np.isfinite(x)])
    y = np.sort(y[np.isfinite(y)])
    if not len(x) or not len(y):
        return np.nan
    vals = np.sort(np.unique(np.concatenate([x, y])))
    cx   = np.searchsorted(x, vals, side="right") / len(x)
    cy   = np.searchsorted(y, vals, side="right") / len(y)
    return float(np.max(np.abs(cx - cy)))


def _show(ax, img, cmap, title):
    vmax = img.max() if img.max() > 0 else 1.0
    ax.imshow(img.T, origin="lower", cmap=cmap,
              norm=mcolors.PowerNorm(gamma=0.5, vmin=0, vmax=vmax), aspect="auto")
    ax.set_title(title, fontsize=8)
    ax.axis("off")



# Plot helpers

def _plot_mean_images(samples, X_test_physical, imgs_test, out_dir, epoch, writer):
    true_imgs = X_test_physical.reshape(-1, N_NEUTRAL, N_NEUTRAL)
    gen_imgs  = samples.reshape(-1, N_NEUTRAL, N_NEUTRAL)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    _show(axes[0], imgs_test[:, 0].mean(axis=0), "Blues",   "Mean: charged LV (context)")
    _show(axes[1], imgs_test[:, 1].mean(axis=0), "Greens",  "Mean: charged PU (context)")
    _show(axes[2], imgs_test[:, 2].mean(axis=0), "Reds",    "Mean: neutral all (context)")
    _show(axes[3], true_imgs.mean(axis=0),        "Purples", "Mean: neutral LV (true)")
    _show(axes[4], gen_imgs.mean(axis=0),          "Oranges", "Mean: neutral LV (generated)")
    fig.suptitle(f"Average jet images – epoch {epoch}", fontsize=11)
    plt.tight_layout()
    path = os.path.join(out_dir, f"mean_images_{epoch:04d}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if writer:
        writer.add_figure("pumml/mean_images", fig, global_step=epoch)
    plt.close(fig)


def _plot_pixel_hist(samples, X_test_physical, out_dir, epoch, writer):
    true_flat = X_test_physical.ravel()
    gen_flat  = samples.ravel()
    ks        = _ks(true_flat, gen_flat)

    bins = np.linspace(0, max(true_flat.max(), gen_flat.max()) * 1.05, 80)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(true_flat, bins=bins, histtype="step", lw=1.5, density=True, label="True")
    ax.hist(gen_flat,  bins=bins, histtype="step", lw=1.5, density=True, label="Generated")
    ax.set_yscale("log")
    ax.set_xlabel("Pixel pT (GeV)")
    ax.set_ylabel("Density")
    ax.set_title(f"Pixel pT – epoch {epoch}   KS={ks:.4f}")
    ax.legend(frameon=False)
    plt.tight_layout()
    path = os.path.join(out_dir, f"pixel_hist_{epoch:04d}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if writer:
        writer.add_figure("pumml/pixel_hist", fig, global_step=epoch)
        writer.add_scalar("pumml/pixel_KS",   ks,  global_step=epoch)
    plt.close(fig)


def _plot_jet_total_pt(samples, X_test_physical, out_dir, epoch, writer):
    true_sum = X_test_physical.sum(axis=1)
    gen_sum  = samples.sum(axis=1)
    ks       = _ks(true_sum, gen_sum)

    bins = np.linspace(0, max(true_sum.max(), gen_sum.max()) * 1.05, 80)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(true_sum, bins=bins, histtype="step", lw=1.5, density=True, label="True")
    ax.hist(gen_sum,  bins=bins, histtype="step", lw=1.5, density=True, label="Generated")
    ax.set_xlabel("Total neutral LV pT per jet (GeV)")
    ax.set_ylabel("Density")
    ax.set_title(f"Total neutral LV pT – epoch {epoch}   KS={ks:.4f}")
    ax.legend(frameon=False)
    plt.tight_layout()
    path = os.path.join(out_dir, f"jet_total_pt_{epoch:04d}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if writer:
        writer.add_figure("pumml/jet_total_pt", fig, global_step=epoch)
        writer.add_scalar("pumml/total_pt_KS",  ks,  global_step=epoch)
    plt.close(fig)


def _plot_examples(samples, X_test_physical, imgs_test, out_dir, epoch, n=6):
    true_imgs = X_test_physical.reshape(-1, N_NEUTRAL, N_NEUTRAL)
    gen_imgs  = samples.reshape(-1, N_NEUTRAL, N_NEUTRAL)
    idx       = np.random.choice(len(samples), size=min(n, len(samples)), replace=False)

    fig, axes = plt.subplots(len(idx), 5, figsize=(18, 3.5 * len(idx)))
    titles = ["charged LV", "charged PU", "neutral all", "neutral LV (true)", "neutral LV (gen)"]
    cmaps  = ["Blues", "Greens", "Reds", "Purples", "Oranges"]

    for row, i in enumerate(idx):
        panels = [imgs_test[i, 0], imgs_test[i, 1], imgs_test[i, 2],
                  true_imgs[i], gen_imgs[i]]
        for col, (img, cmap, title) in enumerate(zip(panels, cmaps, titles)):
            ax = axes[row, col]
            _show(ax, img, cmap, title if row == 0 else "")
            if col == 0:
                ax.set_ylabel(f"jet {i}", fontsize=8)

    fig.suptitle(f"Random examples – epoch {epoch}", fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"examples_{epoch:04d}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_csv(samples, X_test_physical, out_dir, epoch):
    with open(os.path.join(out_dir, "pixel_ks.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "pixel", "mean_true", "mean_gen", "ks"])
        for p in range(samples.shape[1]):
            w.writerow([epoch, p,
                        float(np.mean(X_test_physical[:, p])),
                        float(np.mean(samples[:, p])),
                        _ks(X_test_physical[:, p], samples[:, p])])



# Main entry point
def validate_pumml(samples, X_test_physical, imgs_test, save_dir, epoch, writer=None):
    """
    Parameters
    ----------
    samples          : (N, 100)       generated neutral LV, physical pT
    X_test_physical  : (N, 100)       true neutral LV, physical pT
    imgs_test        : (N, 3, 40, 40) three input channels, physical pT
    save_dir         : str
    epoch            : int
    writer           : SummaryWriter or None
    """
    n = min(len(samples), len(X_test_physical), len(imgs_test))
    samples, X_test_physical, imgs_test = samples[:n], X_test_physical[:n], imgs_test[:n]

    out_dir = os.path.join(save_dir, f"pumml_val_{epoch:04d}")
    _ensure_dir(out_dir)
    print(f"[validation_pumml] epoch {epoch}  N={n}  -> {out_dir}")

    _plot_mean_images(samples, X_test_physical, imgs_test, out_dir, epoch, writer)
    _plot_pixel_hist(samples, X_test_physical, out_dir, epoch, writer)
    _plot_jet_total_pt(samples, X_test_physical, out_dir, epoch, writer)
    _plot_examples(samples, X_test_physical, imgs_test, out_dir, epoch)
    _save_csv(samples, X_test_physical, out_dir, epoch)

    print("[validation_pumml] done.")
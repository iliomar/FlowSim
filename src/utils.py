# utils.py
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import stats

from scipy.stats import ks_2samp
import corner
from matplotlib import pyplot as plt
from matplotlib import lines as mlines
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

from scipy.linalg import sqrtm

from sklearn.metrics.pairwise import rbf_kernel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import hamming_loss
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from scipy import interpolate
from scipy.integrate import simpson, trapezoid

from matplotlib import rc
from matplotlib.ticker import FormatStrFormatter

# rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})


def plot_1d_hist(
    flash,
    reco,
    label,
    title,
    rangeHist=None,
    bins=100,
    ratioPlotBounds=(-2, 2),
    logScale=True,
    legendLoc="upper right",
    ymax=None,
):
    # change matplotlib style
    # plt.style.use("fivethirtyeight")
    # if not logScale:
    fig = plt.figure(figsize=(5, 5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1], sharex=ax1)

    plt.subplots_adjust(hspace=0.0, wspace=0.0)

    ax1.tick_params(which="both", labelbottom=False, direction="in")
    ax1.minorticks_on()
    ax2.tick_params(which="both", direction="in", top=True)
    # ax2.minorticks_on()

    ax2.set_xlabel(label, loc="right", fontsize=14)
    # fig.suptitle(f"{title}", fontsize=20)

    # bins = 100
    # if (
    #     (label == "recoNConstituents")
    #     | (label == "nSV")
    #     | (label == "ncharged")
    #     | (label == "nneutral")
    # ):
    #     bins = np.arange(-0.5, 80.5, 1)

    # Linear scale plot
    _, rangeR, _ = ax1.hist(
        reco, histtype="step", color="dodgerblue", lw=1.5, bins=bins, label="Target"
    )

    saturated_samples = np.where(flash < np.min(rangeR), np.min(rangeR), flash)
    saturated_samples = np.where(
        saturated_samples > np.max(rangeR), np.max(rangeR), saturated_samples
    )
    ax1.hist(
        saturated_samples,
        histtype="step",
        lw=1.5,
        color="tomato",
        range=[np.min(rangeR), np.max(rangeR)],
        bins=bins,
        label=f"Flow",
    )
    ax1.legend(frameon=False, loc=legendLoc)
    ax1.set_ylabel("Counts", loc="top", fontsize=14)

    if logScale:
        ax1.set_yscale("log")

    if rangeHist is not None:
        ax1.set_xlim(rangeHist[0], rangeHist[1])

    # Ratio plot for linear scale
    hist_reco, bins_reco = np.histogram(
        reco, bins=bins, range=[np.min(rangeR), np.max(rangeR)]
    )
    hist_flash, bins_flash = np.histogram(
        saturated_samples, bins=bins, range=[np.min(rangeR), np.max(rangeR)]
    )

    # compute ks
    ks = ks_2samp(hist_reco, hist_flash)
    ks, pvalue = ks[0], ks[1]

    # Compute the error on the ratio
    ratio_err = np.sqrt(
        (np.sqrt(hist_flash) / hist_flash) ** 2 + (np.sqrt(hist_reco) / hist_reco) ** 2
    )
    ratio_err = ratio_err * (hist_flash / hist_reco)

    ratio = np.where(hist_reco > 0, hist_flash / hist_reco, 0)
    ratio_err = np.where(hist_reco > 0, ratio_err, 0)
    # ax2.scatter(bins_reco[:-1], ratio, marker=".", color="black")
    bin_centers = (bins_reco[:-1] + bins_reco[1:]) / 2

    ax2.errorbar(
        bin_centers,
        ratio,
        yerr=ratio_err,
        fmt=".",
        color="black",
        ms=2,
        elinewidth=1,
    )
    ax2.set_ylabel("Flow/Target", fontsize=14)
    if rangeHist is not None:
        ax2.set_xlim(rangeHist[0], rangeHist[1])
    ax2.set_ylim(*ratioPlotBounds)
    # horizontal line at 1
    ax2.axhline(y=1, color="black", linestyle="--", alpha=0.5)
    # ax2.text(
    #     s=f"KS: {ks:.3f}",
    #     x=ax2.get_xlim()[0] + (ax2.get_xlim()[1] - ax2.get_xlim()[0]) * 0.05,
    #     y=ax2.get_ylim()[0] + (ax2.get_ylim()[1] - ax2.get_ylim()[0]) * 0.8,
    # )
    # align the y-labels
    fig.align_ylabels([ax1, ax2])

    return fig


def plot_1d_hist_with_condition(
    targets,
    labels,
    ranges,
    df_flash,
    df_reco,
    cond_var,
    cond_vals,
    names,
    colors,
    savepath,
    logScale=True,
    ymax=None,
    density=True,
):
    """
    Plot 1D histogram of flash vs reco, splitting the histograms based by a condition
    we specify a condition variable (in pd df) and a list of values for that variable
    """
    samples = df_flash
    reco = df_reco
    for target, rangeR, label in zip(targets, ranges, labels):
        fig, axs = plt.subplots(1, 2, figsize=(9, 4.5), tight_layout=True)
        # put the ticks on the inside
        axs[0].tick_params(direction="in")
        axs[1].tick_params(direction="in")

        axs[0].set_xlabel(label, fontsize=14)
        axs[1].set_xlabel(label, fontsize=14)

        axs[1].set_yscale("log")

        inf = rangeR[0]
        sup = rangeR[1]

        for cond, color, name in zip(cond_vals, colors, names):
            mask = df_flash[cond_var].values == cond
            full = reco[target].values
            full = full[mask]
            full = full[~np.isnan(full)]
            full = np.where(full > sup, sup, full)
            full = np.where(full < inf, inf, full)

            flash = samples[target].values
            flash = flash[mask]
            flash = flash[~np.isnan(flash)]
            flash = np.where(flash > sup, sup, flash)
            flash = np.where(flash < inf, inf, flash)

            axs[0].hist(
                full,
                bins=50,
                range=rangeR,
                histtype="step",
                ls="--",
                color=color,
                density=density,
            )
            axs[0].hist(
                flash,
                bins=50,
                range=rangeR,
                histtype="step",
                label=f"{name}",
                color=color,
                density=density,
            )
            axs[1].hist(
                flash,
                bins=50,
                range=rangeR,
                histtype="step",
                label=f"{name}",
                color=color,
                density=density,
            )

        axs[0].set_ylabel("Counts", fontsize=14)
        if density:
            axs[0].set_ylabel("Normalized counts", fontsize=14)

        # add a black box to the legend named FlashSim
        m = Patch(color="black", ls="--", label="Target", fill=False)
        mm = Patch(color="black", label="Flow", fill=False)

        handles, _ = axs[0].get_legend_handles_labels()
        handles.append(m)
        handles.append(mm)
        axs[0].legend(frameon=False, handles=handles, loc="upper right")

        plt.savefig(f"{savepath}/{target}_1d_cond.png")

    return fig


def plot_1d_hist_with_condition_figure(
    targets,
    labels,
    ranges,
    df_flash,
    df_reco,
    cond_var,
    cond_vals,
    names,
    colors,
    savepath,
    logScale=True,
    ymax=None,
    density=True,
):
    """
    Plot 1D histogram of flash vs reco, splitting the histograms based by a condition
    we specify a condition variable (in pd df) and a list of values for that variable
    """
    samples = df_flash
    reco = df_reco
    lims = [(0.75, 1.25), (0.89, 1.11)]
    logylims = [None, (2e-2, 3e1)]
    texts = [1.15, 1.07]
    for target, rangeR, label, lim, logylim, texty in zip(
        targets, ranges, labels, lims, logylims, texts
    ):
        fig, axs = plt.subplots(
            2,
            1,
            figsize=(5, 5),
            tight_layout=False,
            sharex=True,
            height_ratios=[3, 1],
            gridspec_kw={"wspace": 0, "hspace": 0},
        )
        plt.subplots_adjust(hspace=0.0, wspace=0.0)

        axs[0].tick_params(which="both", labelbottom=False, direction="in")
        axs[0].minorticks_on()
        axs[1].tick_params(which="both", direction="in", top=True)
        # axs[1].minorticks_on()

        # axs[0].set_xlabel(label, fontsize=16)
        axs[1].set_xlabel(label, loc="right", fontsize=14)
        axs[0].set_yscale("log")
        if logylim is not None:
            axs[0].set_ylim(*logylim)
        # axs[0].set_ylim(*logylim)

        inf = rangeR[0]
        sup = rangeR[1]

        for cond, color, name in zip(cond_vals, colors, names):
            mask = df_flash[cond_var].values == cond
            full = reco[target].values
            full = full[mask]
            full = full[~np.isnan(full)]
            full = np.where(full > sup, sup, full)
            full = np.where(full < inf, inf, full)

            flash = samples[target].values
            flash = flash[mask]
            flash = flash[~np.isnan(flash)]
            flash = np.where(flash > sup, sup, flash)
            flash = np.where(flash < inf, inf, flash)

            axs[0].hist(
                full,
                bins=50,
                range=rangeR,
                histtype="step",
                ls="--",
                lw=1.5,
                color=color,
                density=density,
            )
            axs[0].hist(
                flash,
                bins=50,
                range=rangeR,
                histtype="step",
                lw=1.5,
                label=f"{name}",
                color=color,
                density=density,
            )
            # axs[1].hist(
            #     flash,
            #     bins=50,
            #     range=rangeR,
            #     histtype="step",
            #     label=f"{name}",
            #     lw=1.5,
            #     color=color,
            #     density=density
            # )

        # axs[0].set_ylabel("Counts", fontsize=16)
        if density:
            axs[0].set_ylabel("Normalized counts", loc="top", fontsize=14)

        full = reco[target].values
        flash = samples[target].values
        hist_full, bins_full = np.histogram(full, bins=50, range=rangeR)
        hist_flash, bins_flash = np.histogram(flash, bins=50, range=rangeR)

        # Compute the error on the ratio
        ratio_err = np.sqrt(
            (np.sqrt(hist_flash) / hist_flash) ** 2
            + (np.sqrt(hist_full) / hist_full) ** 2
        )
        ratio_err = ratio_err * (hist_flash / hist_full)

        ratio = np.where(hist_full > 0, hist_flash / hist_full, 0)
        ratio_err = np.where(hist_full > 0, ratio_err, 0)
        # axs[1].scatter(bins_full[:-1], ratio, marker=".", color="black")
        axs[1].errorbar(
            bins_full[:-1],
            ratio,
            yerr=ratio_err,
            fmt=".",
            color="black",
            ms=2,
            elinewidth=1,
        )

        # Horizontal line at 1
        axs[1].axhline(y=1, color="black", linestyle="--", alpha=0.5)
        axs[1].set_ylim(*lim)
        axs[1].set_ylabel("Flow/Target", fontsize=14)
        axs[1].text(s="All flavours", x=0.0, y=texty)

        # add a black box to the legend named FlashSim
        m = Patch(color="black", ls="--", lw=1.5, label="Target", fill=False)
        mm = Patch(color="black", label="Flow", lw=1.5, fill=False)

        handles, _ = axs[0].get_legend_handles_labels()
        handles.append(m)
        handles.append(mm)
        axs[0].legend(frameon=False, handles=handles, loc="upper center")

        plt.savefig(f"{savepath}/{target}_1d_cond.pdf", bbox_inches="tight")

    return


def make_3corner(reco, samples, samples1, labels, title, ranges=None, *args, **kwargs):
    blue_line = Patch(color="dodgerblue", label="udsg", fill=False)
    if samples is not None:
        red_line = Patch(color="tomato", label="c", fill=False)
        green_line = Patch(color="limegreen", label="b", fill=False)

    from matplotlib import colors

    def _forward(x):
        print(x)
        return np.sqrt(x)

    def _inverse(x):
        print(f"inv: {x}")
        return x

    norm = colors.FuncNorm((_forward, _inverse), vmin=73.0, vmax=2507.0)
    # norm = colors.BoundaryNorm([0, 0.51, 0.91, 1], 5, extend="both")
    norm = colors.PowerNorm(0.5, vmin=0, vmax=2507.0)
    print(norm.vmax, norm.vmin)
    if samples is not None:
        fig = corner.corner(
            reco,
            range=ranges,
            # fig=fig,
            color="tomato",
            levels=[0.5, 0.9, 0.99],
            hist_bin_factor=2,
            scale_hist=False,
            plot_datapoints=False,
            plot_density=False,
            hist_kwargs={"ls": "-"},
            contour_kwargs={
                "linestyles": "-",
                "colors": None,
                "cmap": "Reds",
                "norm": norm,
            },
            label_kwargs={"fontsize": 16},
            smooth=0.1,
            *args,
            **kwargs,
        )
    if samples1 is not None:
        corner.corner(
            samples,
            range=ranges,
            fig=fig,
            color="limegreen",
            levels=[0.5, 0.9, 0.99],
            hist_bin_factor=2,
            scale_hist=False,
            plot_density=False,
            hist_kwargs={"ls": "-"},
            contour_kwargs={
                "linestyles": "-",
                "colors": None,
                "cmap": "Greens",
                "norm": norm,
            },
            plot_datapoints=False,
            label_kwargs={"fontsize": 16},
            smooth=0.1,
            *args,
            **kwargs,
        )
    if title is not None:
        plt.suptitle(title, fontsize=16)
    corner.corner(
        samples1,
        fig=fig,
        range=ranges,
        labels=labels,
        color="dodgerblue",
        levels=[0.5, 0.9, 0.99],
        hist_bin_factor=2,
        scale_hist=False,
        plot_datapoints=False,
        plot_density=False,
        hist_kwargs={"ls": "-"},
        contour_kwargs={
            "linestyles": "-",
            "colors": None,
            "cmap": "Blues",
            "norm": norm,
        },
        smooth=0.1,
        label_kwargs={"fontsize": 16},
        *args,
        **kwargs,
    )

    # add legend
    fig.legend(
        handles=[blue_line, red_line, green_line],
        loc="upper right",
        frameon=False,
        bbox_to_anchor=(0.85, 0.85),
        fontsize=14,
    )
    for i, ax in enumerate(fig.get_axes()):
        top = False
        right = False
        if i == 2:
            top = True
            right = True
        if i == 3:
            ax.set_ylim(0, 30000)
        ax.tick_params(which="both", direction="in", top=top, right=right)
        ax.minorticks_on()

    return fig


def recopt_vs_flavour(pt_b, pt_nonb):
    fig, ax = plt.subplots(figsize=(5, 5), tight_layout=False)
    ax.hist(
        pt_nonb,
        bins=50,
        range=(0, 2),
        density=True,
        histtype="step",
        color="coral",
        label="non b",
        lw=1.5,
    )
    ax.axvline(
        np.mean(pt_nonb),
        color="coral",
        linestyle="--",
        # label=r"$\mu_{non-b}$",
        lw=1.5,
        alpha=0.6,
    )
    ax.hist(
        pt_b,
        bins=50,
        range=(0, 2),
        density=True,
        histtype="step",
        color="limegreen",
        label="b",
        lw=1.5,
    )
    ax.axvline(
        np.mean(pt_b),
        color="limegreen",
        linestyle="--",
        # label=r"$\mu_b$",
        lw=1.5,
        alpha=0.6,
    )

    ax.set_xlabel(r"$p_T^{reco}/p_T^{gen}$", loc="right", fontsize=14)
    ax.set_ylabel("Normalized counts", loc="top", fontsize=14)
    ax.tick_params(axis="both", which="both", direction="in")
    ax.minorticks_on()
    ax.set_xlim(0, 2)

    # vertical line at the two mean values
    ax.legend(frameon=False, loc="upper right")

    from matplotlib.lines import Line2D

    means = Line2D(
        [0], [0], color="k", linestyle="--", lw=1.5, alpha=0.6, label="Mean values"
    )

    handles, _ = ax.get_legend_handles_labels()
    handles.append(means)
    ax.legend(handles=handles, frameon=False, loc="upper right")
    fig.suptitle("$p_T$ reconstruction for jets", fontsize=14)

    return fig


def areas_between_rocs(tpr_real, fpr_real, tpr_flash, fpr_flash, x_lim=0.2):
    if x_lim:
        tpr_real_mask = tpr_real > x_lim
        fpr_real = fpr_real[tpr_real_mask]
        tpr_real = tpr_real[tpr_real_mask]

        tpr_flash_mask = tpr_flash > x_lim
        fpr_flash = fpr_flash[tpr_flash_mask]
        tpr_flash = tpr_flash[tpr_flash_mask]

    # apply log to tpr
    fpr_real_log = np.where(fpr_real < 1e-6, 0, np.log(fpr_real))
    # Step 1: Choose the curve with more data points as the base for x-values (in this case fpr_blue)
    # Step 2: Interpolate the curve with fewer data points to these x-values

    # tpr_flash += np.linspace(0, 1e-6, len(tpr_flash)) # NOT NEEDED IF WE USE NEAREST INTERPOLATION

    interpolator = interpolate.interp1d(
        tpr_flash,
        fpr_flash,
        kind="nearest",
        bounds_error=False,
        fill_value="extrapolate",
    )
    fpr_flash_interpolated = interpolator(tpr_real)
    fpr_flash_interpolated_log = np.where(
        fpr_flash_interpolated < 1e-6, 0, np.log(fpr_flash_interpolated)
    )

    # Calculate the absolute differences between the curves
    differences = np.abs(fpr_real_log - fpr_flash_interpolated_log)

    # Integrate the differences using Simpson's rule
    area = trapezoid(differences, tpr_real)

    return area


def roc_curve_figure(
    y_true_target,
    y_pred_target,
    y_true_flow,
    y_pred_flow,
    title,
    perturb=False,
    shade=False,
):
    from sklearn.metrics import roc_curve, auc

    fpr_target, tpr_target, _ = roc_curve(y_true_target, y_pred_target)
    roc_auc_target = auc(fpr_target, tpr_target)

    fpr_flow, tpr_flow, _ = roc_curve(y_true_flow, y_pred_flow)
    roc_auc_flow = auc(fpr_flow, tpr_flow)

    if perturb:
        tpr_target_perturbed_minus = 1 - (1 - tpr_target) * 1.2
        tpr_target_perturbed_plus = 1 - (1 - tpr_target) * 0.8

        interpolator = interpolate.interp1d(
            tpr_target,
            fpr_target,
            kind="nearest",
            bounds_error=False,
            fill_value="extrapolate",
        )
        fpr_target_perturbed_minus = interpolator(tpr_target_perturbed_minus)
        fpr_target_perturbed_plus = interpolator(tpr_target_perturbed_plus)

    fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
    ax.plot(
        tpr_target,
        fpr_target,
        color="dodgerblue",
        lw=1.5,
        label=f"Target",
    )

    if perturb:
        # # shade the area between the perturbed curves
        # # the filling is a grid pattern
        abc_perturbed = areas_between_rocs(
            tpr_target, fpr_target, tpr_target_perturbed_minus, fpr_target, x_lim=0.2
        )

        ax.fill_between(
            tpr_target,
            fpr_target_perturbed_minus,
            fpr_target_perturbed_plus,
            alpha=0.4,
            label="Typ. data vs simulation discrepancy"
            + "\n"
            + f"at LHC, ABC: {abc_perturbed:.3f}",
            color="dodgerblue",
        )

    abc = areas_between_rocs(tpr_target, fpr_target, tpr_flow, fpr_flow, x_lim=0.2)

    ax.plot(
        tpr_flow,
        fpr_flow,
        color="tomato",
        lw=1.5,
        label=f"Flow, ABC: {abc:.3f}",
    )
    if shade:
        # cast the two curves to the same length
        interpolator = interpolate.interp1d(
            tpr_flow,
            fpr_flow,
            kind="nearest",
            bounds_error=False,
            fill_value="extrapolate",
        )
        fpr_flow_interpolated = interpolator(tpr_target)
        ax.fill_between(
            tpr_target,
            fpr_target,
            fpr_flow_interpolated,
            color="grey",
            alpha=0.5,
            label="Area between curves",
        )

    ax.tick_params(axis="both", which="both", direction="in")
    ax.minorticks_on()
    ax.set_xlabel("True Positive Rate", loc="right", fontsize=14)
    ax.set_ylabel("False Positive Rate", loc="top", fontsize=14)

    ax.set_xlim(0.2, 1)
    ax.set_ylim(1e-4, 1.05)
    ax.set_yscale("log")
    ax.grid(True, which="both", axis="both", alpha=0.5, color="darkgrey", ls="--")
    ax.axvline(x=0.2, color="black", linestyle="--", lw=1.5, alpha=0.6)

    leg = ax.legend(
        frameon=False,
        loc="upper center",
    )

    leg._legend_box.align = "left"
    ax.set_title(title, fontsize=14)

    return fig, ax


def c2st(target, flow, clf, loss=hamming_loss, bootstraps=300):
    """
    Perform Classifier Two Sample Test (C2ST) [1].

    This test estimates if a target is predictable from features by comparing the loss of a classifier learning
    the true target with the distribution of losses of classifiers learning a random target with the same average.

    The null hypothesis is that the target is independent of the features - therefore the loss a classifier learning
    to predict the target should not be different from the one of a classifier learning independent, random noise.

    Input:
        - `target` : (n,m) matrix of target features
        - `flow` : (n,m) matrix of flow features
        - `clf` : instance of sklearn compatible classifier (default: `LogisticRegression`)
        - `loss` : sklearn compatible loss function (default: `hamming_loss`)
        - `bootstraps` : number of resamples for generating the loss scores under the null hypothesis

    Return: (
        loss value of classifier predicting `y`,
        loss values of bootstraped random targets,
        p-value of the test
    )

    Usage:
    >>> emp_loss, random_losses, pvalue = c2st(X, y)

    Plotting H0 and target loss:
    >>>bins, _, __ = plt.hist(random_losses)
    >>>med = np.median(random_losses)
    >>>plt.plot((med,med),(0, max(bins)), 'b')
    >>>plt.plot((emp_loss,emp_loss),(0, max(bins)), 'r--')

    [1] Lopez-Paz, D., & Oquab, M. (2016). Revisiting classifier two-sample tests. arXiv preprint arXiv:1610.06545.
    """
    if clf == "random_forest":
        clf = RandomForestClassifier(verbose=1)
    elif clf == "hist_gradient_boosting":
        clf = HistGradientBoostingClassifier(verbose=1)
    else:
        raise ValueError("No classifier specified!")

    # create labels 1 for target and 0 for flow
    y = np.concatenate((np.ones(target.shape[0]), np.zeros(flow.shape[0])))
    # concatenate target and flow
    X = np.concatenate((target, flow))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, shuffle=True, train_size=0.5
    )
    y_pred = clf.fit(X_train, y_train).predict(X_test)
    emp_loss = loss(y_test, y_pred)
    # bs_losses = []
    # y_bar = np.mean(y)
    # for b in range(bootstraps + 1):
    #     y_random = np.random.binomial(1, y_bar, size=y.shape[0])
    #     X_train, X_test, y_train, y_test = train_test_split(X, y_random)
    #     y_pred_bs = clf.fit(X_train, y_train).predict(X_test)
    #     bs_losses += [loss(y_test, y_pred_bs)]
    # pc = stats.percentileofscore(sorted(bs_losses), emp_loss) / 100.0
    # pvalue = pc if pc < y_bar else 1 - pc
    return emp_loss  # , np.array(bs_losses), pvalue


def compute_mmd(X, Y, gamma=None):
    K_XX = rbf_kernel(X, X, gamma=gamma)
    K_YY = rbf_kernel(Y, Y, gamma=gamma)
    K_XY = rbf_kernel(X, Y, gamma=gamma)

    mmd = np.mean(K_XX) + np.mean(K_YY) - 2 * np.mean(K_XY)
    return mmd


def calculate_fgd(data_real, data_fake):
    mu_real = np.mean(data_real, axis=0)
    mu_fake = np.mean(data_fake, axis=0)
    cov_real = np.cov(data_real, rowvar=False)
    cov_fake = np.cov(data_fake, rowvar=False)

    diff_mean = np.sum((mu_real - mu_fake) ** 2)
    cov_mean = sqrtm(cov_real.dot(cov_fake))

    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real

    fgd = diff_mean + np.trace(cov_real + cov_fake - 2 * cov_mean)
    return fgd


def covariance_matching(data_real, data_fake):
    cov_real = np.cov(data_real, rowvar=False)
    cov_fake = np.cov(data_fake, rowvar=False)
    return np.linalg.norm(cov_real - cov_fake, "fro")


def cdf(x, y):
    return np.tanh((x * y / 10) ** 1.2)


def profile(
    datasets,
    labels,
    ix,
    iy,
    nbins,
    save_dir,
    filename,
    logScale=True,
    ymax=None,
    text=False,
):
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax.set_ylim(0.1, 0.4)
    ax2.set_ylim(0.9, 1.3)

    ax.set_xlabel(r"Generator $p_T$ [GeV]", fontsize=14)
    ax2.set_xlabel(r"Generator $p_T$ [GeV]", fontsize=14)

    if logScale:
        ax.set_xscale("log")
        ax2.set_xscale("log")

    ymin = datasets[0][:, iy].min()
    if ymax is None:
        ymax = datasets[0][:, iy].max()

    if logScale:
        bin_edges = np.logspace(np.log10(ymin), np.log10(ymax), nbins + 1)
    else:
        bin_edges = np.linspace(ymin, ymax, nbins + 1)

    means_list = []
    stds_list = []
    for i, dataset in enumerate(datasets):
        means, edges, _ = stats.binned_statistic(
            dataset[:, iy], dataset[:, ix], statistic="mean", bins=bin_edges
        )
        stds, _, _ = stats.binned_statistic(
            dataset[:, iy], dataset[:, ix], statistic="std", bins=bin_edges
        )

        bin_centers = (edges[:-1] + edges[1:]) / 2
        ax.errorbar(bin_centers, stds, fmt="o", label=labels[i])
        ax2.errorbar(bin_centers, means, fmt="o", label=labels[i])

        means_list.append(means)
        stds_list.append(stds)

    ks_mean = ks_std = (None, None)
    if len(datasets) == 2:
        ks_mean = ks_2samp(means_list[0], means_list[1])
        ks_std = ks_2samp(stds_list[0], stds_list[1])

    if text:
        ax.text(
            0.05,
            0.95,
            f"KS test on mean: {ks_mean[0]:.3f}, p-value: {ks_mean[1]:.3f}\nKS test on std: {ks_std[0]:.3f}, p-value: {ks_std[1]:.3f}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    ax.legend(labels=labels)
    ax.set_ylabel(r"std")
    ax2.set_ylabel(r"mean")
    # fig.suptitle(rf"Profile histogram (std and mean)", fontsize=20)
    fig.savefig(os.path.join(save_dir, filename))

    return fig, ks_mean, ks_std


def mean_std(x):
    # Return the standard deviation of the sample mean
    return np.std(x) / np.sqrt(len(x))


def std_std(x):
    # Return the standard deviation of the sample standard deviation
    # return np.sqrt(2) * np.var(x) / np.sqrt((len(x)-1))
    from scipy.stats import moment

    n = len(x)
    if n < 3:
        res = 0
    else:
        res = np.sqrt(
            (1.0 / n) * (moment(x, moment=4) - (n - 3) / (n - 1) * np.var(x) ** 2)
        )
    return res


def profile_figure(
    datasets,
    labels,
    ix,
    iy,
    nbins,
    save_dir,
    filename,
    logScale=True,
    ymax=None,
    text=False,
    flow=True,
):
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6, 3), tight_layout=True)
    ax2.set_ylim(0.15, 0.27)
    ax.set_ylim(0.97, 1.18)

    ax.set_xlim(1e1, 1e3)
    ax2.set_xlim(1e1, 1e3)

    ax2.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    # fig.suptitle(rf"Profile histogram")

    # ax.set_xlabel(r"Generator $p_T$ [GeV]")
    ax.set_xlabel(r"Generator $p_T$ [GeV]", loc="right", fontsize=14)
    ax2.set_xlabel(r"Generator $p_T$ [GeV]", loc="right", fontsize=14)
    ax2.set_xlim(1e1, 1e3)
    plt.subplots_adjust(hspace=0.08, wspace=0.0, top=0.95, bottom=0.1)

    ax.tick_params(which="both", labelbottom=True, direction="in")
    ax.minorticks_on()
    ax2.tick_params(which="both", direction="in", top=False)
    ax2.minorticks_on()

    if logScale:
        ax.set_xscale("log")
        ax2.set_xscale("log")

    ymin = datasets[0][:, iy].min()
    if ymax is None:
        ymax = datasets[0][:, iy].max()

    if logScale:
        bin_edges = np.logspace(np.log10(ymin), np.log10(ymax), nbins + 1)
    else:
        bin_edges = np.linspace(ymin, ymax, nbins + 1)

    means_list = []
    stds_list = []
    #! Temporary
    colors = ["tomato", "dodgerblue"]
    fmts = ["o", "^"]
    colors = ["dodgerblue", "tomato"]
    fmts = ["^", "o"]
    # mss = [7, 4.5]
    mss = [2, 2]

    for i, dataset in enumerate(datasets):
        means, edges, _ = stats.binned_statistic(
            dataset[:, iy], dataset[:, ix], statistic="mean", bins=bin_edges
        )
        stds, _, _ = stats.binned_statistic(
            dataset[:, iy], dataset[:, ix], statistic="std", bins=bin_edges
        )
        means_err, _, _ = stats.binned_statistic(
            dataset[:, iy], dataset[:, ix], statistic=mean_std, bins=bin_edges
        )
        stds_err, _, _ = stats.binned_statistic(
            dataset[:, iy], dataset[:, ix], statistic=std_std, bins=bin_edges
        )

        bin_centers = (edges[:-1] + edges[1:]) / 2
        ax2.errorbar(
            bin_centers,
            stds,
            yerr=stds_err,
            fmt=fmts[i],
            color=colors[i],
            ms=mss[i],
            label=labels[i],
            elinewidth=1,
        )
        ax.errorbar(
            bin_centers,
            means,
            yerr=means_err,
            fmt=fmts[i],
            color=colors[i],
            ms=mss[i],
            label=labels[i],
            elinewidth=1,
        )

        means_list.append(means)
        stds_list.append(stds)

        if flow is False:
            break

    ks_mean = ks_std = (None, None)
    if len(datasets) == 2 and flow:
        ks_mean = ks_2samp(means_list[0], means_list[1])
        ks_std = ks_2samp(stds_list[0], stds_list[1])

    if text:
        ax.text(
            0.05,
            0.95,
            f"KS test on mean: {ks_mean[0]:.3f}, p-value: {ks_mean[1]:.3f}\nKS test on std: {ks_std[0]:.3f}, p-value: {ks_std[1]:.3f}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    ax.legend(labels=labels, frameon=False)
    ax2.legend(labels=labels, frameon=False)
    ax2.set_ylabel(r"Std. Dev. (Resolution)", fontsize=14)
    ax.set_ylabel(r"Mean (Response)", fontsize=14)
    # fig.suptitle(rf"Profile histogram (std and mean)", fontsize=20)
    fig.savefig(os.path.join(save_dir, filename), bbox_inches="tight")

    return fig, ks_mean, ks_std


def make_corner(reco, samples, labels, title, ranges=None, *args, **kwargs):
    blue_line = Patch(color="dodgerblue", label="Target", fill=False)
    if samples is not None:
        red_line = Patch(color="tomato", label="Flow", fill=False)
    fig = corner.corner(
        reco,
        range=ranges,
        labels=labels,
        color="dodgerblue",
        levels=[0.5, 0.9, 0.99],
        hist_bin_factor=3,
        max_n_ticks=5,
        scale_hist=False,
        plot_datapoints=False,
        plot_density=True,
        hist_kwargs={"ls": "--"},
        contour_kwargs={"linestyles": "--"},
        label_kwargs={"fontsize": 18},
        *args,
        **kwargs,
    )
    if samples is not None:
        corner.corner(
            samples,
            range=ranges,
            fig=fig,
            color="tomato",
            levels=[0.5, 0.9, 0.99],
            hist_bin_factor=3,
            scale_hist=False,
            plot_density=True,
            plot_datapoints=False,
            max_n_ticks=5,
            label_kwargs={"fontsize": 18},
            *args,
            **kwargs,
        )
    if title is not None:
        plt.suptitle(title, fontsize=18)
    # add legend
    fig.legend(
        handles=[blue_line, red_line],
        loc="upper right",
        frameon=False,
        fontsize=25,
        bbox_to_anchor=(0.85, 0.85),
    )
    for i, ax in enumerate(fig.get_axes()):
        top = True
        right = True
        if i in [0, 6, 12, 18, 24]:
            top = False
            right = False
        ax.tick_params(axis="both", which="both", direction="in", top=top, right=right)
        ax.minorticks_on()
    return fig

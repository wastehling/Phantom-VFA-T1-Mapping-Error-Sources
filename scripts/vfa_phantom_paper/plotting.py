import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

mpl.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

FIGWIDTH_DOUBLE = 7.0
FIGHEIGHT_DOUBLE = 3.5
FONTSIZE = 14


def compute_overall_errors(df_fitted_values):
    """One row per study with RMSE, bias (mean and std of the absolute error) and mean relative error."""
    return df_fitted_values.groupby("study_name").apply(
        lambda g: pd.Series({
            "bias_mean": g["abs_err_T1"].mean(),
            "bias_std": g["abs_err_T1"].std(),
            "rmse_T1": np.sqrt(np.mean(g["abs_err_T1"] ** 2)),
            "rel_err_mean": g["rel_err_T1"].mean(),
        }), include_groups=False
    ).reset_index()


def pooled_wcv(df):
    """Root mean square of the per-vial coefficient of variation over repeated measurements, in percent."""
    vial_cv = df.groupby("idx_sphere")["meas_T1"].apply(lambda x: x.std() / x.mean())
    return np.sqrt((vial_cv ** 2).mean()) * 100


def plot_phantom_fit(df_fitted_values, cfg, title, path):
    """
    Fitted T1 vs reference T1 (left) and fitting error (right) for the studies of one figure panel.
    Args:
        df_fitted_values: fitted values of one phantom, restricted to the studies of this panel
        cfg: figure settings with studies, name_to_show and color
        title: panel title
        path: output file
    """
    df_overall_errors = compute_overall_errors(df_fitted_values)
    study_to_name = dict(zip(cfg["studies"], cfg["name_to_show"]))
    study_to_color = dict(zip(cfg["studies"], cfg["color"]))

    fig, (ax_T1, ax_T1err) = plt.subplots(1, 2, figsize=(FIGWIDTH_DOUBLE, FIGHEIGHT_DOUBLE))

    wcv_lines = []
    if "repeatability" in title.lower():
        wcv_lines.append(f"Pooled wCV: {pooled_wcv(df_fitted_values):.1f}%")
    if "positioning" in title.lower():
        for has_nob1, group in df_fitted_values.groupby(df_fitted_values["study_name"].str.contains("NoB1")):
            label = "without B1 correction" if has_nob1 else "with B1 correction"
            wcv_lines.append(f"Pooled wCV ({label}): {pooled_wcv(group):.1f}%")

    legend_handles, legend_labels = [], []
    for study_name, sub in df_fitted_values.groupby("study_name"):
        errors = df_overall_errors[df_overall_errors["study_name"] == study_name].iloc[0]
        rmse, bias_mean, bias_sd = (round(errors[k]) for k in ("rmse_T1", "bias_mean", "bias_std"))
        color = study_to_color.get(study_name)

        ref = sub["ref_T1"].to_numpy()
        meas = sub["meas_T1"].to_numpy()
        scatter = ax_T1.scatter(ref, meas, marker="x", s=20, color=color)
        ax_T1err.scatter(ref, meas - ref, marker="x", s=20, color=color)

        legend_handles.append(scatter)
        legend_labels.append(f"{study_to_name.get(study_name, study_name)}\n"
                             f"RMSE: {rmse:.0f} ms | Bias: {bias_mean:.0f}±{bias_sd:.0f} ms")

    if wcv_lines:
        ax_T1err.text(0.02, 0.05, "\n".join(wcv_lines), transform=ax_T1err.transAxes,
                      fontsize=FONTSIZE - (4 if len(wcv_lines) == 1 else 6), va="bottom", ha="left")

    max_T1 = 2000
    ax_T1.plot([0, max_T1], [0, max_T1], "k--")
    ax_T1err.axhline(0, color="k", linestyle="--")
    ax_T1.grid(True)
    ax_T1err.grid(True)
    ax_T1.set_xlim(0, 1.05 * max_T1)
    ax_T1.set_ylim(0, 1.05 * max_T1)
    yerr_min, yerr_max = ax_T1err.get_ylim()
    if yerr_min > -400 and yerr_max < 400:
        ax_T1err.set_ylim(-400, 400)

    ax_T1.set_xlabel("Reference T$_1$ [ms]", fontsize=FONTSIZE)
    ax_T1.set_ylabel("Fitted T$_1$ [ms]", fontsize=FONTSIZE)
    ax_T1err.set_xlabel("Reference T$_1$ [ms]", fontsize=FONTSIZE)
    ax_T1err.set_ylabel("Fitting error T$_1$ [ms]", fontsize=FONTSIZE)

    fig.suptitle(title, fontsize=FONTSIZE + 2, y=0.98)
    fig.legend(legend_handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=2,
               fontsize=FONTSIZE - 6, columnspacing=1.5, handletextpad=0.5, frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(top=0.73, bottom=0.15)
    fig.savefig(path, bbox_inches="tight", dpi=600)
    plt.close(fig)


def plot_b1_maps(b1_maps, column_titles, path, arrow=None, vmin=50, vmax=150):
    """
    Grid of B1+ maps, rows GSP and CM, one column per B1+ mapping method.
    Args:
        b1_maps: {'GSP': [map, ...], 'CM': [map, ...]}, maps in percent, displayed as stored (rows S->I, columns R->L)
        column_titles: one title per column
        path: output file
        arrow: optional (column, row) in the GSP map of the last column to point at
    """
    fig = plt.figure(figsize=(FIGWIDTH_DOUBLE, FIGHEIGHT_DOUBLE * 1.2))
    n_cols = len(column_titles)
    grid = fig.add_gridspec(nrows=2, ncols=n_cols + 1, width_ratios=[1] * n_cols + [0.05], wspace=0.02, hspace=0.02)
    axes = np.array([[fig.add_subplot(grid[r, c]) for c in range(n_cols)] for r in range(2)])
    cax = fig.add_subplot(grid[:, n_cols])

    for r, phantom in enumerate(['GSP', 'CM']):
        for c, b1_map in enumerate(b1_maps[phantom]):
            ax = axes[r, c]
            im = ax.imshow(b1_map, cmap='gray', vmin=vmin, vmax=vmax)
            h, w = b1_map.shape
            for x, y, text, va, ha in [(0.03, 0.5, 'R', 'center', 'left'), (0.97, 0.5, 'L', 'center', 'right'),
                                       (0.5, 0.03, 'S', 'top', 'center'), (0.5, 0.97, 'I', 'bottom', 'center')]:
                ax.text(w * x, h * y, text, color='white', fontsize=7, fontweight='bold', va=va, ha=ha)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(column_titles[c])
        axes[r, 0].set_ylabel(phantom, rotation=90, labelpad=10)

    if arrow is not None:
        x, y = arrow
        axes[0, -1].annotate('', xy=(x, y), xytext=(x - 10, y + 10),
                             arrowprops=dict(facecolor="white", edgecolor="white", arrowstyle="-|>", linewidth=2.5))

    multiline_titles = any('\n' in t for t in column_titles)
    fig.suptitle("B1$^+$-maps", y=1.04 if multiline_titles else 0.98)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("B1$^+$ Scaling Factor [%]", rotation=270, labelpad=12)
    plt.tight_layout()
    fig.savefig(path, bbox_inches='tight', dpi=600)
    plt.close(fig)

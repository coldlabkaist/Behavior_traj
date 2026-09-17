"""DAE, factor-analysis and latent-projection validation figures."""
from __future__ import annotations
from analysis.plots.style import apply_style, CONTROL_COLOR, VPA_COLOR
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from analysis.plots.export import save_figure
from matplotlib.patches import Rectangle


# DAE validation

PROJECT_ROOT = Path(__file__).resolve().parents[2]


CHECKPOINT = PROJECT_ROOT / "checkpoints" / "best_clean.pth"


SPLIT_MANIFEST = PROJECT_ROOT / "checkpoints" / "split_manifest.json"


AUDIT_DIR = PROJECT_ROOT / "analysis" / "output" / "Final" / "FigS3AB" / "stat"


DAE_TEXT_SCALE = 1.0


def _dae_fontsize(value: float) -> float:
    return value * DAE_TEXT_SCALE


NAVY = "#2F5B8A"


RED = "#CB4B5C"


DARK = "#202833"


MID_GREY = "#6D7682"


GRID = "#DCE1E7"


TARGET_ORDER = [
    "body_distance_log1p",
    "nose_nose_distance_log1p",
    "bidirectional_nose_tailbase_distance_log1p",
    "dyadic_grouping",
    "mutual_facing_mean",
    "approach_rate_asinh",
    "relative_speed_log1p",
    "configuration_speed_log1p",
]


TARGET_LABELS = {
    "body_distance_log1p": "Body distance",
    "nose_nose_distance_log1p": "Nose–nose distance",
    "bidirectional_nose_tailbase_distance_log1p": "Nose–tailbase distance",
    "dyadic_grouping": "Dyadic grouping",
    "mutual_facing_mean": "Mutual facing",
    "approach_rate_asinh": "Approach rate",
    "relative_speed_log1p": "Relative speed",
    "configuration_speed_log1p": "Configuration speed",
}


TARGET_COLORS = {
    "body_distance_log1p": "#C92F32",
    "nose_nose_distance_log1p": "#C92F32",
    "bidirectional_nose_tailbase_distance_log1p": "#C92F32",
    "dyadic_grouping": "#C92F32",
    "mutual_facing_mean": "#2A9D58",
    "approach_rate_asinh": "#2A9D58",
    "relative_speed_log1p": "#2F6FB0",
    "configuration_speed_log1p": "#2F6FB0",
}


def _style_dae_axis(ax: plt.Axes, *, horizontal_grid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.7)
    ax.spines["bottom"].set_linewidth(1.7)
    ax.tick_params(axis="both", labelsize=_dae_fontsize(15), width=1.5, length=6)
    if horizontal_grid:
        ax.grid(axis="y", color=GRID, linewidth=1.0, alpha=0.75, zorder=0)


def render_dae_validation(output_dir: Path, *, talk: bool = False) -> None:
    global DAE_TEXT_SCALE
    DAE_TEXT_SCALE = 2.10 if talk else 1.0
    output_dir.mkdir(parents=True, exist_ok=True)
    apply_style('dae_validation', text_scale=DAE_TEXT_SCALE)

    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    train_loss = np.asarray(checkpoint["train_losses"], dtype=float)
    val_history = checkpoint["val_losses"]
    val_epochs = np.asarray([i + 1 for i, value in enumerate(val_history) if value is not None])
    val_loss = np.asarray([value for value in val_history if value is not None], dtype=float)
    best_epoch = int(checkpoint["epoch"])
    best_loss = float(checkpoint["best_val_loss"])

    with SPLIT_MANIFEST.open("r", encoding="utf-8") as handle:
        split_manifest = json.load(handle)
    train_clips = int(split_manifest["train"]["n_clips"])
    val_clips = int(split_manifest["validation"]["n_clips"])
    train_cages = int(split_manifest["train"]["n_subjects"])
    val_cages = int(split_manifest["validation"]["n_subjects"])

    nonlinear = pd.read_csv(AUDIT_DIR / "FigS3B_latent_feature_r2.csv")
    nonlinear = nonlinear.set_index("target").loc[TARGET_ORDER].reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(32.0, 12.5) if talk else (19.5, 7.7))
    fig.suptitle(
        "DAE training and latent representation validation",
        fontsize=_dae_fontsize(29),
        fontweight="bold",
        y=0.98,
    )

    # Training dynamics
    ax = axes[0]
    epochs = np.arange(1, len(train_loss) + 1)
    ax.plot(epochs, train_loss, color=NAVY, linewidth=3.1, label="Training")
    ax.plot(
        val_epochs,
        val_loss,
        color=RED,
        linewidth=2.6,
        marker="o",
        markersize=7.5,
        markeredgecolor="white",
        markeredgewidth=1.0,
        label="Validation",
    )
    ax.scatter([best_epoch], [best_loss], s=125, color=RED, edgecolor="white", linewidth=1.2, zorder=5)
    ax.annotate(
        f"Best Val loss = {best_loss:.4f}",
        xy=(best_epoch, best_loss),
        xytext=(64, 0.080),
        fontsize=_dae_fontsize(13),
        fontweight="bold",
        arrowprops=dict(arrowstyle="-", color=MID_GREY, linewidth=1.2),
    )
    ax.set_yscale("log")
    ax.set_ylim(0.038, 2.25)
    ax.set_xlim(1, 102)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Composite loss (log scale)")
    ax.set_title("Training dynamics", pad=15)
    ax.legend(frameon=False, loc="upper right")
    _style_dae_axis(ax)

    # Held-cage-out behavior information in latent z
    ax = axes[1]
    y = np.arange(len(nonlinear))[::-1]
    colors = [TARGET_COLORS[target] for target in nonlinear["target"]]
    ax.barh(
        y,
        nonlinear["nonlinear_r2"],
        height=0.60,
        color=colors,
        alpha=0.90,
        label="Nonlinear probe",
        zorder=2,
    )
    for yi, value in zip(y, nonlinear["nonlinear_r2"]):
        ax.text(
            min(float(value) - 0.02, 0.965),
            yi,
            f"{float(value):.2f}",
            ha="right",
            va="center",
            fontsize=_dae_fontsize(12),
            fontweight="bold",
            color="white",
        )
    ax.set_yticks(y, [TARGET_LABELS[target] for target in nonlinear["target"]])
    ax.set_xlim(0, 1.02)
    ax.set_ylim(-0.7, 8.4)
    ax.set_xlabel(r"Held-cage-out $R^2$")
    ax.set_title("Behavioral feature decoding from latent z", pad=15)
    _style_dae_axis(ax, horizontal_grid=False)
    ax.grid(axis="x", color=GRID, linewidth=1.0, alpha=0.75, zorder=0)

    fig.subplots_adjust(
        left=0.105,
        right=0.98,
        bottom=0.14 if talk else 0.12,
        top=0.82 if talk else 0.84,
        wspace=0.34 if talk else 0.29,
    )
    output_stem = output_dir / "FigS3AB_dae_training_and_latent_validation"
    for output in save_figure(fig, output_stem, svg_dpi=300, bbox_inches="tight", facecolor="white"):
        print(output)
    plt.close(fig)


# Factor analysis


LOADINGS_CSV = Path(__file__).resolve().parents[2] / "analysis/output/Final/FigS4AB/stat/FigS4B_pattern_loadings.csv"


PARALLEL_CSV = Path(__file__).resolve().parents[2] / "analysis/output/Final/FigS4AB/stat/FigS4A_parallel_analysis.csv"


FACTOR_TEXT_SCALE = 1.0


def _factor_fontsize(value: float) -> float:
    return value * FACTOR_TEXT_SCALE


FACTOR_NAMES = ["Spacing", "Orientation", "Dynamics"]


FEATURE_ORDER = [
    "Body distance",
    "Nose-Nose distance",
    "Nose-Tailbase distance",
    "Dyadic grouping",
    "Mutual facing",
    "Approach rate",
    "Relative speed",
    "Configuration speed",
]


GROUPS = [
    ("Spacing axis", 0, 4, "#C92F32"),
    ("Orientation axis", 4, 2, "#2A9D58"),
    ("Dynamics axis", 6, 2, "#2F6FB0"),
]


def _style_factor() -> None:
    apply_style('factor_analysis', text_scale=FACTOR_TEXT_SCALE)


def _load_factor_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    loadings = pd.read_csv(LOADINGS_CSV, index_col=0)
    loadings = loadings.loc[FEATURE_ORDER, FACTOR_NAMES]
    parallel = pd.read_csv(PARALLEL_CSV).sort_values("component")
    return loadings, parallel


def _draw_group_labels(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(8, 0)
    ax.axis("off")

    for label, start, height, color in GROUPS:
        # The label axis shares the heatmap's 0..8 row coordinate system.
        rect = Rectangle(
            (0.02, start + 0.02),
            0.96,
            height - 0.04,
            fill=False,
            edgecolor=color,
            linewidth=5.2 if FACTOR_TEXT_SCALE > 1.0 else 3.2,
            clip_on=False,
        )
        ax.add_patch(rect)
        ax.text(
            0.50,
            start + (0.075 if FACTOR_TEXT_SCALE > 1.0 else 0.03),
            label,
            ha="center",
            va="center" if FACTOR_TEXT_SCALE > 1.0 else "bottom",
            fontsize=_factor_fontsize(21),
            fontweight="bold",
            color=color,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "pad": 0.7 if FACTOR_TEXT_SCALE > 1.0 else 1.2,
            },
        )

    for row, feature in enumerate(FEATURE_ORDER):
        ax.text(
            0.94,
            row + 0.57,
            feature,
            ha="right",
            va="center",
            fontsize=_factor_fontsize(18.5),
            color="#20242B",
        )


def _draw_heatmap(ax: plt.Axes, cax: plt.Axes, loadings: pd.DataFrame) -> None:
    values = loadings.to_numpy(dtype=float)
    image = ax.imshow(values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(3))
    ax.set_xticklabels(
        ["F1\nSpacing", "F2\nOrientation", "F3\nDynamics"],
        fontsize=_factor_fontsize(19),
    )
    ax.set_yticks([])
    ax.tick_params(axis="x", length=0, pad=10)
    ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 8, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.6)
        spine.set_color("#20242B")

    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            ax.text(
                column,
                row,
                f"{value:+.2f}",
                ha="center",
                va="center",
                fontsize=_factor_fontsize(17),
                fontweight="bold" if abs(value) >= 0.40 else "normal",
                color="white" if abs(value) >= 0.62 else "#20242B",
            )

    colorbar = plt.colorbar(image, cax=cax)
    colorbar.set_label("Pattern loading", fontsize=_factor_fontsize(19), labelpad=8)
    colorbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    colorbar.ax.tick_params(labelsize=_factor_fontsize(17), width=1.4, length=6)
    colorbar.outline.set_linewidth(1.4)


def _draw_parallel(ax: plt.Axes, parallel: pd.DataFrame) -> None:
    components = parallel["component"].to_numpy(dtype=int)
    observed = parallel["observed_eigenvalue"].to_numpy(dtype=float)
    null95 = parallel["block_null_95"].to_numpy(dtype=float)
    retained = parallel["retained"].astype(str).str.lower().eq("true").to_numpy()
    x = np.arange(len(components))

    colors = np.where(retained, "#3279A7", "#C5CBD3")
    ax.bar(
        x,
        observed,
        width=0.64,
        color=colors,
        edgecolor="#26313B",
        linewidth=1.4,
        zorder=2,
    )
    ax.plot(
        x,
        null95,
        color="#A94031",
        marker="o",
        markersize=8.5,
        linewidth=3.0,
        label="Block-null 95th percentile",
        zorder=3,
    )

    for xpos, value, keep in zip(x, observed, retained):
        if keep:
            ax.text(
                xpos,
                value + 0.10,
                "retain",
                ha="center",
                va="bottom",
                color="#3279A7",
                fontsize=_factor_fontsize(17),
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([f"C{component}" for component in components])
    ax.set_xlabel("Component", labelpad=10)
    ax.set_ylabel("Eigenvalue", labelpad=10)
    ax.set_ylim(0, 4.25)
    ax.set_xlim(-0.6, len(x) - 0.4)
    ax.legend(loc="upper right", frameon=False, handlelength=2.1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


def render_factor_validation(output_dir: Path, *, talk: bool = False) -> None:
    global FACTOR_TEXT_SCALE
    FACTOR_TEXT_SCALE = 1.78 if talk else 1.0
    _style_factor()
    loadings, parallel = _load_factor_tables()

    fig = plt.figure(figsize=(34.0, 15.0) if talk else (25.5, 11.8), facecolor="white")
    outer = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.28, 1.00],
        left=0.035,
        right=0.985,
        bottom=0.12,
        top=0.89,
        wspace=0.28 if talk else 0.24,
    )
    left = outer[0].subgridspec(1, 3, width_ratios=[0.86, 1.08, 0.08], wspace=0.05)
    label_ax = fig.add_subplot(left[0])
    heat_ax = fig.add_subplot(left[1])
    colorbar_ax = fig.add_subplot(left[2])
    parallel_ax = fig.add_subplot(outer[1])

    _draw_group_labels(label_ax)
    _draw_heatmap(heat_ax, colorbar_ax, loadings)
    _draw_parallel(parallel_ax, parallel)

    heat_ax.set_title("Factor Analysis", fontweight="bold", pad=18)
    parallel_ax.set_title("Parallel Analysis", fontweight="bold", pad=18)
    if not talk:
        fig.text(0.020, 0.935, "A", fontsize=_factor_fontsize(31), fontweight="bold", ha="left", va="center")
        fig.text(0.563, 0.935, "B", fontsize=_factor_fontsize(31), fontweight="bold", ha="left", va="center")

    stem = "talk_axis_design_factor_parallel_analysis" if talk else "control_axis_design_factor_parallel_analysis"
    save_figure(fig, output_dir / stem, facecolor="white")
    plt.close(fig)


# Projection

FACTORS = ("Spacing", "Orientation", "Dynamics")


FACTOR_COLUMNS = ("spacing", "orientation", "dynamics")


INK = "#202934"


def render_projection_validation(
    target: np.ndarray,
    prediction: np.ndarray,
    cage_week: pd.DataFrame,
    metrics: pd.DataFrame,
    output_dir: Path,
) -> None:
    apply_style('projection_validation')
    figure, axes = plt.subplots(2, 3, figsize=(15, 9))
    for index, (factor, label) in enumerate(zip(FACTOR_COLUMNS, FACTORS)):
        low, high = np.quantile(np.concatenate([target[:, index], prediction[:, index]]), [0.005, 0.995])
        axes[0, index].hexbin(
            target[:, index], prediction[:, index], gridsize=70, bins="log", cmap="Blues", mincnt=1
        )
        axes[0, index].plot([low, high], [low, high], "--", color=VPA_COLOR, linewidth=1.3)
        axes[0, index].set(xlim=(low, high), ylim=(low, high), xlabel="Direct feature-derived score", ylabel="OOF latent-z projection", title=label)
        row = metrics[(metrics.factor == factor) & (metrics.level == "window") & (metrics.subset == "all")].iloc[0]
        axes[0, index].text(0.04, 0.94, f"R²={row.r2:.3f}\nr={row.pearson_r:.3f}", transform=axes[0, index].transAxes, va="top")

        true = cage_week[f"target_median_{factor}"].to_numpy(dtype=float)
        pred = cage_week[f"projected_median_{factor}"].to_numpy(dtype=float)
        colors = np.where(cage_week["condition"].eq("control"), CONTROL_COLOR, VPA_COLOR)
        axes[1, index].scatter(true, pred, c=colors, s=34, alpha=0.85, edgecolor="white", linewidth=0.4)
        low, high = np.quantile(np.concatenate([true, pred]), [0.01, 0.99])
        axes[1, index].plot([low, high], [low, high], "--", color=INK, linewidth=1.0)
        axes[1, index].set(xlim=(low, high), ylim=(low, high), xlabel="Direct cage-week median", ylabel="Latent-derived cage-week median")
        row = metrics[(metrics.factor == factor) & (metrics.level == "cage_week_median")].iloc[0]
        axes[1, index].set_title(f"Cage-week median R²={row.r2:.3f}")
    figure.suptitle("Control-LOCO projection of frozen DAE z to interpretable behavioral axes", fontsize=17, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(figure, output_dir / "latent_z_to_behavioral_axes_validation", bbox_inches="tight", facecolor="white")
    plt.close(figure)

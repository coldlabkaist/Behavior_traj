"""Frozen motif space and representative-pose figures."""
from __future__ import annotations
from analysis.plots.style import apply_style, MOTIF_COLORS as TOKEN_COLORS
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
from analysis.plots.export import save_figure
from analysis.core.definitions import TOKENS as TOKEN_KEYS, MOTIF_NAMES as TOKEN_NAMES


# Motif space

FACTOR_COLUMNS = ("spacing", "orientation", "dynamics")


def _style_motif_space() -> None:
    apply_style('motifs')


def _rgba_by_motif(tokens: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    colors = np.empty((len(tokens), 4), dtype=float)
    for token, color in enumerate(TOKEN_COLORS):
        colors[tokens == token] = to_rgba(color)
    colors[:, 3] = 0.40 + 0.46 * np.clip(confidence, 0.0, 1.0)
    return colors


def render_motif_space(
    assignments_path: Path,
    codebook_path: Path,
    output_base: Path,
    *,
    seed: int = 42,
) -> None:
    _style_motif_space()
    usecols = ["condition", *FACTOR_COLUMNS, "hard_token", "max_posterior"]
    assignments = pd.read_csv(assignments_path, usecols=usecols)
    control = assignments[assignments["condition"].astype(str).str.lower() == "control"]
    with np.load(codebook_path, allow_pickle=True) as archive:
        means = np.asarray(archive["means"], dtype=float)

    values = control[list(FACTOR_COLUMNS)].to_numpy(dtype=float)
    ranges: list[tuple[float, float]] = []
    for factor in range(3):
        lower, upper = np.quantile(values[:, factor], [0.001, 0.999])
        center = 0.5 * (lower + upper)
        radius = 0.54 * (upper - lower)
        ranges.append((float(center - radius), float(center + radius)))

    rng = np.random.default_rng(seed)
    subset = control
    if len(subset) > 20_000:
        subset = subset.iloc[rng.choice(len(subset), 20_000, replace=False)]
    points = subset[list(FACTOR_COLUMNS)].to_numpy(dtype=float)
    tokens = subset["hard_token"].to_numpy(dtype=int)
    confidence = subset["max_posterior"].to_numpy(dtype=float)
    order = rng.permutation(len(subset))

    fig = plt.figure(figsize=(19.0, 15.5), facecolor="white")
    ax = fig.add_subplot(1, 1, 1, projection="3d", computed_zorder=False)
    ax.scatter(
        points[order, 0],
        points[order, 1],
        points[order, 2],
        s=8.0 + 16.0 * confidence[order] ** 2,
        c=_rgba_by_motif(tokens[order], confidence[order]),
        linewidths=0,
        depthshade=False,
        rasterized=True,
    )
    for token, (key, color) in enumerate(zip(TOKEN_KEYS, TOKEN_COLORS)):
        center = means[token]
        ax.scatter(
            [center[0]],
            [center[1]],
            [center[2]],
            s=760,
            marker="X",
            color=color,
            edgecolor="white",
            linewidth=4.5,
            depthshade=False,
            zorder=20,
        )
        ax.text(
            center[0],
            center[1],
            center[2] + 0.13,
            key,
            ha="center",
            va="bottom",
            fontsize=30,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 2.0},
        )

    ax.set_xlim(ranges[0])
    ax.set_ylim(ranges[1])
    ax.set_zlim(ranges[2])
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.set_proj_type("ortho")
    ax.view_init(elev=21, azim=-57)
    ax.set_box_aspect((1, 1, 1), zoom=1.12)
    ax.set_xlabel("Spacing", fontsize=44, labelpad=22, fontweight="bold")
    ax.set_ylabel("Orientation", fontsize=44, labelpad=22, fontweight="bold")
    ax.set_zlabel("Dynamics", fontsize=44, labelpad=20, fontweight="bold")
    ax.tick_params(axis="both", labelsize=31, width=2.2, length=9, pad=5)
    ax.grid(False)
    ax.xaxis.pane.set_alpha(0.0)
    ax.yaxis.pane.set_alpha(0.0)
    ax.zaxis.pane.set_alpha(0.0)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.line.set_color("#111111")
        axis.line.set_linewidth(2.4)

    handles = [
        Patch(facecolor=color, edgecolor="#333333", label=f"{key}: {name}")
        for key, name, color in zip(TOKEN_KEYS, TOKEN_NAMES, TOKEN_COLORS)
    ]
    fig.suptitle(
        "Control GMM motifs in 3D factor space",
        fontsize=48,
        fontweight="bold",
        y=0.982,
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=2,
        frameon=False,
        fontsize=31,
        columnspacing=1.45,
        handlelength=2.0,
        handleheight=1.2,
    )
    fig.subplots_adjust(left=0.015, right=0.955, bottom=0.025, top=0.815)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, output_base, svg_dpi=300, facecolor="white")
    plt.close(fig)


# Representative poses

POSE_STATIC_DPI = 600


ASPECT = 1920.0 / 1080.0


ANIMAL_COLORS = ("#4c78a8", "#f58518", "#54a24b")


KEYPOINTS = ("Nose", "Body_C", "Ear_L", "Ear_R", "Neck", "Tail")


DISPLAY_EDGES = (
    ("Nose", "Neck"),
    ("Neck", "Body_C"),
    ("Body_C", "Tail"),
    ("Neck", "Ear_L"),
    ("Neck", "Ear_R"),
    ("Nose", "Ear_L"),
    ("Nose", "Ear_R"),
    ("Ear_L", "Ear_R"),
)


SKELETON_EDGE_COLOR = "#6b7280"


KEYPOINT_COLORS = {
    "Nose": "#e07a2d",
    "Body_C": "#e53935",
    "Ear_L": "#f2c94c",
    "Ear_R": "#9b59b6",
    "Neck": "#4c78a8",
    "Tail": "#6b7280",
}


def _draw_skeleton(
    ax: plt.Axes,
    coordinates: np.ndarray,
    *,
    frame_index: int,
    alpha: float,
    linewidth: float,
) -> None:
    keypoint_index = {name: index for index, name in enumerate(KEYPOINTS)}
    for animal in range(len(ANIMAL_COLORS)):
        pose = coordinates[frame_index, animal]
        for start_name, end_name in DISPLAY_EDGES:
            start = pose[keypoint_index[start_name]]
            end = pose[keypoint_index[end_name]]
            if np.isfinite(start).all() and np.isfinite(end).all():
                ax.plot(
                    [start[0], end[0]],
                    [start[1], end[1]],
                    color=SKELETON_EDGE_COLOR,
                    linewidth=linewidth,
                    alpha=alpha,
                    solid_capstyle="round",
                )
        for name, point in zip(KEYPOINTS, pose):
            if not np.isfinite(point).all():
                continue
            ax.scatter(
                point[0],
                point[1],
                s=31 if name == "Body_C" else 18,
                color=KEYPOINT_COLORS[name],
                marker="x" if name == "Body_C" else "o",
                alpha=alpha,
                edgecolors="none" if name == "Body_C" else SKELETON_EDGE_COLOR,
                linewidths=1.8 if name == "Body_C" else 0.4,
                zorder=5,
            )


def _style_pose() -> None:
    apply_style('pose')


def _draw_pose_panel(
    ax: plt.Axes,
    coordinates: np.ndarray,
    *,
    frame_index: int,
    token_index: int,
) -> None:
    ax.clear()
    body_index = KEYPOINTS.index("Body_C")
    for animal, color in enumerate(ANIMAL_COLORS):
        trail = coordinates[: frame_index + 1, animal, body_index, :]
        valid = np.isfinite(trail).all(axis=1)
        if valid.any():
            ax.plot(
                trail[valid, 0],
                trail[valid, 1],
                color=color,
                linewidth=3.0,
                alpha=0.75,
                solid_capstyle="round",
                zorder=2,
            )
    _draw_skeleton(
        ax,
        coordinates,
        frame_index=frame_index,
        alpha=1.0,
        linewidth=2.15,
    )
    ax.set_xlim(0, ASPECT)
    ax.set_ylim(1.0, 0.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"{TOKEN_KEYS[token_index]}\n{TOKEN_NAMES[token_index]}",
        fontsize=24,
        pad=7,
    )
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_color(TOKEN_COLORS[token_index])
        spine.set_linewidth(4.0)


def render_representative_poses(pose: dict[str, np.ndarray], output_base: Path) -> None:
    _style_pose()
    fig, axes = plt.subplots(2, 2, figsize=(16.0, 10.3), facecolor="white")
    for token, (key, ax) in enumerate(zip(TOKEN_KEYS, axes.flat)):
        _draw_pose_panel(
            ax,
            pose[key],
            frame_index=len(pose[key]) - 1,
            token_index=token,
        )
    fig.subplots_adjust(
        left=0.025, right=0.985, top=0.925, bottom=0.035, wspace=0.075, hspace=0.24
    )
    _save_pose_figure(fig, output_base, dpi=POSE_STATIC_DPI)


def _save_pose_figure(fig: plt.Figure, output_base: Path, *, dpi: int) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, output_base, dpi=dpi, facecolor="white")
    plt.close(fig)

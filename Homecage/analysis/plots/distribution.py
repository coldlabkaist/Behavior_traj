"""plots / distribution.

Published-analysis functions. Inputs and current sources are recorded in Final manifests.
"""
from __future__ import annotations

from analysis.plots.style import apply_style, CONTROL_COLOR, VPA_COLOR

from pathlib import Path

from typing import Any

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap, LogNorm

from matplotlib.patches import Rectangle
from analysis.plots.export import save_figure

FIGURE_DPI = 300


SEED = 42


TEXT_COLOR = "#202833"


DENSITY_CMAP = LinearSegmentedColormap.from_list(
    "primary_factor_density",
    ("#eef6ff", "#93c5fd", "#2563eb", "#6d28d9", "#dc2626", "#7f1d1d"),
)


def _style() -> None:
    apply_style('distribution')


def _shared_ranges(coordinates: np.ndarray) -> list[tuple[float, float]]:
    ranges: list[tuple[float, float]] = []
    for factor in range(3):
        lower, upper = np.quantile(coordinates[:, factor], [0.001, 0.999])
        radius = max(abs(float(lower)), abs(float(upper)), 1.0) * 1.04
        ranges.append((-radius, radius))
    return ranges


def _inside_ranges(points: np.ndarray, ranges: list[tuple[float, float]]) -> np.ndarray:
    inside = np.ones(len(points), dtype=bool)
    for factor, (lower, upper) in enumerate(ranges):
        inside &= (points[:, factor] >= lower) & (points[:, factor] <= upper)
    return inside


def _cage_balanced_density(
    points: np.ndarray,
    cages: np.ndarray,
    *,
    ranges: list[tuple[float, float]],
    bins: int,
) -> np.ndarray:
    weights = np.zeros(len(points), dtype=float)
    unique_cages = np.unique(cages)
    for cage in unique_cages:
        mask = cages == cage
        weights[mask] = 1.0 / (len(unique_cages) * int(mask.sum()))
    histogram, edges = np.histogramdd(points, bins=(bins, bins, bins), range=ranges, weights=weights)
    positions = []
    for factor in range(3):
        index = np.searchsorted(edges[factor], points[:, factor], side="right") - 1
        positions.append(np.clip(index, 0, bins - 1))
    return histogram[positions[0], positions[1], positions[2]]


def _sample_equal_cages(
    cages: np.ndarray,
    *,
    points_per_cage: int,
    rng: np.random.Generator,
) -> np.ndarray:
    selected: list[np.ndarray] = []
    for cage in sorted(np.unique(cages)):
        indices = np.flatnonzero(cages == cage)
        if len(indices) > points_per_cage:
            indices = np.sort(rng.choice(indices, size=points_per_cage, replace=False))
        selected.append(indices)
    return np.concatenate(selected)


def _prepare_panels(
    coordinates: np.ndarray,
    metadata: pd.DataFrame,
    ranges: list[tuple[float, float]],
    *,
    bins: int,
    points_per_cage_week: int,
    points_per_cage_overall: int,
) -> tuple[dict[tuple[str, Any], dict], pd.DataFrame, LogNorm]:
    rng = np.random.default_rng(SEED)
    conditions = metadata["condition"].to_numpy(dtype=str)
    weeks = metadata["week"].to_numpy(dtype=int)
    cages = metadata["cage_id"].to_numpy(dtype=str)
    panels: dict[tuple[str, Any], dict] = {}
    audit_rows: list[dict] = []
    density_pool: list[np.ndarray] = []

    specs: list[tuple[str, Any, np.ndarray, int]] = []
    for condition in ("control", "vpa"):
        for week in range(3, 9):
            specs.append((condition, week, (conditions == condition) & (weeks == week), points_per_cage_week))
        specs.append((condition, "all", conditions == condition, points_per_cage_overall))
    specs.append(("combined", "all", np.ones(len(metadata), dtype=bool), points_per_cage_overall))

    for condition, week, mask, cap in specs:
        original_points = coordinates[mask]
        original_cages = cages[mask]
        inside = _inside_ranges(original_points, ranges)
        points = original_points[inside]
        panel_cages = original_cages[inside]
        if len(points) == 0:
            raise RuntimeError(f"No coordinates remain for {condition} {week}.")
        density = _cage_balanced_density(points, panel_cages, ranges=ranges, bins=bins)
        display_index = _sample_equal_cages(panel_cages, points_per_cage=cap, rng=rng)
        display_index = display_index[np.argsort(density[display_index], kind="stable")]
        panels[(condition, week)] = {
            "points": points[display_index],
            "density": density[display_index],
            "n_cages": int(np.unique(panel_cages).size),
            "n_windows": int(len(original_points)),
            "n_displayed": int(len(display_index)),
        }
        positive = density[display_index]
        density_pool.append(positive[positive > 0])
        audit_rows.append(
            {
                "condition": condition,
                "week": week,
                "n_cages": int(np.unique(original_cages).size),
                "n_windows": int(len(original_points)),
                "n_windows_inside_shared_range": int(inside.sum()),
                "outside_shared_range_fraction": float(1 - inside.mean()),
                "n_displayed_points": int(len(display_index)),
                "density_mass_sum": 1.0,
            }
        )

    pooled = np.concatenate(density_pool)
    lower, upper = np.quantile(pooled, [0.03, 0.997])
    lower = max(float(lower), np.finfo(float).tiny)
    upper = max(float(upper), lower * 1.01)
    return panels, pd.DataFrame(audit_rows), LogNorm(vmin=lower, vmax=upper, clip=True)


def _format_3d_axis(
    ax: Any,
    ranges: list[tuple[float, float]],
    *,
    title: str | None = None,
    label_size: float = 21,
    tick_size: float = 17,
    title_size: float = 32,
    label_weight: str = "normal",
    label_pad: float = 9,
    tick_pad: float = 1,
    show_tick_labels: bool = True,
    box_zoom: float = 1.0,
) -> None:
    ax.set_xlim(ranges[0])
    ax.set_ylim(ranges[1])
    ax.set_zlim(ranges[2])
    ax.set_box_aspect((1, 1, 1), zoom=box_zoom)
    ax.set_proj_type("ortho")
    ax.view_init(elev=22, azim=-57)
    ax.set_xlabel("Spacing", fontsize=label_size, fontweight=label_weight, labelpad=label_pad)
    ax.set_ylabel("Orientation", fontsize=label_size, fontweight=label_weight, labelpad=label_pad)
    ax.set_zlabel("Dynamics", fontsize=label_size, fontweight=label_weight, labelpad=max(1, label_pad - 4))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_tick_params(labelsize=tick_size, pad=tick_pad)
    if not show_tick_labels:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
    ax.xaxis.pane.set_alpha(.035)
    ax.yaxis.pane.set_alpha(.035)
    ax.zaxis.pane.set_alpha(.035)
    ax.grid(True, linewidth=.4, alpha=.25)
    ax.plot([ranges[0][0], ranges[0][1]], [0, 0], [0, 0], color="#374151", linewidth=.8, alpha=.75)
    ax.plot([0, 0], [ranges[1][0], ranges[1][1]], [0, 0], color="#374151", linewidth=.8, alpha=.75)
    ax.plot([0, 0], [0, 0], [ranges[2][0], ranges[2][1]], color="#374151", linewidth=.8, alpha=.75)
    if title:
        ax.set_title(title, fontsize=title_size, fontweight="bold", pad=18)


def _scatter_density(ax: Any, panel: dict, norm: LogNorm) -> None:
    scaled = np.asarray(norm(panel["density"]), dtype=float)
    ax.scatter(
        panel["points"][:, 0],
        panel["points"][:, 1],
        panel["points"][:, 2],
        c=panel["density"],
        cmap=DENSITY_CMAP,
        norm=norm,
        s=4.0 + 16.0 * scaled,
        alpha=.60,
        linewidths=0,
        depthshade=False,
        rasterized=True,
    )


def _plot_by_week_talk(
    panels: dict,
    ranges: list[tuple[float, float]],
    norm: LogNorm,
    out_dir: Path,
    *,
    poster: bool = False,
) -> None:
    """Large-type 2x6 distribution map for projection or a poster."""
    _style()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(48, 19.5) if poster else (48, 17.5), facecolor="white")
    for row, condition in enumerate(("control", "vpa")):
        for column, week in enumerate(range(3, 9)):
            ax = fig.add_subplot(2, 6, row * 6 + column + 1, projection="3d")
            panel = panels[(condition, week)]
            _scatter_density(ax, panel, norm)
            _format_3d_axis(
                ax,
                ranges,
                title=f"{week}W (n={panel['n_cages']})",
                label_size=38 if poster else 32,
                tick_size=26 if poster else 25,
                title_size=76 if poster else 52,
                label_weight="bold",
                label_pad=2 if poster else 9,
                tick_pad=0 if poster else 1,
                show_tick_labels=not poster,
                box_zoom=1.13 if poster else 1.0,
            )

    fig.text(
        .014,
        .710,
        "Control",
        rotation=90,
        va="center",
        ha="center",
        fontsize=76 if poster else 52,
        fontweight="bold",
        color=CONTROL_COLOR,
    )
    fig.text(
        .014,
        .285,
        "VPA",
        rotation=90,
        va="center",
        ha="center",
        fontsize=76 if poster else 52,
        fontweight="bold",
        color=VPA_COLOR,
    )
    # Extra horizontal breathing room is intentional for projected talk slides:
    # each 3D z-axis label ("Dynamics") must remain fully inside its own panel.
    fig.subplots_adjust(
        left=.045 if poster else .032,
        right=.978,
        top=.940 if poster else .955,
        bottom=.035 if poster else .025,
        wspace=.14 if poster else .16,
        hspace=.06 if poster else .02,
    )

    # Theme-colored row frames group all six weekly panels without covering labels.
    fig.add_artist(
        Rectangle(
            (.041, .519),
            .953,
            .449,
            transform=fig.transFigure,
            fill=False,
            edgecolor=CONTROL_COLOR,
            linewidth=6.0,
            joinstyle="miter",
            clip_on=False,
            zorder=20,
        )
    )
    fig.add_artist(
        Rectangle(
            (.041, .025),
            .953,
            .480,
            transform=fig.transFigure,
            fill=False,
            edgecolor=VPA_COLOR,
            linewidth=6.0,
            joinstyle="miter",
            clip_on=False,
            zorder=20,
        )
    )

    path = out_dir / (
        "talk_primary_3d_distribution_by_week_poster.png"
        if poster
        else "talk_primary_3d_distribution_by_week.png"
    )
    save_figure(fig, path, dpi=FIGURE_DPI, facecolor="white")
    plt.close(fig)


def _plot_colorbar(norm: LogNorm, out_dir: Path) -> None:
    _style()
    fig = plt.figure(figsize=(14, 1.8), facecolor="white")
    ax = fig.add_axes([.055, .58, .89, .17])
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=DENSITY_CMAP), cax=ax, orientation="horizontal")
    colorbar.set_label("Cage-balanced local density (log scale)", fontsize=20, labelpad=13)
    colorbar.ax.tick_params(labelsize=17, length=6, width=1.2)
    path = out_dir / "primary_3d_distribution_colorbar.png"
    save_figure(fig, path, dpi=FIGURE_DPI, facecolor="white")
    plt.close(fig)


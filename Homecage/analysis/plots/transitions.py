"""Weekly motif transitions and adjacent-week network changes."""

from __future__ import annotations
from pathlib import Path
import itertools
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle, Polygon
import argparse
import json
import matplotlib
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analysis.core.definitions import (
    TOKENS as TOKEN_KEYS,
    OCCUPANCY_COLUMNS,
    TOKENS,
    TRANSITION_COLUMNS,
)
from analysis.core.transitions import build_change_summary
from analysis.plots.export import save_figure
from analysis.plots.style import (
    apply_style,
    MOTIF_COLORS as TOKEN_COLORS,
    CONTROL_COLOR,
    VPA_COLOR,
    NETWORK_DECREASE as DECREASE,
    NETWORK_INK as INK,
    NETWORK_MOTIF_COLORS as MOTIF_COLORS,
)


# Transitions

WEEKS = (3, 4, 5, 6, 7, 8)


K = 4


HATCH = "//"


WEEKLY_NODE_POSITIONS = {
    0: np.asarray([0.19, 0.76]),
    1: np.asarray([0.81, 0.76]),
    2: np.asarray([0.81, 0.23]),
    3: np.asarray([0.19, 0.23]),
}


def _style_weekly_transitions() -> None:
    apply_style('transitions')


def _transition_cmap() -> ListedColormap:
    return ListedColormap(
        plt.get_cmap("YlOrRd")(np.linspace(0.20, 1.0, 256)),
        name="YlOrRd_visible",
    )


def _draw_week(
    ax: plt.Axes,
    week_transition: pd.DataFrame,
    week_occupancy: pd.DataFrame,
    *,
    week: int,
    condition: str,
    norm: Normalize,
    cmap: ListedColormap,
    occupancy_column: str,
    poster: bool = False,
) -> None:
    transition_lookup = week_transition.set_index(["source", "target"])
    occupancy_lookup = week_occupancy.set_index("token")
    minimum_radius = 0.065 if poster else 0.052
    radius_scale = 0.175 if poster else 0.155
    radii = {
        token: max(
            minimum_radius,
            radius_scale
            * np.sqrt(
                float(occupancy_lookup.loc[TOKEN_KEYS[token], occupancy_column])
                / 0.50
            ),
        )
        for token in range(K)
    }

    for source, target in itertools.permutations(range(K), 2):
        row = transition_lookup.loc[(TOKEN_KEYS[source], TOKEN_KEYS[target])]
        probability = float(row["transition_probability"])
        total_count = int(row["total_count"])
        if not np.isfinite(probability) or total_count <= 0 or probability <= 0.0:
            continue

        strength = float(np.clip(norm(probability), 0.0, 1.0))
        delta = WEEKLY_NODE_POSITIONS[target] - WEEKLY_NODE_POSITIONS[source]
        unit = delta / np.linalg.norm(delta)
        pair_start = WEEKLY_NODE_POSITIONS[min(source, target)]
        pair_end = WEEKLY_NODE_POSITIONS[max(source, target)]
        pair_unit = (pair_end - pair_start) / np.linalg.norm(pair_end - pair_start)
        perpendicular = np.asarray([-pair_unit[1], pair_unit[0]])
        offset = perpendicular * (0.024 if source < target else -0.024)
        start = WEEKLY_NODE_POSITIONS[source] + unit * (radii[source] + 0.018) + offset
        end = WEEKLY_NODE_POSITIONS[target] - unit * (radii[target] + 0.018) + offset
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                connectionstyle="arc3,rad=0",
                color=cmap(norm(probability)),
                linewidth=1.05 + 11.5 * strength,
                alpha=0.86 + 0.14 * strength,
                mutation_scale=12.0 + 17.0 * strength,
                zorder=1,
            )
        )

    node_hatch = HATCH if condition == "vpa" else None
    for token, key in enumerate(TOKEN_KEYS):
        value = float(occupancy_lookup.loc[key, occupancy_column])
        ax.add_patch(
            Circle(
                WEEKLY_NODE_POSITIONS[token],
                radii[token],
                facecolor=TOKEN_COLORS[token],
                edgecolor="#29232f",
                linewidth=2.4,
                hatch=node_hatch,
                zorder=3,
            )
        )
        text_color = "white" if token in (1, 3) else "#29232f"
        text_effects = None
        if poster:
            text_color = "white"
            text_effects = [
                path_effects.withStroke(linewidth=3.2, foreground="#29232f")
            ]
        ax.text(
            WEEKLY_NODE_POSITIONS[token][0],
            WEEKLY_NODE_POSITIONS[token][1] if poster else WEEKLY_NODE_POSITIONS[token][1] + 0.019,
            key,
            ha="center",
            va="center",
            fontsize=32 if poster else 18,
            fontfamily="Arial",
            fontweight="bold",
            color=text_color,
            path_effects=text_effects,
            zorder=4,
        )
        if not poster:
            ax.text(
                WEEKLY_NODE_POSITIONS[token][0],
                WEEKLY_NODE_POSITIONS[token][1] - 0.031,
                f"{value:.0%}",
                ha="center",
                va="center",
                fontsize=16,
                fontweight="bold",
                color=text_color,
                zorder=4,
            )

    ax.set_title(
        f"{week}W",
        fontsize=42 if poster else 29,
        fontweight="normal",
        y=0.865,
        pad=0,
    )
    ax.set_xlim(0.01, 0.99)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")


def _save_figure(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, output_base, facecolor="white")
    plt.close(fig)


def render_weekly_transitions(
    transition_path: Path,
    occupancy_path: Path,
    output_base: Path,
    *,
    hard_occupancy: bool = False,
    poster: bool = False,
) -> None:
    _style_weekly_transitions()
    transition = pd.read_csv(transition_path)
    occupancy = pd.read_csv(occupancy_path)
    # Keep the shared scale identical to the established Control/VPA figure.
    # Values above 35% are shown with the saturated upper-end color.
    maximum = 0.35
    norm = Normalize(vmin=0.0, vmax=maximum)
    cmap = _transition_cmap()
    occupancy_column = "hard_occupancy" if hard_occupancy else "soft_occupancy"

    fig = plt.figure(figsize=(32.0, 16.2) if poster else (32.0, 14.4), facecolor="white")
    grid = fig.add_gridspec(
        3,
        6,
        height_ratios=(1.0, 1.0, 0.18),
        left=0.025,
        right=0.985,
        top=0.885 if poster else 0.905,
        bottom=0.065,
        wspace=0.0,
        hspace=0.24 if poster else 0.20,
    )

    fig.suptitle(
        "Developmental changes in motif transition patterns",
        fontsize=62 if poster else 42,
        fontweight="bold",
        y=0.975,
    )

    row_axes: dict[str, list[plt.Axes]] = {"control": [], "vpa": []}
    for row_index, condition in enumerate(("control", "vpa")):
        condition_transition = transition[transition["condition"] == condition]
        condition_occupancy = occupancy[occupancy["condition"] == condition]
        for column_index, week in enumerate(WEEKS):
            ax = fig.add_subplot(grid[row_index, column_index])
            row_axes[condition].append(ax)
            _draw_week(
                ax,
                condition_transition[condition_transition["week"] == week],
                condition_occupancy[condition_occupancy["week"] == week],
                week=week,
                condition=condition,
                norm=norm,
                cmap=cmap,
                occupancy_column=occupancy_column,
                poster=poster,
            )

    # Row-level theme borders are placed in figure coordinates so every node,
    # arrow, and week title remains untouched inside its own panel.
    fig.canvas.draw()
    row_titles = {
        "control": "Control motif transition",
        "vpa": "VPA motif transition",
    }
    for condition, color in (("control", CONTROL_COLOR), ("vpa", VPA_COLOR)):
        positions = [ax.get_position() for ax in row_axes[condition]]
        left = min(position.x0 for position in positions) - 0.010
        right = max(position.x1 for position in positions) + 0.010
        bottom = min(position.y0 for position in positions) - 0.010
        top = max(position.y1 for position in positions) + 0.010
        fig.add_artist(
            Rectangle(
                (left, bottom),
                right - left,
                top - bottom,
                transform=fig.transFigure,
                fill=False,
                edgecolor=color,
                linewidth=5.0,
                zorder=20,
                clip_on=False,
            )
        )
        fig.text(
            (left + right) / 2,
            top,
            row_titles[condition],
            ha="center",
            va="center",
            fontsize=44 if poster else 30,
            fontweight="bold",
            color=color,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 4.0},
            zorder=30,
        )

    colorbar_slot = grid[2, 1:5].get_position(fig)
    color_ax = fig.add_axes(
        [
            colorbar_slot.x0 + 0.012,
            colorbar_slot.y0 + 0.040,
            colorbar_slot.width - 0.024,
            colorbar_slot.height * 0.28,
        ]
    )
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=color_ax,
        orientation="horizontal",
    )
    colorbar.ax.xaxis.set_major_formatter(
        matplotlib.ticker.PercentFormatter(xmax=1.0, decimals=0)
    )
    colorbar.ax.tick_params(
        labelsize=29 if poster else 22,
        length=7,
        width=1.5,
        pad=5,
    )
    colorbar.outline.set_linewidth(1.6)
    fig.text(
        0.5,
        colorbar_slot.y0 - 0.002,
        "Transition probability",
        ha="center",
        va="top",
        fontsize=36 if poster else 27,
    )

    _save_figure(fig, output_base)


# Network changes

CONDITIONS = ("control", "vpa")


INTERVALS = ((3, 4), (4, 5), (5, 6))


LARGE_NODE_RADIUS = 0.124


CONDITION_LABELS = {"control": "Control", "vpa": "VPA"}


MOTIF_LABELS = {
    "M0": "Compact-static",
    "M1": "Dispersal /\nnon-oriented dynamic",
    "M2": "Separated-static",
    "M3": "Approach-oriented\ndynamic",
}


CHANGE_NODE_POSITIONS = {
    "M0": np.array([0.23, 0.68]),
    "M1": np.array([0.77, 0.68]),
    "M2": np.array([0.23, 0.23]),
    "M3": np.array([0.77, 0.23]),
}


def large_node_edge_endpoints(source: str, target: str, reciprocal: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Return endpoints outside the enlarged nodes, with reciprocal lanes split."""
    start = CHANGE_NODE_POSITIONS[source].copy()
    end = CHANGE_NODE_POSITIONS[target].copy()
    direction = end - start
    unit = direction / np.linalg.norm(direction)
    first, second = sorted((source, target))
    canonical = CHANGE_NODE_POSITIONS[second] - CHANGE_NODE_POSITIONS[first]
    canonical /= np.linalg.norm(canonical)
    normal = np.array([-canonical[1], canonical[0]])
    lane = 1.0 if source == first else -1.0
    # Smaller lane offsets keep diagonal and horizontal/vertical ports from
    # converging near a node. A small radial gap keeps their thick heads apart.
    offset = normal * (0.045 if reciprocal else 0.025) * lane
    endpoint_radius = 0.170 if reciprocal else LARGE_NODE_RADIUS + 0.022
    return (
        start + unit * endpoint_radius + offset,
        end - unit * endpoint_radius + offset,
    )


def draw_small_self_loop(
    axis: plt.Axes,
    token: str,
    color: str,
    linewidth: float,
    linestyle: str | tuple,
) -> np.ndarray:
    """Draw a near-circular self-loop with one integrated tangent arrowhead."""
    center = CHANGE_NODE_POSITIONS[token]
    side = -1 if center[0] < 0.5 else 1
    loop_radius = 0.082
    head_length = 0.036
    head_half_width = 0.023
    # The inner corner of the triangular head just meets the node boundary;
    # none of the head is hidden underneath the subsequently drawn node.
    loop_offset = LARGE_NODE_RADIUS + loop_radius + head_half_width
    loop_center = center + np.array([side * loop_offset, 0.0])
    if side < 0:
        # The loop opens toward the node on the right and terminates at the
        # inner edge with a vertically downward arrowhead, as in the reference.
        angles = np.linspace(300.0, 0.0, 121)
    else:
        # Mirrored geometry for nodes in the right column; the terminal
        # arrowhead again points vertically downward rather than upward.
        angles = np.linspace(240.0, 540.0, 121)

    radians = np.deg2rad(angles)
    vertices = loop_center + loop_radius * np.column_stack(
        [np.cos(radians), np.sin(radians)]
    )
    axis.plot(
        vertices[:, 0],
        vertices[:, 1],
        linewidth=linewidth,
        linestyle=linestyle,
        color=color,
        solid_capstyle="round",
        dash_capstyle="round",
        solid_joinstyle="round",
        zorder=2,
    )
    # The loop always terminates downward, matching the supplied reference.
    # Drawing the head independently keeps it solid even when the loop is dashed.
    base = vertices[-1]
    tangent = np.array([0.0, -1.0])
    normal = np.array([1.0, 0.0])
    tip = base + tangent * head_length
    axis.add_patch(
        Polygon(
            np.vstack(
                [tip, base + normal * head_half_width, base - normal * head_half_width]
            ),
            closed=True,
            facecolor=color,
            edgecolor=color,
            linewidth=0,
            zorder=3.2,
        )
    )
    # Leave a short visible piece of the loop immediately before the head.
    return loop_center + np.array([side * 0.022, 0.0])


def draw_panel(
    axis: plt.Axes,
    changes: pd.DataFrame,
    condition: str,
    start: int,
    end: int,
    max_abs_transition: float,
) -> None:
    current = changes[
        changes["condition"].eq(condition)
        & changes["start_week"].eq(start)
        & changes["end_week"].eq(end)
    ].copy()
    transitions = current[current["feature_type"].eq("transition")].copy()
    transitions["absolute_change"] = transitions["mean_change_pp"].abs()
    displayed = transitions.nlargest(5, "absolute_change")
    displayed_edges = set(zip(displayed["source"], displayed["target"]))
    crossing_diagonals = bool(
        displayed_edges & {("M0", "M3"), ("M3", "M0")}
    ) and bool(displayed_edges & {("M1", "M2"), ("M2", "M1")})

    for row in displayed.itertuples():
        raw_change = float(row.mean_change_pp)
        magnitude = abs(raw_change) / max_abs_transition
        color = INK if raw_change >= 0 else DECREASE
        linewidth = 2.6 + 7.2 * np.sqrt(max(magnitude, 0.0))
        linestyle = "-" if raw_change >= 0 else (0, (5, 3))
        if row.source == row.target:
            label_pos = draw_small_self_loop(axis, row.source, color, linewidth, linestyle)
        else:
            reciprocal = (row.target, row.source) in displayed_edges
            start_point, end_point = large_node_edge_endpoints(row.source, row.target, reciprocal)
            axis.add_patch(
                FancyArrowPatch(
                    start_point,
                    end_point,
                    arrowstyle="-|>",
                    mutation_scale=20 + linewidth,
                    linewidth=linewidth,
                    linestyle=linestyle,
                    color=color,
                    alpha=0.98,
                    shrinkA=0,
                    shrinkB=0,
                    capstyle="round",
                    joinstyle="round",
                    zorder=2,
                )
            )
            # Keep the box on the shaft, shifted toward the source so the
            # arrowhead remains visible.
            fraction = 0.43
            if reciprocal:
                fraction = 0.35
            if crossing_diagonals:
                fraction = {
                    ("M0", "M3"): 0.24,
                    ("M3", "M0"): 0.76,
                    ("M1", "M2"): 0.24,
                    ("M2", "M1"): 0.70,
                }.get((row.source, row.target), fraction)
            label_pos = start_point + fraction * (end_point - start_point)
        axis.text(
            label_pos[0],
            label_pos[1],
            f"{raw_change:+.1f}%",
            ha="center",
            va="center",
            fontsize=20.5,
            fontweight="bold",
            color=INK,
            bbox=dict(
                boxstyle="round,pad=0.20",
                facecolor="white",
                edgecolor=color,
                linewidth=1.0,
                alpha=0.97,
            ),
            zorder=5,
        )

    occupancies = current[current["feature_type"].eq("occupancy")].set_index("source")
    for token in TOKENS:
        raw_change = float(occupancies.loc[token, "mean_change_pp"])
        border_color = INK if raw_change >= 0 else DECREASE
        center = CHANGE_NODE_POSITIONS[token]
        axis.scatter(
            [center[0]],
            [center[1]],
            s=9000,
            color=MOTIF_COLORS[token],
            edgecolor=border_color,
            linewidth=2.4,
            zorder=3,
        )
        text_color = "white" if token in {"M1", "M3"} else INK
        axis.text(
            center[0],
            center[1] + 0.033,
            token,
            ha="center",
            va="center",
            fontsize=34.0,
            fontweight="bold",
            color=text_color,
            zorder=4,
        )
        axis.text(
            center[0] + 0.062,
            center[1] + 0.033,
            "↑" if raw_change >= 0 else "↓",
            ha="center",
            va="center",
            fontsize=32.0,
            fontweight="normal",
            color=text_color,
            zorder=4,
        )
        axis.text(
            center[0],
            center[1] - 0.030,
            f"{raw_change:+.1f}%",
            ha="center",
            va="center",
            fontsize=20.0,
            fontweight="bold",
            color=INK,
            bbox=dict(
                boxstyle="round,pad=0.15",
                facecolor="white",
                edgecolor=border_color,
                linewidth=0.9,
                alpha=0.96,
            ),
            zorder=4,
        )
        label_y = center[1] + 0.165 if token in {"M0", "M1"} else center[1] - 0.165
        axis.text(
            center[0],
            label_y,
            MOTIF_LABELS[token],
            ha="center",
            va="center",
            fontsize=21.0,
            fontweight="bold",
            color=INK,
            linespacing=0.95,
            zorder=6,
        )

    axis.set_title(
        f"W{start}→W{end}",
        fontsize=30.0,
        fontweight="bold",
        y=0.93,
        pad=0,
        color=INK,
    )
    axis.set_xlim(-0.10, 1.10)
    axis.set_ylim(-0.02, 1.00)
    axis.set_aspect("equal")
    axis.axis("off")


def render_network_changes(args: argparse.Namespace, *, assignments: pd.DataFrame | None = None) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    changes = build_change_summary(args.motif_dir, args.transition_mode, assignments=assignments)
    max_abs_transition = float(
        changes.loc[changes["feature_type"].eq("transition"), "mean_change_pp"].abs().max()
    )

    apply_style('network_changes')
    # Treat each condition row as an independent subfigure so that it carries
    # its own legend and internal layout.
    fig = plt.figure(figsize=(24.0, 17.0), facecolor="white")
    row_figures = fig.subfigures(2, 1, hspace=0.035)
    legend_handles = [
        Line2D([0], [0], color=INK, linewidth=6, linestyle="-", label="Probability increased"),
        Line2D([0], [0], color=DECREASE, linewidth=6, linestyle=(0, (5, 3)), label="Probability decreased"),
    ]
    for row_figure, condition in zip(row_figures, CONDITIONS):
        axes = row_figure.subplots(1, 3)
        for column_index, (start, end) in enumerate(INTERVALS):
            draw_panel(
                axes[column_index],
                changes,
                condition,
                start,
                end,
                max_abs_transition,
            )
        row_figure.text(
            0.500,
            0.995,
            CONDITION_LABELS[condition],
            ha="center",
            va="top",
            fontsize=32.0,
            fontweight="bold",
            color=INK,
        )
        row_figure.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.915),
            ncol=2,
            frameon=False,
            fontsize=25.0,
            handlelength=3.4,
            columnspacing=3.0,
        )
        row_figure.subplots_adjust(
            left=0.018,
            right=0.982,
            bottom=0.015,
            # Reserve a clearer band between each row-level legend and the
            # three adjacent-week panel titles below it.
            top=0.800,
            wspace=-0.075,
        )

    output_base = args.output_dir / "Fig_S6_early_motif_network_changes"
    save_figure(fig, output_base, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    changes.to_csv(args.output_dir / "Fig_S2_early_motif_network_change_values.csv", index=False)
    changes.to_csv(args.output_dir / "Fig_S6_early_motif_network_change_values.csv", index=False)
    (args.output_dir / "Fig_S6_manifest.json").write_text(json.dumps({
        "source": str(args.motif_dir.resolve()),
        "transition_mode": args.transition_mode,
        "occupancy": "mean posterior probability, matching main figure",
        "transition": "argmax label counts, 30-frame adjacency within recording, source-row normalization" if args.transition_mode == "hard" else "soft expected conditional probability",
        "change": "mean within-cage adjacent-week difference in percentage points",
        "display_unit": "% denotes subtraction of percentages, not relative percent change",
        "display": "all four occupancies and five largest absolute transition changes per panel",
        "main_transition_table_verified": args.transition_mode == "hard",
    }, indent=2), encoding="utf-8")


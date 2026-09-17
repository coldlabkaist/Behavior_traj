from __future__ import annotations
from pathlib import Path
import sys
from analysis.paths import ROOT
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from analysis.core.mom_pup.cluster_zone_movement import PNDS

from analysis.core.mom_pup.cluster_zone_movement import CONDITIONS

from analysis.core.mom_pup.cluster_zone_movement import ZONES

from analysis.core.mom_pup.cluster_zone_movement import COLORS

def _significance_stars(p_value: float) -> str:
	if p_value < 0.001:
		return "***"
	if p_value < 0.01:
		return "**"
	if p_value < 0.05:
		return "*"
	return "ns"

def _style_axis(ax: plt.Axes) -> None:
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)
	ax.tick_params(direction="out", width=1.2)

def plot_occupancy(
	metrics: pd.DataFrame,
	between: pd.DataFrame,
	*,
	threshold_mm: float,
	output_dir: Path,
) -> None:
	subset = metrics[metrics["threshold_mm"] == float(threshold_mm)].copy()
	fig, axes = plt.subplots(1, 3, figsize=(10.8, 4.2), sharey=True)
	x_base = np.arange(len(ZONES), dtype=float)
	bar_width = 0.34
	bar_offsets = {"Control": -bar_width / 2, "VPA": bar_width / 2}
	rng = np.random.default_rng(20260729)
	bracket_padding = 3.2

	for ax, pnd in zip(axes, PNDS):
		pnd_data = subset[subset["pnd"] == pnd]
		for condition in CONDITIONS:
			condition_data = pnd_data[pnd_data["condition"] == condition]
			means: list[float] = []
			sems: list[float] = []
			for zone_index, zone in enumerate(ZONES):
				values = pd.to_numeric(
					condition_data.loc[
						condition_data["zone"] == zone,
						"zone_occupancy_pct",
					],
					errors="coerce",
				).dropna()
				means.append(float(values.mean()))
				sems.append(float(values.sem()))
				x_position = x_base[zone_index] + bar_offsets[condition]
				jitter = rng.uniform(-0.045, 0.045, size=len(values))
				ax.scatter(
					np.full(len(values), x_position) + jitter,
					values,
					s=26,
					facecolors="white",
					edgecolors=COLORS[condition],
					linewidths=1.0,
					alpha=0.9,
					zorder=3,
				)
			ax.bar(
				x_base + bar_offsets[condition],
				means,
				yerr=sems,
				width=bar_width,
				color=COLORS[condition],
				alpha=0.86,
				capsize=4,
				label=condition if pnd == PNDS[0] else None,
				zorder=2,
			)

		for zone_index, zone in enumerate(ZONES):
			zone_values = pd.to_numeric(
				pnd_data.loc[
					pnd_data["zone"] == zone,
					"zone_occupancy_pct",
				],
				errors="coerce",
			).dropna()
			if zone_values.empty:
				continue
			bracket_y = float(zone_values.max()) + bracket_padding
			left = x_base[zone_index] + bar_offsets["Control"]
			right = x_base[zone_index] + bar_offsets["VPA"]
			ax.plot(
				[left, left, right, right],
				[
					bracket_y - bracket_padding * 0.25,
					bracket_y,
					bracket_y,
					bracket_y - bracket_padding * 0.25,
				],
				color="#222222",
				linewidth=1.1,
				zorder=4,
			)
			stat_row = between[
				(between["threshold_mm"] == float(threshold_mm))
				& (between["pnd"] == pnd)
				& (between["zone"] == zone)
			]
			if not stat_row.empty and pd.notna(stat_row.iloc[0]["p_holm"]):
				ax.text(
					(left + right) / 2,
					bracket_y + bracket_padding * 0.12,
					_significance_stars(float(stat_row.iloc[0]["p_holm"])),
					ha="center",
					va="bottom",
					fontsize=13,
				)
		ax.set_title(f"PND{pnd}")
		ax.set_xticks(x_base, [f"Near\n(≤{threshold_mm:g} mm)", "Outside"])
		_style_axis(ax)

	for ax in axes:
		ax.set_ylim(0, 112)

	axes[0].set_ylabel("Time in zone (% of valid time)")
	handles, labels = axes[0].get_legend_handles_labels()
	fig.legend(
		handles,
		labels,
		frameon=False,
		loc="center left",
		bbox_to_anchor=(0.94, 0.52),
	)
	fig.suptitle("Maternal occupancy near versus outside the pup cluster")
	fig.tight_layout(rect=(0, 0, 0.94, 0.94))
	fig.savefig(
		output_dir / "cluster_zone_occupancy_by_pnd.png",
		dpi=300,
		bbox_inches="tight",
	)
	fig.savefig(
		output_dir / "cluster_zone_occupancy_by_pnd.svg",
		bbox_inches="tight",
	)
	plt.close(fig)

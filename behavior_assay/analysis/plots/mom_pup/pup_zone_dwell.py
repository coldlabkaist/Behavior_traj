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

from analysis.core.mom_pup.pup_zone_dwell import CONDITIONS

from analysis.core.mom_pup.pup_zone_dwell import COLORS

from analysis.core.mom_pup.pup_zone_dwell import ZONES

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

def plot_dwell(
	metrics: pd.DataFrame,
	interaction: pd.DataFrame,
	zone_results: pd.DataFrame,
	output_dir: Path,
) -> None:
	fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.8))
	rng = np.random.default_rng(20260729)
	zone_x = np.arange(len(ZONES), dtype=float)
	offsets = {"Control": -0.18, "VPA": 0.18}
	width = 0.32
	plot_specs = (
		(
			"occupancy_pct",
			"Recording time in zone (%)",
			"Spatial time allocation",
		),
		(
			"mean_visit_duration_sec",
			"Mean duration per visit (s)",
			"Residence after zone entry",
		),
	)
	for ax, (metric, y_label, title) in zip(axes, plot_specs):
		zone_maxima: dict[str, float] = {}
		for zone_index, zone in enumerate(ZONES):
			zone_values: list[float] = []
			for condition in CONDITIONS:
				values = pd.to_numeric(
					metrics.loc[
						(metrics["condition"] == condition)
						& (metrics["zone"] == zone),
						metric,
					],
					errors="coerce",
				).dropna()
				zone_values.extend(values.to_list())
				x_position = zone_x[zone_index] + offsets[condition]
				ax.bar(
					x_position,
					values.mean(),
					yerr=values.sem(),
					width=width,
					color=COLORS[condition],
					alpha=0.86,
					edgecolor=COLORS[condition],
					linewidth=1.0,
					capsize=4,
					zorder=2,
				)
				jitter = rng.uniform(-0.045, 0.045, size=len(values))
				ax.scatter(
					np.full(len(values), x_position) + jitter,
					values,
					s=28,
					marker="o" if zone == "Pup zone" else "s",
					facecolors="white",
					edgecolors=COLORS[condition],
					linewidths=1.0,
					zorder=3,
				)
			zone_maxima[zone] = max(zone_values)
		all_values = pd.to_numeric(
			metrics[metric],
			errors="coerce",
		).dropna()
		y_max = float(all_values.max())
		y_min = min(0.0, float(all_values.min()))
		y_span = max(y_max - y_min, 1.0)
		max_bracket_y = y_max
		for zone_index, zone in enumerate(ZONES):
			left = zone_x[zone_index] + offsets["Control"]
			right = zone_x[zone_index] + offsets["VPA"]
			bracket_y = zone_maxima[zone] + 0.07 * y_span
			max_bracket_y = max(max_bracket_y, bracket_y)
			ax.plot(
				[left, left, right, right],
				[
					bracket_y - 0.015 * y_span,
					bracket_y,
					bracket_y,
					bracket_y - 0.015 * y_span,
				],
				color="#222222",
				linewidth=1.1,
			)
			stat_row = zone_results[
				(zone_results["metric"] == metric)
				& (zone_results["zone"] == zone)
			]
			if stat_row.empty:
				continue
			p_value = float(stat_row.iloc[0]["p_welch_two_sided"])
			if metric == "mean_visit_duration_sec":
				p_value = float(
					stat_row.iloc[0][
						"p_welch_analysis_scale_two_sided"
					]
				)
			ax.text(
				(left + right) / 2.0,
				bracket_y + 0.01 * y_span,
				_significance_stars(p_value),
				ha="center",
				va="bottom",
				fontsize=14,
			)
		interaction_row = interaction[interaction["metric"] == metric]
		if not interaction_row.empty:
			interaction_p = float(
				interaction_row.iloc[0][
					"p_welch_analysis_scale_two_sided"
				]
			)
			ax.text(
				0.98,
				0.98,
				f"Group × Zone: {_significance_stars(interaction_p)}",
				transform=ax.transAxes,
				ha="right",
				va="top",
				fontsize=10,
			)
		ax.set_ylim(y_min, max_bracket_y + 0.12 * y_span)
		ax.set_xticks(zone_x, ZONES)
		ax.set_ylabel(y_label)
		ax.set_title(title)
		_style_axis(ax)
	legend_handles = [
		plt.Rectangle((0, 0), 1, 1, color=COLORS[condition], label=condition)
		for condition in CONDITIONS
	]
	fig.legend(
		handles=legend_handles,
		frameon=False,
		loc="lower center",
		bbox_to_anchor=(0.5, 0.01),
		ncol=2,
	)
	fig.suptitle("Maternal dwell around the pup cluster at PND10")
	fig.tight_layout(rect=(0, 0.10, 1, 0.93))
	fig.savefig(
		output_dir / "pnd10_pup_zone_dwell.png",
		dpi=300,
		bbox_inches="tight",
	)
	fig.savefig(
		output_dir / "pnd10_pup_zone_dwell.svg",
		bbox_inches="tight",
	)
	plt.close(fig)

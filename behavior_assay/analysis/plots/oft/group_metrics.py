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

COLORS = {"Control": "#304F78", "VPA": "#C4475B"}

CONDITIONS = ("Control", "VPA")

CORE_METRICS = (
	("total_distance_cm", "Total distance", "Total distance (cm)"),
	("distance_centre_cm", "Centre distance", "Distance in centre (cm)"),
	("time_centre_s", "Centre time", "Time in centre (s)"),
)

def _significance_stars(value: float) -> str:
	if value < 0.001:
		return "***"
	if value < 0.01:
		return "**"
	if value < 0.05:
		return "*"
	return "ns"

def _style_axis(ax: plt.Axes) -> None:
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)
	ax.spines["left"].set_linewidth(1.5)
	ax.spines["bottom"].set_linewidth(1.5)
	ax.tick_params(
		axis="both",
		which="major",
		direction="out",
		labelsize=15,
		width=1.4,
		length=5,
	)

def _save_figure_bundle(fig: plt.Figure, output_dir: Path, stem: str) -> None:
	fig.savefig(
		output_dir / f"{stem}.png",
		dpi=300,
		bbox_inches="tight",
	)
	fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")

def plot_core_oft_bars(
	df: pd.DataFrame,
	tests: pd.DataFrame,
	output_dir: Path,
	*, p_column: str = "p_value",
) -> None:
	fig, axes = plt.subplots(1, 3, figsize=(12.0, 5.0))
	rng = np.random.default_rng(20260730)

	for ax, (metric, title, ylabel) in zip(axes, CORE_METRICS):
		values_by_condition: list[pd.Series] = []
		for condition in CONDITIONS:
			values = pd.to_numeric(
				df.loc[df["condition"] == condition, metric],
				errors="coerce",
			).dropna()
			values_by_condition.append(values)

		means = [float(values.mean()) for values in values_by_condition]
		sems = [float(values.sem()) for values in values_by_condition]
		x_positions = np.arange(len(CONDITIONS), dtype=float)
		for x_pos, condition, values, mean, sem in zip(
			x_positions,
			CONDITIONS,
			values_by_condition,
			means,
			sems,
		):
			ax.bar(
				x_pos,
				mean,
				yerr=sem,
				width=0.58,
				color=COLORS[condition],
				edgecolor=COLORS[condition],
				linewidth=1.2,
				alpha=0.88,
				capsize=5,
				error_kw={"elinewidth": 1.7, "capthick": 1.7},
				zorder=2,
			)
			jitter = rng.uniform(-0.085, 0.085, size=len(values))
			ax.scatter(
				np.full(len(values), x_pos) + jitter,
				values,
				s=38,
				facecolors="white",
				edgecolors=COLORS[condition],
				linewidths=1.25,
				zorder=3,
			)

		all_values = pd.concat(values_by_condition, ignore_index=True)
		data_max = float(all_values.max())
		bracket_y = data_max * 1.08
		bracket_height = max(data_max * 0.025, 1e-9)
		ax.plot(
			[0, 0, 1, 1],
			[
				bracket_y - bracket_height,
				bracket_y,
				bracket_y,
				bracket_y - bracket_height,
			],
			color="#222222",
			linewidth=1.4,
		)
		test = tests[
			(tests["analysis_unit"] == "animal")
			& (tests["metric"] == metric)
		]
		q_value = (
			float(test.iloc[0][p_column])
			if not test.empty and pd.notna(test.iloc[0][p_column])
			else np.nan
		)
		ax.text(
			0.5,
			bracket_y + bracket_height * 0.55,
			_significance_stars(q_value) if np.isfinite(q_value) else "ns",
			ha="center",
			va="bottom",
			fontsize=19,
		)
		ax.set_ylim(0, bracket_y + bracket_height * 3.2)
		ax.set_xticks(x_positions, CONDITIONS)
		ax.set_title(title, fontsize=20, pad=12)
		ax.set_ylabel(ylabel, fontsize=17, labelpad=9)
		_style_axis(ax)

	fig.suptitle("Open Field Test", fontsize=25, y=0.995)
	fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94), w_pad=2.8)
	_save_figure_bundle(fig, output_dir, "core_oft_metrics")
	plt.close(fig)

from __future__ import annotations
from pathlib import Path
import sys
from analysis.paths import ROOT
from dataclasses import dataclass
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

PNDS = (10, 15, 20)

CONDITIONS = ("Control", "VPA")

COLORS = {"Control": "#304F78", "VPA": "#C4475B"}

OFFSETS = {"Control": -0.17, "VPA": 0.17}

@dataclass(frozen=True)
class FigureStyle:
	figsize: tuple[float, float]
	title_size: float
	label_size: float
	tick_size: float
	legend_size: float
	star_size: float
	spine_width: float
	tick_width: float
	point_size: float
	main_line_width: float
	individual_line_width: float
	panel_gap: float

STYLES = {
	"talk": FigureStyle(
		figsize=(19.0, 7.2),
		title_size=38,
		label_size=31,
		tick_size=26,
		legend_size=26,
		star_size=31,
		spine_width=2.4,
		tick_width=2.2,
		point_size=58,
		main_line_width=3.5,
		individual_line_width=1.15,
		panel_gap=0.28,
	),
	"poster": FigureStyle(
		figsize=(17.2, 7.2),
		title_size=34,
		label_size=28,
		tick_size=24,
		legend_size=24,
		star_size=28,
		spine_width=2.3,
		tick_width=2.1,
		point_size=54,
		main_line_width=3.3,
		individual_line_width=1.1,
		panel_gap=0.25,
	),
}

def _stars(p_value: float) -> str:
	if p_value < 0.001:
		return "***"
	if p_value < 0.01:
		return "**"
	if p_value < 0.05:
		return "*"
	return "ns"

def _style_axis(ax: plt.Axes, style: FigureStyle) -> None:
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)
	for side in ("left", "bottom"):
		ax.spines[side].set_linewidth(style.spine_width)
	ax.tick_params(
		axis="both",
		direction="out",
		labelsize=style.tick_size,
		width=style.tick_width,
		length=8,
		pad=7,
	)

def _plot_proximity(
	ax: plt.Axes,
	metrics: pd.DataFrame,
	tests: pd.DataFrame,
	style: FigureStyle,
) -> None:
	x_positions = np.arange(len(PNDS), dtype=float)
	bar_offsets = {"Control": -0.19, "VPA": 0.19}
	bar_width = 0.34
	rng = np.random.default_rng(20260730)
	pnd_maxima: dict[int, float] = {}

	for pnd_index, pnd in enumerate(PNDS):
		pnd_values: list[float] = []
		for condition in CONDITIONS:
			values = pd.to_numeric(
				metrics.loc[
					(metrics["pnd"] == pnd)
					& (metrics["condition"] == condition),
					"pct_time_body_scale_proximity",
				],
				errors="coerce",
			).dropna()
			pnd_values.extend(values.to_list())
			x_value = x_positions[pnd_index] + bar_offsets[condition]
			ax.bar(
				x_value,
				values.mean(),
				yerr=values.sem(),
				width=bar_width,
				color=COLORS[condition],
				alpha=0.86,
				edgecolor=COLORS[condition],
				linewidth=1.4,
				capsize=5,
				error_kw={"elinewidth": 2.0, "capthick": 2.0},
				label=condition if pnd_index == 0 else None,
				zorder=2,
			)
			jitter = rng.uniform(-0.055, 0.055, size=len(values))
			ax.scatter(
				np.full(len(values), x_value) + jitter,
				values,
				s=style.point_size,
				facecolors="white",
				edgecolors=COLORS[condition],
				linewidths=1.55,
				zorder=3,
			)
		pnd_maxima[pnd] = max(pnd_values)

	stat_rows = tests.set_index("pnd")
	annotation_max = float(pd.to_numeric(metrics["pct_time_body_scale_proximity"], errors="coerce").max())
	for pnd_index, pnd in enumerate(PNDS):
		left = x_positions[pnd_index] + bar_offsets["Control"]
		right = x_positions[pnd_index] + bar_offsets["VPA"]
		bracket_y = pnd_maxima[pnd] + 5.0
		annotation_max = max(annotation_max, bracket_y + 6.0)
		ax.plot(
			[left, left, right, right],
			[bracket_y - 1.6, bracket_y, bracket_y, bracket_y - 1.6],
			color="#222222",
			linewidth=1.9,
		)
		p_value = float(stat_rows.loc[pnd, "p_welch_holm_across_pnd"])
		ax.text(
			x_positions[pnd_index],
			bracket_y + 1.1,
			_stars(p_value),
			ha="center",
			va="bottom",
			fontsize=style.star_size,
		)

	ax.set_ylim(0, min(120.0, annotation_max + 4.0))
	ax.set_xticks(x_positions, [f"PND {pnd}" for pnd in PNDS])
	ax.set_ylabel("Time near pup (%)", fontsize=style.label_size, labelpad=14)
	ax.set_title(
		"Maternal proximity across postnatal days",
		fontsize=style.title_size,
		pad=20,
	)
	ax.legend(
		frameon=False,
		loc="upper center",
		bbox_to_anchor=(0.81, 0.985),
		fontsize=style.legend_size,
		handlelength=1.5,
		borderaxespad=0.0,
	)
	_style_axis(ax, style)

def _plot_movement(
	ax: plt.Axes,
	metrics: pd.DataFrame,
	tests: pd.DataFrame,
	style: FigureStyle,
) -> None:
	rng = np.random.default_rng(20260728)
	metric = "avg_velocity_mm_s"

	for condition in CONDITIONS:
		group = metrics[metrics["condition"] == condition].copy()
		offset = OFFSETS[condition]
		for _, subject in group.groupby("subject_id"):
			subject = subject.sort_values("pnd")
			ax.plot(
				subject["pnd"].to_numpy(dtype=float) + offset,
				pd.to_numeric(subject[metric], errors="coerce"),
				color=COLORS[condition],
				alpha=0.16,
				linewidth=style.individual_line_width,
				zorder=1,
			)

		means: list[float] = []
		sems: list[float] = []
		for pnd in PNDS:
			values = pd.to_numeric(
				group.loc[group["pnd"] == pnd, metric],
				errors="coerce",
			).dropna()
			x = pnd + offset + rng.uniform(-0.045, 0.045, size=len(values))
			ax.scatter(
				x,
				values,
				s=style.point_size,
				facecolors="white",
				edgecolors=COLORS[condition],
				linewidths=1.55,
				alpha=0.95,
				zorder=2,
			)
			means.append(float(values.mean()))
			sems.append(float(values.sem()))
		ax.errorbar(
			np.asarray(PNDS, dtype=float) + offset,
			means,
			yerr=sems,
			color=COLORS[condition],
			marker="o",
			markersize=10,
			linewidth=style.main_line_width,
			elinewidth=2.2,
			capsize=5,
			capthick=2.2,
			zorder=3,
		)

	values_all = pd.to_numeric(metrics[metric], errors="coerce").dropna()
	y_max = float(values_all.max())
	ax.set_ylim(0, y_max * 1.20)
	stat_rows = tests[tests["metric"] == metric].set_index("pnd")
	for pnd in PNDS:
		p_value = stat_rows.loc[pnd, "p_holm_within_metric"]
		if pd.notna(p_value):
			ax.text(
				pnd,
				y_max * 1.075,
				_stars(float(p_value)),
				ha="center",
				va="bottom",
				fontsize=style.star_size,
			)

	ax.set_xticks(PNDS)
	ax.set_xlabel("Postnatal day", fontsize=style.label_size, labelpad=14)
	ax.set_ylabel("mom mean velocity (mm/s)", fontsize=style.label_size, labelpad=14)
	ax.set_title("Maternal movement", fontsize=style.title_size, pad=20)
	_style_axis(ax, style)

def _save_bundle(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
	output_dir.mkdir(parents=True, exist_ok=True)
	paths = [
		output_dir / f"{stem}.png",
		output_dir / f"{stem}.pdf",
		output_dir / f"{stem}.svg",
	]
	fig.savefig(paths[0], dpi=600, bbox_inches="tight", pad_inches=0.08)
	fig.savefig(paths[1], bbox_inches="tight", pad_inches=0.08)
	fig.savefig(paths[2], bbox_inches="tight", pad_inches=0.08)
	return paths

def create_summary_figure(
	proximity_metrics: pd.DataFrame,
	proximity_tests: pd.DataFrame,
	movement_metrics: pd.DataFrame,
	movement_tests: pd.DataFrame,
	output_dir: Path,
	profile: str,
) -> list[Path]:
	style = STYLES[profile]
	fig, axes = plt.subplots(
		1,
		2,
		figsize=style.figsize,
		gridspec_kw={"width_ratios": [0.94, 1.16]},
	)
	fig.subplots_adjust(
		left=0.075,
		right=0.985,
		bottom=0.17,
		top=0.88,
		wspace=style.panel_gap,
	)
	_plot_proximity(axes[0], proximity_metrics, proximity_tests, style)
	_plot_movement(axes[1], movement_metrics, movement_tests, style)
	stem = "talk_maternal_proximity_and_movement"
	if profile == "poster":
		stem += "_poster"
	paths = _save_bundle(fig, output_dir, stem)
	plt.close(fig)
	return paths

def create_individual_figures(
	proximity_metrics: pd.DataFrame,
	proximity_tests: pd.DataFrame,
	movement_metrics: pd.DataFrame,
	movement_tests: pd.DataFrame,
	output_dir: Path,
	profile: str,
) -> list[Path]:
	style = STYLES[profile]
	poster_suffix = "_poster" if profile == "poster" else ""
	paths: list[Path] = []

	proximity_size = (11.2, 7.5) if profile == "talk" else (10.5, 7.4)
	fig, ax = plt.subplots(figsize=proximity_size)
	fig.subplots_adjust(left=0.15, right=0.97, bottom=0.16, top=0.86)
	_plot_proximity(ax, proximity_metrics, proximity_tests, style)
	paths.extend(
		_save_bundle(
			fig,
			output_dir,
			f"talk_maternal_proximity{poster_suffix}",
		)
	)
	plt.close(fig)

	movement_size = (11.0, 7.5) if profile == "talk" else (10.3, 7.4)
	fig, ax = plt.subplots(figsize=movement_size)
	fig.subplots_adjust(left=0.17, right=0.98, bottom=0.18, top=0.86)
	_plot_movement(ax, movement_metrics, movement_tests, style)
	paths.extend(
		_save_bundle(
			fig,
			output_dir,
			f"talk_maternal_movement{poster_suffix}",
		)
	)
	plt.close(fig)
	return paths

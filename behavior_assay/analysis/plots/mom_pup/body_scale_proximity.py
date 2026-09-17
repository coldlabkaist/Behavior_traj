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

from analysis.core.mom_pup.body_scale_proximity import PNDS

from analysis.core.mom_pup.body_scale_proximity import CONDITIONS

from analysis.core.mom_pup.body_scale_proximity import COLORS

from analysis.core.mom_pup.body_scale_proximity import METRIC

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
	ax.spines["left"].set_linewidth(1.5)
	ax.spines["bottom"].set_linewidth(1.5)
	ax.tick_params(
		direction="out",
		width=1.4,
		length=5,
		labelsize=17,
	)

def plot_body_scale_proximity(
	metrics: pd.DataFrame,
	tests: pd.DataFrame,
	figure_dir: Path,
) -> None:
	fig, ax = plt.subplots(figsize=(7.6, 5.5))
	x_positions = np.arange(len(PNDS), dtype=float)
	offsets = {"Control": -0.19, "VPA": 0.19}
	width = 0.34
	rng = np.random.default_rng(20260730)

	pnd_maxima: dict[int, float] = {}
	for pnd_index, pnd in enumerate(PNDS):
		pnd_values: list[float] = []
		for condition in CONDITIONS:
			values = pd.to_numeric(
				metrics.loc[
					(metrics["pnd"] == pnd)
					& (metrics["condition"] == condition),
					METRIC,
				],
				errors="coerce",
			).dropna()
			pnd_values.extend(values.to_list())
			x_value = x_positions[pnd_index] + offsets[condition]
			ax.bar(
				x_value,
				values.mean(),
				yerr=values.sem(),
				width=width,
				color=COLORS[condition],
				alpha=0.86,
				edgecolor=COLORS[condition],
				linewidth=1.0,
				capsize=4,
				label=condition if pnd_index == 0 else None,
				zorder=2,
			)
			jitter = rng.uniform(-0.055, 0.055, size=len(values))
			ax.scatter(
				np.full(len(values), x_value) + jitter,
				values,
				s=31,
				facecolors="white",
				edgecolors=COLORS[condition],
				linewidths=1.05,
				zorder=3,
			)
		pnd_maxima[pnd] = max(pnd_values)

	data_max = float(pd.to_numeric(metrics[METRIC], errors="coerce").max())
	annotation_max = data_max
	stat_rows = tests.set_index("pnd")
	for pnd_index, pnd in enumerate(PNDS):
		left = x_positions[pnd_index] + offsets["Control"]
		right = x_positions[pnd_index] + offsets["VPA"]
		bracket_y = pnd_maxima[pnd] + 5.0
		annotation_max = max(annotation_max, bracket_y + 5.0)
		ax.plot(
			[left, left, right, right],
			[bracket_y - 1.5, bracket_y, bracket_y, bracket_y - 1.5],
			color="#222222",
			linewidth=1.1,
		)
		p_value = stat_rows.loc[pnd, "p_welch_holm_across_pnd"]
		ax.text(
			x_positions[pnd_index],
			bracket_y + 1.1,
			_significance_stars(float(p_value)),
			ha="center",
			va="bottom",
			fontsize=19,
		)

	ax.set_ylim(0, min(120.0, annotation_max + 3.0))
	ax.set_xticks(x_positions, [f"PND {pnd}" for pnd in PNDS])
	ax.set_ylabel("Time near pup (%)", fontsize=21, labelpad=12)
	ax.set_title(
		"Maternal proximity across postnatal days",
		fontsize=24,
		pad=15,
	)
	ax.legend(
		frameon=False,
		loc="upper center",
		bbox_to_anchor=(0.84, 0.98),
		fontsize=17,
	)
	_style_axis(ax)
	fig.tight_layout()
	figure_dir.mkdir(parents=True, exist_ok=True)
	fig.savefig(
		figure_dir / "body_scale_proximity_by_pnd.png",
		dpi=300,
		bbox_inches="tight",
	)
	fig.savefig(
		figure_dir / "body_scale_proximity_by_pnd_1200dpi.png",
		dpi=1200,
		bbox_inches="tight",
	)
	fig.savefig(
		figure_dir / "body_scale_proximity_by_pnd_600dpi.tiff",
		dpi=600,
		bbox_inches="tight",
		pil_kwargs={"compression": "tiff_lzw"},
	)
	fig.savefig(
		figure_dir / "body_scale_proximity_by_pnd.svg",
		bbox_inches="tight",
	)
	plt.close(fig)

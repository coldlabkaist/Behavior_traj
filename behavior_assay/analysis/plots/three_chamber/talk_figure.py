from __future__ import annotations
from pathlib import Path
import sys
from analysis.paths import ROOT
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analysis.core.three_chamber.talk_figure import _one_sample_p
from analysis.core.three_chamber.talk_figure import _p_marker
from analysis.core.three_chamber.talk_figure import _paired_p
from analysis.core.three_chamber.talk_figure import _sem

from analysis.core.three_chamber.talk_figure import FIGURE_DPI

from analysis.core.three_chamber.talk_figure import CONDITION_ORDER

from analysis.core.three_chamber.talk_figure import CONDITION_COLORS

from analysis.core.three_chamber.talk_figure import PHASE_CONFIG

def _clean_axis(ax: plt.Axes, *, zero_axis: bool = False) -> None:
	ax.grid(False)
	for side in ("top", "right"):
		ax.spines[side].set_visible(False)
	for side in ("left", "bottom"):
		ax.spines[side].set_linewidth(1.8)
		ax.spines[side].set_color("#333333")
	ax.tick_params(axis="both", width=2.0, length=7, labelsize=18, color="#333333")
	if zero_axis:
		ax.spines["bottom"].set_visible(False)
		ax.tick_params(axis="x", length=0)
		ax.axhline(0.0, color="#333333", linewidth=1.8, zorder=1)

def _sig_bracket(ax: plt.Axes, x1: float, x2: float, y: float, h: float, label: str) -> None:
	ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="#333333", linewidth=1.6, clip_on=False)
	ax.text((x1 + x2) / 2.0, y + h, label, ha="center", va="bottom", fontsize=16, fontweight="bold")

def _sig_line(ax: plt.Axes, x: float, y: float, label: str) -> None:
	ax.plot([x - 0.29, x + 0.29], [y, y], color="#333333", linewidth=1.6, clip_on=False)
	ax.text(x, y + 0.015, label, ha="center", va="bottom", fontsize=16, fontweight="bold")

def _save(fig: plt.Figure, output_path: Path) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	kwargs = {"facecolor": "white", "bbox_inches": "tight", "pad_inches": 0.08}
	fig.savefig(output_path, dpi=FIGURE_DPI, **kwargs)
	fig.savefig(output_path.with_suffix(".pdf"), format="pdf", **kwargs)
	fig.savefig(output_path.with_suffix(".svg"), format="svg", **kwargs)

def plot_talk_figure(sample_df: pd.DataFrame, *, phase: str, output_path: Path) -> None:
	cfg = PHASE_CONFIG[phase]
	phase_df = sample_df[sample_df["phase"].astype(str).str.lower().eq(phase)].copy()
	conditions = [condition for condition in CONDITION_ORDER if condition in set(phase_df["condition"].astype(str))]
	if not conditions:
		raise ValueError(f"No {phase} rows found in sample summary")

	fig, (ax_time, ax_pref) = plt.subplots(
		1,
		2,
		figsize=(7.4, 4.2),
		gridspec_kw={"width_ratios": [1.55, 0.78]},
		constrained_layout=False,
	)
	fig.subplots_adjust(left=0.13, right=0.985, bottom=0.23, top=0.77, wspace=0.31)

	within_gap = 0.60
	group_gap = 0.82
	time_values: list[float] = []
	pair_annotations: list[tuple[float, float, float, str]] = []
	for index, condition in enumerate(conditions):
		group = phase_df[phase_df["condition"].astype(str).eq(condition)]
		base_x = index * (within_gap + group_gap)
		left = pd.to_numeric(group[cfg["left_time"]], errors="coerce")
		right = pd.to_numeric(group[cfg["right_time"]], errors="coerce")
		means = [float(left.mean()), float(right.mean())]
		sems = [_sem(left), _sem(right)]
		positions = [base_x, base_x + within_gap]
		color = CONDITION_COLORS[condition]
		ax_time.bar(positions, means, width=0.46, color=color, edgecolor="none", zorder=2)
		ax_time.errorbar(positions, means, yerr=sems, fmt="none", ecolor="#262626", elinewidth=1.5, capsize=4, capthick=1.5, zorder=3)
		for position, values in zip(positions, (left, right)):
			points = values.dropna().to_numpy(dtype=float)
			jitter = np.linspace(-0.075, 0.075, max(len(points), 1))[: len(points)]
			ax_time.scatter(
				position + jitter,
				points,
				s=22,
				color=color,
				edgecolors="#202020",
				linewidths=1.05,
				zorder=4,
			)
			time_values.extend(float(value) for value in points if np.isfinite(value))
		ax_time.text(base_x + within_gap / 2.0, -0.17, condition, ha="center", va="top", fontsize=18, transform=ax_time.get_xaxis_transform())
		group_top = max(
			[max(time_values[-(left.notna().sum() + right.notna().sum()):], default=0.0), means[0] + sems[0], means[1] + sems[1]]
		)
		pair_annotations.append((base_x, base_x + within_gap, group_top, _p_marker(_paired_p(left, right))))

	bar_positions = [value for index in range(len(conditions)) for value in (index * (within_gap + group_gap), index * (within_gap + group_gap) + within_gap)]
	bar_labels = [label for _ in conditions for label in (cfg["left_label"], cfg["right_label"])]
	ax_time.set_xticks(bar_positions)
	ax_time.set_xticklabels(bar_labels)
	ax_time.set_ylabel("Total Investigation Time (s)", fontsize=20)
	ax_time.set_title("Investigation Time", fontsize=22, pad=11)
	_clean_axis(ax_time)
	time_max = max(time_values, default=1.0)
	time_range = max(time_max, 1.0)
	annotation_top = 0.0
	for x1, x2, group_top, marker in pair_annotations:
		y = group_top + 0.07 * time_range
		h = 0.025 * time_range
		_sig_bracket(ax_time, x1, x2, y, h, marker)
		annotation_top = max(annotation_top, y + h)
	ax_time.set_ylim(0.0, max(time_max + 0.18 * time_range, annotation_top + 0.13 * time_range))

	pref_x = np.arange(len(conditions), dtype=float)
	pref_all: list[float] = []
	for index, condition in enumerate(conditions):
		values = pd.to_numeric(
			phase_df[phase_df["condition"].astype(str).eq(condition)][cfg["preference"]],
			errors="coerce",
		).dropna()
		mean = float(values.mean())
		sem = _sem(values)
		color = CONDITION_COLORS[condition]
		ax_pref.bar(index, mean, width=0.52, color=color, edgecolor="none", zorder=2)
		ax_pref.errorbar(index, mean, yerr=sem, fmt="none", ecolor="#262626", elinewidth=1.5, capsize=4, capthick=1.5, zorder=3)
		points = values.to_numpy(dtype=float)
		jitter = np.linspace(-0.09, 0.09, max(len(points), 1))[: len(points)]
		ax_pref.scatter(
			index + jitter,
			points,
			s=22,
			color=color,
			edgecolors="#202020",
			linewidths=1.05,
			zorder=4,
		)
		pref_all.extend(float(value) for value in points if np.isfinite(value))

	ax_pref.set_xticks(pref_x)
	ax_pref.set_xticklabels(conditions)
	ax_pref.set_ylabel("Preference Index", fontsize=20)
	ax_pref.set_title("Preference Index", fontsize=22, pad=11)
	_clean_axis(ax_pref, zero_axis=True)
	pref_min = min(pref_all + [0.0])
	pref_max = max(pref_all + [0.0])
	pref_range = max(pref_max - pref_min, 0.2)
	line_y = pref_max + 0.10 * pref_range
	for index, condition in enumerate(conditions):
		values = phase_df[phase_df["condition"].astype(str).eq(condition)][cfg["preference"]]
		_sig_line(ax_pref, float(index), line_y, _p_marker(_one_sample_p(values)))
	ax_pref.set_ylim(pref_min - 0.10 * pref_range, line_y + 0.16 * pref_range)

	fig.suptitle(cfg["title"], fontsize=28, y=0.99)
	_save(fig, output_path)
	plt.close(fig)

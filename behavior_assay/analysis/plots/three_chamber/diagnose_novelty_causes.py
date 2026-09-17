from __future__ import annotations
from pathlib import Path
import sys
from analysis.paths import ROOT
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analysis.core.three_chamber.diagnose_novelty_causes import _clean
from analysis.core.three_chamber.diagnose_novelty_causes import _sem

from analysis.core.three_chamber.diagnose_novelty_causes import FIGURE_DPI

from analysis.core.three_chamber.diagnose_novelty_causes import CONDITION_ORDER

from analysis.core.three_chamber.diagnose_novelty_causes import CONDITION_COLORS

from analysis.core.three_chamber.diagnose_novelty_causes import SEX_MARKERS

def _style_axis(ax: plt.Axes) -> None:
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)
	ax.grid(False)
	ax.tick_params(labelsize=9)

def _scatter_with_mean(ax: plt.Axes, x: float, values: pd.Series, color: str, marker: str = "o") -> None:
	array = _clean(values)
	if not len(array):
		return
	jitter = np.linspace(-0.09, 0.09, len(array)) if len(array) > 1 else np.array([0.0])
	ax.scatter(np.full(len(array), x) + jitter, array, s=34, color=color, marker=marker, edgecolors="black", linewidths=0.45, alpha=0.85, zorder=3)
	mean = float(np.mean(array))
	sem = _sem(array)
	ax.errorbar([x], [mean], yerr=[sem], fmt="none", color="black", capsize=4, linewidth=1.5, zorder=4)
	ax.plot([x - 0.13, x + 0.13], [mean, mean], color="black", linewidth=2.0, zorder=4)

def _plot_correlation(ax: plt.Axes, sample_df: pd.DataFrame, x_col: str, title: str, xlabel: str) -> None:
	for condition in CONDITION_ORDER:
		group = sample_df[sample_df["condition"].eq(condition)][[x_col, "preference_index"]].dropna()
		ax.scatter(group[x_col], group["preference_index"], s=32, color=CONDITION_COLORS[condition], edgecolors="black", linewidths=0.4, alpha=0.82, label=condition)
		if len(group) >= 3 and group[x_col].nunique() >= 2:
			slope, intercept = np.polyfit(group[x_col], group["preference_index"], 1)
			x_values = np.linspace(group[x_col].min(), group[x_col].max(), 50)
			ax.plot(x_values, slope * x_values + intercept, color=CONDITION_COLORS[condition], linewidth=1.5)
	ax.axhline(0.0, color="#555555", linewidth=0.8)
	ax.set_title(title, fontsize=12)
	ax.set_xlabel(xlabel)
	ax.set_ylabel("Novel preference index")
	_style_axis(ax)

def _plot_qc(sample_df: pd.DataFrame, cage_df: pd.DataFrame, leave_one_df: pd.DataFrame, output_path: Path) -> None:
	fig, axes = plt.subplots(2, 3, figsize=(17.5, 10.5), dpi=FIGURE_DPI)
	date_pairs = sample_df[["condition", "date"]].drop_duplicates().sort_values(["condition", "date"])
	date_labels = [f"{row.condition[:1]}\n{row.date}" for row in date_pairs.itertuples(index=False)]
	for index, row in enumerate(date_pairs.itertuples(index=False)):
		group = sample_df[(sample_df["condition"].eq(row.condition)) & (sample_df["date"].eq(row.date))]
		for sex, sex_group in group.groupby("sex"):
			array = _clean(sex_group["preference_index"])
			jitter = np.linspace(-0.09, 0.09, len(array)) if len(array) > 1 else np.array([0.0])
			axes[0, 0].scatter(
				np.full(len(array), float(index)) + jitter,
				array,
				s=34,
				color=CONDITION_COLORS[row.condition],
				marker=SEX_MARKERS.get(sex, "o"),
				edgecolors="black",
				linewidths=0.45,
				alpha=0.85,
				zorder=3,
			)
		batch_values = _clean(group["preference_index"])
		batch_mean = float(np.mean(batch_values))
		axes[0, 0].errorbar([index], [batch_mean], yerr=[_sem(batch_values)], fmt="none", color="black", capsize=4, linewidth=1.5, zorder=4)
		axes[0, 0].plot([index - 0.13, index + 0.13], [batch_mean, batch_mean], color="black", linewidth=2.0, zorder=4)
	axes[0, 0].axhline(0.0, color="#555555", linewidth=0.8)
	axes[0, 0].set_xticks(range(len(date_labels)), date_labels, rotation=30, ha="right")
	axes[0, 0].set_title("Batch/date distribution", fontsize=12)
	axes[0, 0].set_ylabel("Novel preference index")
	_style_axis(axes[0, 0])

	sex_labels = [(condition, sex) for condition in CONDITION_ORDER for sex in ["f", "m"]]
	for index, (condition, sex) in enumerate(sex_labels):
		group = sample_df[(sample_df["condition"].eq(condition)) & (sample_df["sex"].eq(sex))]
		_scatter_with_mean(axes[0, 1], float(index), group["preference_index"], CONDITION_COLORS[condition], SEX_MARKERS[sex])
	axes[0, 1].axhline(0.0, color="#555555", linewidth=0.8)
	axes[0, 1].set_xticks(range(len(sex_labels)), [f"{condition}\n{sex.upper()}" for condition, sex in sex_labels])
	axes[0, 1].set_title("Sex-stratified distribution", fontsize=12)
	axes[0, 1].set_ylabel("Novel preference index")
	_style_axis(axes[0, 1])

	for index, condition in enumerate(CONDITION_ORDER):
		_scatter_with_mean(axes[0, 2], float(index), cage_df[cage_df["condition"].eq(condition)]["preference_index"], CONDITION_COLORS[condition])
	axes[0, 2].axhline(0.0, color="#555555", linewidth=0.8)
	axes[0, 2].set_xticks(range(len(CONDITION_ORDER)), CONDITION_ORDER)
	axes[0, 2].set_title("Cage-level means", fontsize=12)
	axes[0, 2].set_ylabel("Cage mean novel preference index")
	_style_axis(axes[0, 2])

	leave_plot = leave_one_df[~leave_one_df["omitted_date"].eq("ALL")].copy()
	colors = [CONDITION_COLORS.get(value, "#777777") for value in leave_plot["omitted_condition"]]
	axes[1, 0].scatter(range(len(leave_plot)), leave_plot["difference_vpa_minus_control"], c=colors, s=45, edgecolors="black", linewidths=0.45)
	overall_diff = float(leave_one_df.loc[leave_one_df["omitted_date"].eq("ALL"), "difference_vpa_minus_control"].iloc[0])
	axes[1, 0].axhline(overall_diff, color="black", linestyle="--", linewidth=1.2, label=f"Full data: {overall_diff:.3f}")
	axes[1, 0].axhline(0.0, color="#777777", linewidth=0.8)
	axes[1, 0].set_xticks(range(len(leave_plot)), leave_plot["omitted_date"], rotation=35, ha="right")
	axes[1, 0].set_title("Leave-one-batch-out", fontsize=12)
	axes[1, 0].set_ylabel("VPA - Control PI")
	axes[1, 0].legend(frameon=False, fontsize=8)
	_style_axis(axes[1, 0])

	if "nov_distance_per_min_norm" in sample_df.columns:
		_plot_correlation(axes[1, 1], sample_df, "nov_distance_per_min_norm", "Novelty PI vs locomotion", "Distance per min (normalized)")
	else:
		axes[1, 1].set_visible(False)
	if "open_spatial_preference_index" in sample_df.columns:
		_plot_correlation(axes[1, 2], sample_df, "open_spatial_preference_index", "Novelty PI vs open-session bias", "Open-session right-side PI")
	else:
		axes[1, 2].set_visible(False)

	handles = [
		plt.Line2D([0], [0], marker="o", linestyle="", color="#555555", markeredgecolor="black", label="Female"),
		plt.Line2D([0], [0], marker="s", linestyle="", color="#555555", markeredgecolor="black", label="Male"),
	]
	fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.985))
	fig.suptitle("Social novelty cause-tracing QC", fontsize=20, y=1.01)
	fig.tight_layout(rect=(0, 0, 1, 0.965))
	fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
	plt.close(fig)

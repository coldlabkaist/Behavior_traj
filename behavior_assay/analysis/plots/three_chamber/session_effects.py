from __future__ import annotations
from pathlib import Path
import sys
from analysis.paths import ROOT
import argparse
import math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from analysis.core.three_chamber.session_effects import _condition_color
from analysis.core.three_chamber.session_effects import _condition_offset
from analysis.core.three_chamber.session_effects import _condition_sort_key
from analysis.core.three_chamber.session_effects import _paired_ttest
from analysis.core.three_chamber.session_effects import _sem

from analysis.core.three_chamber.session_effects import FIGURE_DPI

from analysis.core.three_chamber.session_effects import PHASE_TITLES

from analysis.core.three_chamber.session_effects import METRICS

def _p_to_marker(pvalue: float) -> str:
	if not np.isfinite(pvalue):
		return "n/a"
	if pvalue < 1e-4:
		return "****"
	if pvalue < 1e-3:
		return "***"
	if pvalue < 1e-2:
		return "**"
	if pvalue < 5e-2:
		return "*"
	return "n.s."

def _summarize_stats(pairs_df: pd.DataFrame) -> pd.DataFrame:
	rows = []
	for phase in sorted(pairs_df["phase"].dropna().unique()):
		phase_df = pairs_df[pairs_df["phase"].eq(phase)]
		for condition in sorted(phase_df["condition"].dropna().unique(), key=_condition_sort_key):
			cond_df = phase_df[phase_df["condition"].eq(condition)]
			for metric, _label in METRICS:
				n1 = pd.to_numeric(cond_df[f"{metric}_n1"], errors="coerce")
				n2 = pd.to_numeric(cond_df[f"{metric}_n2"], errors="coerce")
				test = _paired_ttest(n1, n2)
				pair_df = pd.DataFrame({"n1": n1, "n2": n2}).dropna()
				rows.append(
					{
						"phase": phase,
						"condition": condition,
						"metric": metric,
						**test,
						"mean_n1": float(pair_df["n1"].mean()) if len(pair_df) else math.nan,
						"mean_n2": float(pair_df["n2"].mean()) if len(pair_df) else math.nan,
						"mean_delta_n2_minus_n1": float((pair_df["n2"] - pair_df["n1"]).mean()) if len(pair_df) else math.nan,
						"p_label": _p_to_marker(float(test["pvalue"])) if pd.notna(test["pvalue"]) else "n/a",
					}
				)
	return pd.DataFrame(rows)

def _plot_phase(
	pairs_df: pd.DataFrame,
	stats_df: pd.DataFrame,
	phase: str,
	conditions: list[str],
	args: argparse.Namespace,
	out_path: Path,
) -> None:
	phase_df = pairs_df[pairs_df["phase"].eq(phase)].copy()
	fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.5), dpi=FIGURE_DPI)
	axes = axes.ravel()
	for ax, (metric, label) in zip(axes, METRICS):
		for condition in conditions:
			cond_df = phase_df[phase_df["condition"].eq(condition)]
			if cond_df.empty:
				continue
			color = _condition_color(condition)
			offset = _condition_offset(condition)
			x1 = 1.0 + offset
			x2 = 2.0 + offset
			n1 = pd.to_numeric(cond_df[f"{metric}_n1"], errors="coerce")
			n2 = pd.to_numeric(cond_df[f"{metric}_n2"], errors="coerce")
			for y1, y2 in zip(n1, n2):
				if not np.isfinite(y1) or not np.isfinite(y2):
					continue
				ax.plot([x1, x2], [y1, y2], color=color, alpha=0.28, linewidth=1.0)
				ax.scatter([x1, x2], [y1, y2], color=color, edgecolor="black", linewidth=0.4, s=22, alpha=0.7, zorder=3)
			pair_df = pd.DataFrame({"n1": n1, "n2": n2}).dropna()
			if not pair_df.empty:
				means = [pair_df["n1"].mean(), pair_df["n2"].mean()]
				sems = [_sem(pair_df["n1"]), _sem(pair_df["n2"])]
				ax.errorbar(
					[x1, x2],
					means,
					yerr=sems,
					color=color,
					linewidth=3.0,
					marker="o",
					markersize=7,
					capsize=4,
					zorder=5,
				)
		if metric in {"preference_index", "delta_time_s"}:
			ax.axhline(0, color="#666666", linewidth=1.0)
		ax.set_title(label, fontsize=16)
		ax.set_xlim(0.65, 2.35)
		ax.set_xticks([1, 2])
		ax.set_xticklabels(["n1", "n2"], fontsize=12)
		ax.tick_params(axis="y", labelsize=11)
		ax.grid(axis="y", alpha=0.25)
		stat_lines = []
		for condition in conditions:
			row = stats_df[
				stats_df["phase"].eq(phase)
				& stats_df["condition"].eq(condition)
				& stats_df["metric"].eq(metric)
			]
			if row.empty:
				continue
			pvalue = float(row.iloc[0]["pvalue"]) if pd.notna(row.iloc[0]["pvalue"]) else math.nan
			stat_lines.append((condition, f"{condition}: {_p_to_marker(pvalue)} (p={pvalue:.3g})" if np.isfinite(pvalue) else f"{condition}: n/a"))
		for idx, (condition, text) in enumerate(stat_lines):
			ax.text(
				0.03,
				0.96 - idx * 0.08,
				text,
				transform=ax.transAxes,
				color=_condition_color(condition),
				fontsize=11,
				va="top",
				fontweight="bold",
			)
	axes[0].set_ylabel("Value", fontsize=13)
	axes[3].set_ylabel("Value", fontsize=13)
	legend_handles = [
		Line2D([0], [0], color=_condition_color(condition), marker="o", linewidth=3, label=condition)
		for condition in conditions
	]
	fig.legend(handles=legend_handles, loc="upper center", ncol=max(1, len(legend_handles)), frameon=False, fontsize=14)
	phase_title = PHASE_TITLES.get(phase, phase.upper())
	fig.suptitle(
		f"n1 vs n2 session effect QC - {phase_title} ({args.keypoint}, {args.roi_mode} x{args.radius_scale:.2f})",
		fontsize=22,
		y=0.995,
	)
	fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.92))
	out_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out_path, dpi=FIGURE_DPI)
	plt.close(fig)

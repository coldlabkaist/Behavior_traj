from __future__ import annotations
from pathlib import Path
import sys
from analysis.paths import ROOT
import math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analysis.core.three_chamber.roi_sensitivity_distance import _condition_color
from analysis.core.three_chamber.roi_sensitivity_distance import _condition_sort_key
from analysis.core.three_chamber.roi_sensitivity_distance import _onesample_ttest
from analysis.core.three_chamber.roi_sensitivity_distance import _paired_ttest
from analysis.core.three_chamber.roi_sensitivity_distance import _sem
from analysis.core.three_chamber.roi_sensitivity_distance import _welch_ttest

from analysis.core.three_chamber.roi_sensitivity_distance import FIGURE_DPI

from analysis.core.three_chamber.roi_sensitivity_distance import PHASE_CONFIG

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

def _summarize_sensitivity_stats(sample_df: pd.DataFrame) -> pd.DataFrame:
	rows = []
	for (phase, radius_scale, condition), group in sample_df.groupby(["phase", "radius_scale", "condition"], dropna=False):
		paired = _paired_ttest(group["opposite_time_s"], group["target_time_s"])
		onesample = _onesample_ttest(group["preference_index"], popmean=0.0)
		rows.append(
			{
				"analysis": "within_condition",
				"phase": phase,
				"radius_scale": radius_scale,
				"condition": condition,
				"metric": "target_vs_opposite_time",
				"comparison": f"{condition}: {PHASE_CONFIG[str(phase)]['opposite_label']} vs {PHASE_CONFIG[str(phase)]['target_label']}",
				**paired,
				"p_label": _p_to_marker(float(paired["pvalue"])) if pd.notna(paired["pvalue"]) else "n/a",
			}
		)
		rows.append(
			{
				"analysis": "within_condition",
				"phase": phase,
				"radius_scale": radius_scale,
				"condition": condition,
				"metric": "preference_index",
				"comparison": f"{condition} vs 0",
				**onesample,
				"p_label": _p_to_marker(float(onesample["pvalue"])) if pd.notna(onesample["pvalue"]) else "n/a",
			}
		)
	for (phase, radius_scale), group in sample_df.groupby(["phase", "radius_scale"], dropna=False):
		conditions = sorted(group["condition"].dropna().astype(str).unique(), key=_condition_sort_key)
		if len(conditions) < 2:
			continue
		a, b = conditions[0], conditions[1]
		ga = group[group["condition"].astype(str).eq(a)]
		gb = group[group["condition"].astype(str).eq(b)]
		for metric in ["preference_index", "target_time_s", "opposite_time_s"]:
			test = _welch_ttest(ga[metric], gb[metric])
			rows.append(
				{
					"analysis": "group_comparison",
					"phase": phase,
					"radius_scale": radius_scale,
					"condition": "",
					"metric": metric,
					"comparison": f"{a} vs {b}",
					**test,
					"p_label": _p_to_marker(float(test["pvalue"])) if pd.notna(test["pvalue"]) else "n/a",
				}
			)
	return pd.DataFrame(rows)

def _summarize_distance_stats(sample_df: pd.DataFrame) -> pd.DataFrame:
	rows = []
	for (phase, condition), group in sample_df.groupby(["phase", "condition"], dropna=False):
		for left_col, right_col, metric, comparison_suffix in [
			("target_distance_mean_norm", "opposite_distance_mean_norm", "mean_distance", "target vs opposite"),
			("target_distance_median_norm", "opposite_distance_median_norm", "median_distance", "target vs opposite"),
		]:
			test = _paired_ttest(group[left_col], group[right_col])
			rows.append(
				{
					"analysis": "within_condition",
					"phase": phase,
					"condition": condition,
					"metric": metric,
					"comparison": f"{condition}: {comparison_suffix}",
					**test,
					"p_label": _p_to_marker(float(test["pvalue"])) if pd.notna(test["pvalue"]) else "n/a",
				}
			)
		for metric_col in ["distance_preference_index", "median_distance_preference_index", "distance_delta_norm"]:
			test = _onesample_ttest(group[metric_col], popmean=0.0)
			rows.append(
				{
					"analysis": "within_condition",
					"phase": phase,
					"condition": condition,
					"metric": metric_col,
					"comparison": f"{condition} vs 0",
					**test,
					"p_label": _p_to_marker(float(test["pvalue"])) if pd.notna(test["pvalue"]) else "n/a",
				}
			)
	for phase, group in sample_df.groupby("phase", dropna=False):
		conditions = sorted(group["condition"].dropna().astype(str).unique(), key=_condition_sort_key)
		if len(conditions) < 2:
			continue
		a, b = conditions[0], conditions[1]
		ga = group[group["condition"].astype(str).eq(a)]
		gb = group[group["condition"].astype(str).eq(b)]
		for metric in ["distance_preference_index", "distance_delta_norm", "target_distance_mean_norm", "opposite_distance_mean_norm"]:
			test = _welch_ttest(ga[metric], gb[metric])
			rows.append(
				{
					"analysis": "group_comparison",
					"phase": phase,
					"condition": "",
					"metric": metric,
					"comparison": f"{a} vs {b}",
					**test,
					"p_label": _p_to_marker(float(test["pvalue"])) if pd.notna(test["pvalue"]) else "n/a",
				}
			)
	return pd.DataFrame(rows)

def _plot_sensitivity_curve(sample_df: pd.DataFrame, stats_df: pd.DataFrame, output_path: Path) -> None:
	phases = [phase for phase in ["soc", "nov"] if phase in set(sample_df["phase"].astype(str))]
	conditions = sorted(sample_df["condition"].dropna().astype(str).unique(), key=_condition_sort_key)
	fig, axes = plt.subplots(len(phases), 3, figsize=(16.0, 5.0 * max(len(phases), 1)), dpi=FIGURE_DPI, squeeze=False)
	for row_idx, phase in enumerate(phases):
		phase_df = sample_df[sample_df["phase"].astype(str).eq(phase)]
		ax_pi, ax_within, ax_group = axes[row_idx]
		for condition in conditions:
			cond_df = phase_df[phase_df["condition"].astype(str).eq(condition)]
			if cond_df.empty:
				continue
			summary = cond_df.groupby("radius_scale", as_index=False).agg(
				mean_pi=("preference_index", "mean"),
				sem_pi=("preference_index", _sem),
			)
			color = _condition_color(condition)
			ax_pi.errorbar(
				summary["radius_scale"],
				summary["mean_pi"],
				yerr=summary["sem_pi"],
				color=color,
				marker="o",
				linewidth=2.0,
				capsize=3,
				label=condition,
			)
			pvals = []
			scales = []
			for radius_scale in summary["radius_scale"]:
				row = stats_df[
					stats_df["analysis"].eq("within_condition")
					& stats_df["phase"].astype(str).eq(phase)
					& stats_df["condition"].astype(str).eq(condition)
					& stats_df["metric"].eq("preference_index")
					& np.isclose(pd.to_numeric(stats_df["radius_scale"], errors="coerce"), float(radius_scale))
				]
				if row.empty:
					continue
				pval = float(row.iloc[0]["pvalue"]) if pd.notna(row.iloc[0]["pvalue"]) else math.nan
				if np.isfinite(pval):
					scales.append(float(radius_scale))
					pvals.append(max(pval, 1e-12))
			if pvals:
				ax_within.plot(scales, -np.log10(pvals), color=color, marker="o", linewidth=2.0, label=condition)
		group_rows = stats_df[
			stats_df["analysis"].eq("group_comparison")
			& stats_df["phase"].astype(str).eq(phase)
			& stats_df["metric"].eq("preference_index")
		].copy()
		if not group_rows.empty:
			group_rows["radius_scale"] = pd.to_numeric(group_rows["radius_scale"], errors="coerce")
			group_rows["pvalue"] = pd.to_numeric(group_rows["pvalue"], errors="coerce")
			group_rows = group_rows.dropna(subset=["radius_scale", "pvalue"]).sort_values("radius_scale")
			ax_group.plot(
				group_rows["radius_scale"],
				-np.log10(np.maximum(group_rows["pvalue"], 1e-12)),
				color="#222222",
				marker="o",
				linewidth=2.0,
			)
		target = PHASE_CONFIG[phase]["target_label"]
		opposite = PHASE_CONFIG[phase]["opposite_label"]
		ax_pi.axhline(0.0, color="#666666", linewidth=1.0)
		ax_pi.set_title(f"{PHASE_CONFIG[phase]['title']}: {target} preference index", fontsize=15)
		ax_pi.set_xlabel("ROI radius scale")
		ax_pi.set_ylabel(f"({target} - {opposite}) / ({target} + {opposite})")
		ax_pi.grid(alpha=0.25)
		ax_pi.legend(frameon=False)
		for ax in [ax_within, ax_group]:
			ax.axhline(-math.log10(0.05), color="#aa3333", linestyle="--", linewidth=1.0, label="p=0.05")
			ax.set_xlabel("ROI radius scale")
			ax.set_ylabel("-log10(p)")
			ax.grid(alpha=0.25)
		ax_within.set_title("Within-group PI vs 0", fontsize=15)
		ax_group.set_title("Control vs VPA PI", fontsize=15)
		if row_idx == 0:
			ax_within.legend(frameon=False)
	fig.suptitle("ROI radius sensitivity", fontsize=22, y=0.995)
	fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.96))
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_path, dpi=FIGURE_DPI, facecolor="white")
	plt.close(fig)

def _draw_sig_bar(ax: plt.Axes, x1: float, x2: float, y: float, h: float, label: str) -> None:
	ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="black", linewidth=1.2, clip_on=False)
	ax.text((x1 + x2) / 2.0, y + h, label, ha="center", va="bottom", fontsize=11)

def _plot_distance_phase(sample_df: pd.DataFrame, stats_df: pd.DataFrame, phase: str, output_path: Path) -> None:
	cfg = PHASE_CONFIG[phase]
	phase_df = sample_df[sample_df["phase"].astype(str).eq(phase)].copy()
	conditions = sorted(phase_df["condition"].dropna().astype(str).unique(), key=_condition_sort_key)
	fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.0), dpi=FIGURE_DPI, constrained_layout=True)
	ax_mean, ax_delta, ax_pi = axes
	group_gap = 1.35
	within_gap = 0.85

	def plot_distance_pair(ax: plt.Axes) -> None:
		positions = []
		means = []
		sems = []
		colors = []
		labels = []
		ymax_values = []
		for idx, condition in enumerate(conditions):
			base_x = idx * (2.0 + group_gap)
			group = phase_df[phase_df["condition"].astype(str).eq(condition)]
			target = pd.to_numeric(group["target_distance_mean_norm"], errors="coerce")
			opposite = pd.to_numeric(group["opposite_distance_mean_norm"], errors="coerce")
			color = _condition_color(condition)
			for xbase, vals, label in [(base_x, opposite, cfg["opposite_label"]), (base_x + within_gap, target, cfg["target_label"])]:
				positions.append(xbase)
				means.append(float(vals.mean()))
				sems.append(_sem(vals))
				colors.append(color)
				labels.append(label)
				points = vals.dropna().to_numpy(dtype=float)
				if points.size:
					jitter = np.linspace(-0.08, 0.08, points.size)
					ax.scatter(np.full(points.size, xbase) + jitter, points, s=24, color=color, edgecolors="black", linewidths=0.4, zorder=4)
					ymax_values.extend([float(v) for v in points if np.isfinite(v)])
			ax.text(base_x + within_gap / 2.0, -0.17, condition, ha="center", va="top", fontsize=11, transform=ax.get_xaxis_transform())
			row = stats_df[
				stats_df["phase"].astype(str).eq(phase)
				& stats_df["condition"].astype(str).eq(condition)
				& stats_df["metric"].eq("mean_distance")
			]
			y = max([float(opposite.mean()), float(target.mean())], default=0.0)
			yerr = max([_sem(opposite), _sem(target)], default=0.0)
			label = _p_to_marker(float(row.iloc[0]["pvalue"])) if not row.empty and pd.notna(row.iloc[0]["pvalue"]) else "n/a"
			_draw_sig_bar(ax, base_x, base_x + within_gap, y + yerr + 0.02, 0.01, label)
		ax.bar(positions, means, width=0.62, color=colors, alpha=0.92, edgecolor="none")
		ax.errorbar(positions, means, yerr=sems, fmt="none", ecolor="black", elinewidth=1.2, capsize=4, capthick=1.2)
		ax.set_xticks(positions)
		ax.set_xticklabels(labels)
		ax.set_ylabel("Mean Nose-to-cup distance (norm)")
		ax.set_title("Nose-to-cup distance", fontsize=15)
		ax.grid(axis="y", alpha=0.22)

	def plot_single_metric(ax: plt.Axes, metric: str, ylabel: str, title: str) -> None:
		xs = np.arange(len(conditions), dtype=float)
		means = []
		sems = []
		for idx, condition in enumerate(conditions):
			group = phase_df[phase_df["condition"].astype(str).eq(condition)]
			vals = pd.to_numeric(group[metric], errors="coerce")
			means.append(float(vals.mean()))
			sems.append(_sem(vals))
			points = vals.dropna().to_numpy(dtype=float)
			if points.size:
				jitter = np.linspace(-0.08, 0.08, points.size)
				ax.scatter(np.full(points.size, xs[idx]) + jitter, points, s=24, color=_condition_color(condition), edgecolors="black", linewidths=0.4, zorder=4)
			row = stats_df[
				stats_df["phase"].astype(str).eq(phase)
				& stats_df["condition"].astype(str).eq(condition)
				& stats_df["metric"].eq(metric)
			]
			label = _p_to_marker(float(row.iloc[0]["pvalue"])) if not row.empty and pd.notna(row.iloc[0]["pvalue"]) else "n/a"
			y = means[-1] + sems[-1]
			ax.text(xs[idx], y + 0.04, label, ha="center", va="bottom", fontsize=11)
		ax.bar(xs, means, width=0.62, color=[_condition_color(c) for c in conditions], alpha=0.92, edgecolor="none")
		ax.errorbar(xs, means, yerr=sems, fmt="none", ecolor="black", elinewidth=1.2, capsize=4, capthick=1.2)
		ax.axhline(0.0, color="#666666", linewidth=1.0)
		ax.set_xticks(xs)
		ax.set_xticklabels(conditions)
		ax.set_ylabel(ylabel)
		ax.set_title(title, fontsize=15)
		ax.grid(axis="y", alpha=0.22)

	plot_distance_pair(ax_mean)
	plot_single_metric(ax_delta, "distance_delta_norm", f"{cfg['opposite_label']} distance - {cfg['target_label']} distance", "Distance delta")
	plot_single_metric(ax_pi, "distance_preference_index", "Distance preference index", "Distance PI")
	fig.suptitle(f"{cfg['title']} distance metrics ({cfg['target_label']} target)", fontsize=19, y=1.04)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_path, dpi=FIGURE_DPI, facecolor="white", bbox_inches="tight", pad_inches=0.16)
	plt.close(fig)

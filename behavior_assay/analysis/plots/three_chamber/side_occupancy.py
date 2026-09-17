from __future__ import annotations
from pathlib import Path
import sys
from analysis.paths import ROOT
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analysis.core.three_chamber.side_occupancy import _condition_sort_key
from analysis.core.three_chamber.side_occupancy import _onesample_ttest
from analysis.core.three_chamber.side_occupancy import _paired_ttest
from analysis.core.three_chamber.side_occupancy import _sem

from analysis.core.three_chamber.side_occupancy import FIGURE_DPI

from analysis.core.three_chamber.side_occupancy import PHASE_CONFIG

from analysis.core.three_chamber.side_occupancy import CONDITION_COLORS

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

def _draw_sig_bar(ax: plt.Axes, *, x1: float, x2: float, y: float, h: float, label: str) -> None:
	ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="black", linewidth=1.2, clip_on=False)
	ax.text((x1 + x2) / 2.0, y + h, label, ha="center", va="bottom", fontsize=11)

def _plot_phase_bars(
	sample_df: pd.DataFrame,
	*,
	phase: str,
	output_dir: Path,
	keypoint: str,
	file_suffix: str,
) -> tuple[Path, list[dict[str, object]]]:
	cfg = PHASE_CONFIG[phase]
	condition_order = sorted(sample_df["condition"].dropna().astype(str).unique().tolist(), key=_condition_sort_key)
	stats_rows: list[dict[str, object]] = []
	fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.0), constrained_layout=True)
	ax_time, ax_visit, ax_pref = axes
	group_gap = 1.35
	within_gap = 0.85

	def _plot_paired_panel(ax: plt.Axes, *, left_col: str, right_col: str, ylabel: str, title: str, panel_name: str) -> None:
		bar_positions: list[float] = []
		bar_heights: list[float] = []
		bar_errors: list[float] = []
		bar_colors: list[str] = []
		bar_labels: list[str] = []
		pair_positions: dict[str, tuple[float, float]] = {}
		finite_values: list[float] = []

		for idx, condition in enumerate(condition_order):
			base_x = idx * (2.0 + group_gap)
			group = sample_df[sample_df["condition"].astype(str) == condition].copy()
			left_vals = pd.to_numeric(group[left_col], errors="coerce")
			right_vals = pd.to_numeric(group[right_col], errors="coerce")
			color = CONDITION_COLORS.get(condition, "#777777")
			left_mean = float(left_vals.mean())
			right_mean = float(right_vals.mean())
			left_sem = _sem(left_vals)
			right_sem = _sem(right_vals)

			bar_positions.extend([base_x, base_x + within_gap])
			bar_heights.extend([left_mean, right_mean])
			bar_errors.extend([left_sem, right_sem])
			bar_colors.extend([color, color])
			bar_labels.extend([cfg["left_label"], cfg["right_label"]])
			pair_positions[str(condition)] = (base_x, base_x + within_gap)
			ax.text(base_x + within_gap / 2.0, -0.17, condition, ha="center", va="top", fontsize=11, transform=ax.get_xaxis_transform())

			for xbase, vals in ((base_x, left_vals), (base_x + within_gap, right_vals)):
				points = vals.dropna().to_numpy(dtype=float)
				if points.size:
					jitter = np.linspace(-0.08, 0.08, points.size)
					ax.scatter(np.full(points.size, xbase) + jitter, points, s=24, color=color, edgecolors="black", linewidths=0.4, zorder=4)
					finite_values.extend([float(v) for v in points if np.isfinite(v)])
			for top in (left_mean + left_sem, right_mean + right_sem):
				if np.isfinite(top):
					finite_values.append(float(top))

			stat_row = _paired_ttest(left_vals, right_vals)
			stats_rows.append(
				{
					"phase": phase,
					"panel": panel_name,
					"comparison": f"{condition}: {cfg['left_label']} vs {cfg['right_label']}",
					"test": stat_row["test"],
					"n": stat_row["n"],
					"statistic": stat_row["statistic"],
					"pvalue": stat_row["pvalue"],
				}
			)

		ax.bar(bar_positions, bar_heights, width=0.62, color=bar_colors, alpha=0.92, edgecolor="none")
		ax.errorbar(bar_positions, bar_heights, yerr=bar_errors, fmt="none", ecolor="black", elinewidth=1.2, capsize=4, capthick=1.2)
		ax.set_xticks(bar_positions)
		ax.set_xticklabels(bar_labels)
		ax.set_ylabel(ylabel)
		ax.set_title(title, fontsize=15)
		ax.grid(axis="y", alpha=0.22)

		ymax = max(finite_values, default=0.0)
		yrange = max(ymax, 1.0)
		for idx, condition in enumerate(condition_order):
			stat_row = next((row for row in stats_rows if row["panel"] == panel_name and str(row["comparison"]).startswith(f"{condition}:")), None)
			if stat_row is None:
				continue
			x1, x2 = pair_positions[str(condition)]
			_draw_sig_bar(
				ax,
				x1=x1,
				x2=x2,
				y=ymax + (0.08 + 0.12 * idx) * yrange,
				h=0.04 * yrange,
				label=_p_to_marker(float(stat_row["pvalue"])) if pd.notna(stat_row["pvalue"]) else "n/a",
			)
		ax.set_ylim(0.0, ymax + (0.34 + 0.12 * max(len(condition_order) - 1, 0)) * yrange)

	_plot_paired_panel(
		ax_time,
		left_col=cfg["time_left_col"],
		right_col=cfg["time_right_col"],
		ylabel="Side occupancy time (s)",
		title="Side occupancy time",
		panel_name="side_time_s",
	)
	_plot_paired_panel(
		ax_visit,
		left_col=cfg["visit_left_col"],
		right_col=cfg["visit_right_col"],
		ylabel="Side visit count",
		title="Side visit count",
		panel_name="side_visit_count",
	)

	pref_x = np.arange(len(condition_order), dtype=float)
	pref_means: list[float] = []
	pref_sems: list[float] = []
	pref_points: list[np.ndarray] = []
	for condition in condition_order:
		group = sample_df[sample_df["condition"].astype(str) == condition].copy()
		vals = pd.to_numeric(group[cfg["pref_col"]], errors="coerce")
		pref_means.append(float(vals.mean()))
		pref_sems.append(_sem(vals))
		pref_points.append(vals.dropna().to_numpy(dtype=float))
	ax_pref.bar(pref_x, pref_means, width=0.62, color=[CONDITION_COLORS.get(c, "#777777") for c in condition_order], alpha=0.92, edgecolor="none")
	ax_pref.errorbar(pref_x, pref_means, yerr=pref_sems, fmt="none", ecolor="black", elinewidth=1.2, capsize=4, capthick=1.2)
	for idx, condition in enumerate(condition_order):
		vals = pref_points[idx]
		if vals.size:
			jitter = np.linspace(-0.08, 0.08, vals.size)
			ax_pref.scatter(np.full(vals.size, pref_x[idx]) + jitter, vals, s=24, color=CONDITION_COLORS.get(condition, "#777777"), edgecolors="black", linewidths=0.4, zorder=4)
	ax_pref.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
	ax_pref.set_xticks(pref_x)
	ax_pref.set_xticklabels(condition_order)
	ax_pref.set_ylabel("Side preference index")
	ax_pref.set_title("Side preference index", fontsize=15)
	ax_pref.grid(axis="y", alpha=0.22)
	pref_stat_rows: list[dict[str, object]] = []
	for condition in condition_order:
		group = sample_df[sample_df["condition"].astype(str) == condition].copy()
		stat_row = _onesample_ttest(group[cfg["pref_col"]], popmean=0.0)
		pref_stat_rows.append(stat_row)
		stats_rows.append(
			{
				"phase": phase,
				"panel": "side_preference_index",
				"comparison": f"{condition} vs 0",
				"test": stat_row["test"],
				"n": stat_row["n"],
				"popmean": stat_row["popmean"],
				"statistic": stat_row["statistic"],
				"pvalue": stat_row["pvalue"],
			}
		)
	pref_finite = [0.0] + [float(v) for vals in pref_points for v in vals if np.isfinite(v)]
	for mean, sem in zip(pref_means, pref_sems):
		if np.isfinite(mean):
			pref_finite.append(float(mean))
		if np.isfinite(mean) and np.isfinite(sem):
			pref_finite.extend([float(mean + sem), float(mean - sem)])
	pref_ymin = min(pref_finite, default=0.0)
	pref_ymax = max(pref_finite, default=0.0)
	yrange = max(pref_ymax - pref_ymin, 0.25)
	for idx, stat_row in enumerate(pref_stat_rows):
		y = pref_means[idx] + pref_sems[idx]
		if not np.isfinite(y):
			y = 0.0
		ax_pref.text(
			float(pref_x[idx]),
			y + 0.06 * yrange,
			_p_to_marker(float(stat_row["pvalue"])) if pd.notna(stat_row["pvalue"]) else "n/a",
			ha="center",
			va="bottom",
			fontsize=11,
		)
	ax_pref.set_ylim(pref_ymin - 0.10 * yrange, pref_ymax + 0.25 * yrange)

	fig.suptitle(cfg["title"], fontsize=19, y=1.04)
	out_path = output_dir / f"side_occupancy__{phase}__{keypoint}{file_suffix}.png"
	fig.savefig(out_path, dpi=FIGURE_DPI, facecolor="white", bbox_inches="tight", pad_inches=0.16)
	plt.close(fig)
	return out_path, stats_rows

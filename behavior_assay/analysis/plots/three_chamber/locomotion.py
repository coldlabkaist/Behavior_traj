from __future__ import annotations
from pathlib import Path
import sys
from analysis.paths import ROOT
import math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats
from analysis.core.three_chamber.locomotion import _condition_sort_key
from analysis.core.three_chamber.locomotion import _onesample_ttest
from analysis.core.three_chamber.locomotion import _paired_ttest
from analysis.core.three_chamber.locomotion import _sem
from analysis.core.three_chamber.locomotion import _sex_color
from analysis.core.three_chamber.locomotion import _sex_label
from analysis.core.three_chamber.locomotion import _welch_ttest

from analysis.core.three_chamber.locomotion import FIGURE_DPI

from analysis.core.three_chamber.locomotion import PHASE_CONFIG

from analysis.core.three_chamber.locomotion import CONDITION_COLORS

from analysis.core.three_chamber.locomotion import SEX_ORDER

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

def _draw_sig_bar(
	ax: plt.Axes,
	*,
	x1: float,
	x2: float,
	y: float,
	h: float,
	label: str,
	fontsize: float = 11,
	linewidth: float = 1.2,
) -> None:
	ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="black", linewidth=linewidth, clip_on=False)
	ax.text((x1 + x2) / 2.0, y + h, label, ha="center", va="bottom", fontsize=fontsize)

def _plot_metric(
	session_df: pd.DataFrame,
	*,
	phase_list: list[str],
	metric_col: str,
	ylabel: str,
	title: str,
	output_path: Path,
	minimal_axes: bool = False,
) -> tuple[Path, list[dict[str, object]]]:
	phase_list = [phase for phase in phase_list if phase in set(session_df["phase"].astype(str).str.lower())]
	ncols = max(len(phase_list), 1)
	figsize = (5.5 * ncols, 6.4) if minimal_axes else (4.4 * ncols, 4.8)
	fig, axes = plt.subplots(1, ncols, figsize=figsize, constrained_layout=True)
	if ncols == 1:
		axes = [axes]
	stats_rows: list[dict[str, object]] = []

	for ax, phase in zip(axes, phase_list):
		cfg = PHASE_CONFIG.get(phase, {"title": phase.upper()})
		phase_df = session_df[session_df["phase"].astype(str).str.lower() == phase].copy()
		condition_order = sorted(phase_df["condition"].dropna().astype(str).unique().tolist(), key=_condition_sort_key)
		xpos = np.arange(len(condition_order), dtype=float)
		means: list[float] = []
		sems: list[float] = []

		for idx, condition in enumerate(condition_order):
			group = phase_df[phase_df["condition"].astype(str) == condition].copy()
			vals = pd.to_numeric(group[metric_col], errors="coerce")
			means.append(float(vals.mean()))
			sems.append(_sem(vals))
			point_vals = vals.dropna().to_numpy(dtype=float)
			if point_vals.size:
				jitter = np.linspace(-0.08, 0.08, point_vals.size)
				ax.scatter(
					np.full(point_vals.size, xpos[idx]) + jitter,
					point_vals,
					s=58 if minimal_axes else 34,
					color=CONDITION_COLORS.get(condition, "#777777"),
					edgecolors="black",
					linewidths=1.0 if minimal_axes else 0.5,
					zorder=4,
				)

		ax.bar(
			xpos,
			means,
			width=0.62,
			color=[CONDITION_COLORS.get(c, "#777777") for c in condition_order],
			alpha=0.92,
			edgecolor="none",
		)
		line_width = 1.8 if minimal_axes else 1.2
		cap_size = 6 if minimal_axes else 4
		ax.errorbar(
			xpos,
			means,
			yerr=sems,
			fmt="none",
			ecolor="black",
			elinewidth=line_width,
			capsize=cap_size,
			capthick=line_width,
		)
		ax.set_xticks(xpos)
		ax.set_xticklabels(condition_order, fontsize=20 if minimal_axes else None)
		ax.set_title(cfg["title"], fontsize=24 if minimal_axes else 15, pad=12 if minimal_axes else None)
		if minimal_axes:
			ax.grid(False)
			ax.spines["top"].set_visible(False)
			ax.spines["right"].set_visible(False)
			for spine_name in ("left", "bottom"):
				ax.spines[spine_name].set_color("#3A3A3A")
				ax.spines[spine_name].set_linewidth(1.8)
			ax.tick_params(axis="both", labelsize=18, width=1.6, length=6, color="#3A3A3A")
		else:
			ax.grid(axis="y", alpha=0.22)
		ax.set_ylabel(ylabel, fontsize=20 if minimal_axes else None, labelpad=10 if minimal_axes else None)
		mean_sem_max = max((m + s for m, s in zip(means, sems)), default=0.0)
		point_max = pd.to_numeric(phase_df[metric_col], errors="coerce").max()
		point_max = float(point_max) if pd.notna(point_max) else 0.0
		data_max = max(mean_sem_max, point_max, 1.0)
		axis_max = data_max * 1.12
		if len(condition_order) == 2:
			group_a = phase_df[phase_df["condition"].astype(str) == condition_order[0]].copy()
			group_b = phase_df[phase_df["condition"].astype(str) == condition_order[1]].copy()
			stat_row = _welch_ttest(group_a[metric_col], group_b[metric_col])
			stats_rows.append(
				{
					"phase": phase,
					"metric": metric_col,
					"comparison": f"{condition_order[0]} vs {condition_order[1]}",
					"test": stat_row["test"],
					"n_a": stat_row["n_a"],
					"n_b": stat_row["n_b"],
					"statistic": stat_row["statistic"],
					"pvalue": stat_row["pvalue"],
				}
			)
			bar_y = data_max * 1.08
			bar_h = max(data_max * 0.025, 0.02)
			_draw_sig_bar(
				ax,
				x1=float(xpos[0]),
				x2=float(xpos[1]),
				y=bar_y,
				h=bar_h,
				label=_p_to_marker(float(stat_row["pvalue"])) if pd.notna(stat_row["pvalue"]) else "n/a",
				fontsize=18 if minimal_axes else 11,
				linewidth=1.8 if minimal_axes else 1.2,
			)
			axis_max = bar_y + bar_h + data_max * 0.10
		ax.set_ylim(0.0, axis_max)

	fig.suptitle(title, fontsize=30 if minimal_axes else 19, y=1.10 if minimal_axes else 1.03)
	fig.savefig(output_path, dpi=FIGURE_DPI, facecolor="white", bbox_inches="tight", pad_inches=0.14)
	plt.close(fig)
	return output_path, stats_rows

def _plot_metric_by_sex(
	session_df: pd.DataFrame,
	*,
	phase_list: list[str],
	metric_col: str,
	ylabel: str,
	title: str,
	output_path: Path,
) -> tuple[Path | None, list[dict[str, object]]]:
	if "sex" not in session_df.columns:
		return None, []
	plot_df = session_df.copy()
	plot_df["sex"] = plot_df["sex"].astype(str).str.lower().str.strip()
	plot_df = plot_df[plot_df["sex"].isin(SEX_ORDER)].copy()
	phase_list = [phase for phase in phase_list if phase in set(plot_df["phase"].astype(str).str.lower())]
	if plot_df.empty or not phase_list:
		return None, []

	ncols = max(len(phase_list), 1)
	fig, axes = plt.subplots(1, ncols, figsize=(4.9 * ncols, 4.9), constrained_layout=True)
	if ncols == 1:
		axes = [axes]
	stats_rows: list[dict[str, object]] = []

	for ax, phase in zip(axes, phase_list):
		cfg = PHASE_CONFIG.get(phase, {"title": phase.upper()})
		phase_df = plot_df[plot_df["phase"].astype(str).str.lower() == phase].copy()
		condition_order = sorted(phase_df["condition"].dropna().astype(str).unique().tolist(), key=_condition_sort_key)
		bar_positions: list[float] = []
		bar_heights: list[float] = []
		bar_errors: list[float] = []
		bar_colors: list[str] = []
		bar_labels: list[str] = []
		position_lookup: dict[tuple[str, str], float] = {}
		finite_values: list[float] = []
		xpos = 0.0

		for condition in condition_order:
			condition_df = phase_df[phase_df["condition"].astype(str) == condition].copy()
			for sex in SEX_ORDER:
				group = condition_df[condition_df["sex"] == sex].copy()
				if group.empty:
					continue
				vals = pd.to_numeric(group[metric_col], errors="coerce")
				mean = float(vals.mean())
				sem = _sem(vals)
				bar_positions.append(xpos)
				bar_heights.append(mean)
				bar_errors.append(sem)
				bar_colors.append(_sex_color(sex))
				bar_labels.append(f"{condition}\n{_sex_label(sex)}")
				position_lookup[(condition, sex)] = xpos
				points = vals.dropna().to_numpy(dtype=float)
				if points.size:
					jitter = np.linspace(-0.08, 0.08, points.size)
					ax.scatter(np.full(points.size, xpos) + jitter, points, s=30, color=_sex_color(sex), edgecolors="black", linewidths=0.45, zorder=4)
					finite_values.extend([float(v) for v in points if np.isfinite(v)])
				if np.isfinite(mean):
					finite_values.append(mean)
				if np.isfinite(mean) and np.isfinite(sem):
					finite_values.append(mean + sem)
				xpos += 1.0
			xpos += 0.65

		ax.bar(bar_positions, bar_heights, width=0.62, color=bar_colors, alpha=0.9, edgecolor="none")
		ax.errorbar(bar_positions, bar_heights, yerr=bar_errors, fmt="none", ecolor="black", elinewidth=1.2, capsize=4, capthick=1.2)
		ax.set_xticks(bar_positions)
		ax.set_xticklabels(bar_labels)
		ax.set_title(cfg["title"], fontsize=15)
		ax.grid(axis="y", alpha=0.22)
		ax.set_ylabel(ylabel)

		ymax = max(finite_values, default=1.0)
		yrange = max(ymax, 1.0)
		for idx, condition in enumerate(condition_order):
			m_vals = phase_df[(phase_df["condition"].astype(str) == condition) & (phase_df["sex"] == "m")][metric_col]
			f_vals = phase_df[(phase_df["condition"].astype(str) == condition) & (phase_df["sex"] == "f")][metric_col]
			stat_row = _welch_ttest(m_vals, f_vals)
			stats_rows.append(
				{
					"phase": phase,
					"sex": "m_vs_f",
					"panel": "sex_effect",
					"metric": metric_col,
					"comparison": f"{condition}: M vs F",
					"test": stat_row["test"],
					"n_a": stat_row["n_a"],
					"n_b": stat_row["n_b"],
					"statistic": stat_row["statistic"],
					"pvalue": stat_row["pvalue"],
				}
			)
			if (condition, "m") in position_lookup and (condition, "f") in position_lookup:
				_draw_sig_bar(
					ax,
					x1=float(position_lookup[(condition, "m")]),
					x2=float(position_lookup[(condition, "f")]),
					y=ymax + (0.08 + 0.10 * idx) * yrange,
					h=0.035 * yrange,
					label=_p_to_marker(float(stat_row["pvalue"])) if pd.notna(stat_row["pvalue"]) else "n/a",
				)
		ax.set_ylim(0.0, max(ymax + (0.30 + 0.10 * max(len(condition_order) - 1, 0)) * yrange, 1.0))

	legend_handles = [
		Line2D([0], [0], marker="o", linestyle="", color=_sex_color("m"), markeredgecolor="black", label="Male"),
		Line2D([0], [0], marker="o", linestyle="", color=_sex_color("f"), markeredgecolor="black", label="Female"),
	]
	fig.legend(handles=legend_handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02), fontsize=10)
	fig.suptitle(f"{title} - sex stratified", fontsize=19, y=1.08)
	fig.savefig(output_path, dpi=FIGURE_DPI, facecolor="white", bbox_inches="tight", pad_inches=0.14)
	plt.close(fig)
	return output_path, stats_rows

def _plot_open_spatial_preference(sample_df: pd.DataFrame, *, output_path: Path) -> tuple[Path | None, list[dict[str, object]]]:
	open_df = sample_df[sample_df["phase"].astype(str).str.lower() == "hab_open"].copy()
	if open_df.empty:
		return None, []

	condition_order = sorted(open_df["condition"].dropna().astype(str).unique().tolist(), key=_condition_sort_key)
	fig, axes = plt.subplots(1, 2, figsize=(11.0, 6.4), constrained_layout=True)
	ax_time, ax_pref = axes
	stats_rows: list[dict[str, object]] = []

	group_gap = 1.25
	within_gap = 0.82
	bar_positions: list[float] = []
	bar_heights: list[float] = []
	bar_errors: list[float] = []
	bar_colors: list[str] = []
	bar_labels: list[str] = []
	pair_positions: dict[str, tuple[float, float]] = {}
	finite_time_values: list[float] = []

	for idx, condition in enumerate(condition_order):
		base_x = idx * (2.0 + group_gap)
		group = open_df[open_df["condition"].astype(str) == condition].copy()
		left_vals = pd.to_numeric(group["left_zone_time_s"], errors="coerce")
		right_vals = pd.to_numeric(group["right_zone_time_s"], errors="coerce")
		color = CONDITION_COLORS.get(condition, "#777777")
		left_mean = float(left_vals.mean())
		right_mean = float(right_vals.mean())
		left_sem = _sem(left_vals)
		right_sem = _sem(right_vals)

		bar_positions.extend([base_x, base_x + within_gap])
		bar_heights.extend([left_mean, right_mean])
		bar_errors.extend([left_sem, right_sem])
		bar_colors.extend([color, color])
		bar_labels.extend(["L", "R"])
		pair_positions[condition] = (base_x, base_x + within_gap)
		ax_time.text(
			base_x + within_gap / 2.0,
			-0.16,
			condition,
			ha="center",
			va="top",
			fontsize=20,
			transform=ax_time.get_xaxis_transform(),
		)

		left_points = left_vals.dropna().to_numpy(dtype=float)
		right_points = right_vals.dropna().to_numpy(dtype=float)
		if left_points.size:
			jitter = np.linspace(-0.08, 0.08, left_points.size)
			ax_time.scatter(np.full(left_points.size, base_x) + jitter, left_points, s=58, color=color, edgecolors="black", linewidths=1.0, zorder=4)
		if right_points.size:
			jitter = np.linspace(-0.08, 0.08, right_points.size)
			ax_time.scatter(np.full(right_points.size, base_x + within_gap) + jitter, right_points, s=58, color=color, edgecolors="black", linewidths=1.0, zorder=4)
		for value in np.concatenate([left_points, right_points]):
			if np.isfinite(value):
				finite_time_values.append(float(value))
		for top in (left_mean + left_sem, right_mean + right_sem):
			if np.isfinite(top):
				finite_time_values.append(float(top))

		stat_row = _paired_ttest(left_vals, right_vals)
		stats_rows.append(
			{
				"phase": "hab_open",
				"panel": "left_right_time_s",
				"comparison": f"{condition}: L vs R",
				"test": stat_row["test"],
				"n": stat_row["n"],
				"statistic": stat_row["statistic"],
				"pvalue": stat_row["pvalue"],
			}
		)

	ax_time.bar(bar_positions, bar_heights, width=0.6, color=bar_colors, alpha=0.92, edgecolor="none")
	ax_time.errorbar(bar_positions, bar_heights, yerr=bar_errors, fmt="none", ecolor="black", elinewidth=1.8, capsize=6, capthick=1.8)
	ax_time.set_xticks(bar_positions)
	ax_time.set_xticklabels(bar_labels, fontsize=20)
	ax_time.set_ylabel("Zone Time (s)", fontsize=22, labelpad=10)
	ax_time.set_title("Open Session Side Occupancy", fontsize=24, pad=12)
	ax_time.grid(False)
	ax_time.spines["top"].set_visible(False)
	ax_time.spines["right"].set_visible(False)
	for spine_name in ("left", "bottom"):
		ax_time.spines[spine_name].set_color("#3A3A3A")
		ax_time.spines[spine_name].set_linewidth(1.8)
	ax_time.tick_params(axis="both", labelsize=18, width=1.6, length=6, color="#3A3A3A")
	ymax = max(finite_time_values, default=1.0)
	yrange = max(ymax, 1.0)
	bar_y = ymax + 0.10 * yrange
	bar_h = 0.025 * yrange
	for condition in condition_order:
		stat_row = next((row for row in stats_rows if row["panel"] == "left_right_time_s" and str(row["comparison"]).startswith(f"{condition}:")), None)
		if stat_row is None:
			continue
		x1, x2 = pair_positions[condition]
		_draw_sig_bar(
			ax_time,
			x1=x1,
			x2=x2,
			y=bar_y,
			h=bar_h,
			label=_p_to_marker(float(stat_row["pvalue"])) if pd.notna(stat_row["pvalue"]) else "n/a",
			fontsize=18,
			linewidth=1.8,
		)
	ax_time.set_ylim(0.0, bar_y + bar_h + 0.12 * yrange)

	pref_x = np.arange(len(condition_order), dtype=float)
	pref_means: list[float] = []
	pref_sems: list[float] = []
	pref_points: list[np.ndarray] = []
	for condition in condition_order:
		group = open_df[open_df["condition"].astype(str) == condition].copy()
		vals = pd.to_numeric(group["spatial_preference_index"], errors="coerce")
		pref_means.append(float(vals.mean()))
		pref_sems.append(_sem(vals))
		pref_points.append(vals.dropna().to_numpy(dtype=float))
		stat_row = _onesample_ttest(vals, popmean=0.0)
		stats_rows.append(
			{
				"phase": "hab_open",
				"panel": "spatial_preference_index",
				"comparison": f"{condition} vs 0",
				"test": stat_row["test"],
				"n": stat_row["n"],
				"popmean": stat_row["popmean"],
				"statistic": stat_row["statistic"],
				"pvalue": stat_row["pvalue"],
			}
		)

	ax_pref.bar(pref_x, pref_means, width=0.62, color=[CONDITION_COLORS.get(c, "#777777") for c in condition_order], alpha=0.92, edgecolor="none")
	ax_pref.errorbar(pref_x, pref_means, yerr=pref_sems, fmt="none", ecolor="black", elinewidth=1.8, capsize=6, capthick=1.8)
	pref_finite: list[float] = [0.0]
	for idx, condition in enumerate(condition_order):
		vals = pref_points[idx]
		if vals.size:
			jitter = np.linspace(-0.08, 0.08, vals.size)
			ax_pref.scatter(np.full(vals.size, pref_x[idx]) + jitter, vals, s=58, color=CONDITION_COLORS.get(condition, "#777777"), edgecolors="black", linewidths=1.0, zorder=4)
		pref_finite.extend([float(v) for v in vals if np.isfinite(v)])
	ax_pref.axhline(0.0, color="#3A3A3A", linewidth=1.8)
	ax_pref.set_xticks(pref_x)
	ax_pref.set_xticklabels(condition_order, fontsize=20)
	ax_pref.set_ylabel("Spatial Preference Index", fontsize=22, labelpad=10)
	ax_pref.set_title("Right-Side Preference", fontsize=24, pad=12)
	ax_pref.grid(False)
	ax_pref.spines["top"].set_visible(False)
	ax_pref.spines["right"].set_visible(False)
	for spine_name in ("left", "bottom"):
		ax_pref.spines[spine_name].set_color("#3A3A3A")
		ax_pref.spines[spine_name].set_linewidth(1.8)
	ax_pref.tick_params(axis="both", labelsize=18, width=1.6, length=6, color="#3A3A3A")
	pref_ymin = min(pref_finite)
	pref_ymax = max(pref_finite)
	yrange = max(pref_ymax - pref_ymin, 0.4)
	pref_bar_y = pref_ymax + 0.10 * yrange
	pref_bar_h = 0.025 * yrange
	for idx, condition in enumerate(condition_order):
		stat_row = next((row for row in stats_rows if row["panel"] == "spatial_preference_index" and row["comparison"] == f"{condition} vs 0"), None)
		if stat_row is None:
			continue
		_draw_sig_bar(
			ax_pref,
			x1=float(pref_x[idx] - 0.28),
			x2=float(pref_x[idx] + 0.28),
			y=pref_bar_y,
			h=pref_bar_h,
			label=_p_to_marker(float(stat_row["pvalue"])) if pd.notna(stat_row["pvalue"]) else "n/a",
			fontsize=18,
			linewidth=1.8,
		)
	ax_pref.set_ylim(pref_ymin - 0.12 * yrange, pref_bar_y + pref_bar_h + 0.12 * yrange)

	fig.suptitle("Habituation Open Spatial Preference", fontsize=30, y=1.10)
	fig.savefig(output_path, dpi=FIGURE_DPI, facecolor="white", bbox_inches="tight", pad_inches=0.14)
	plt.close(fig)
	return output_path, stats_rows

def _analyze_open_spatial_preference_strength(
	sample_df: pd.DataFrame,
	*,
	thresholds: tuple[float, ...] = (0.10, 0.20, 0.30),
	primary_threshold: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	open_df = sample_df[sample_df["phase"].astype(str).str.lower() == "hab_open"].copy()
	columns = ["condition", "subject_id", "trial_id", "sex", "spatial_preference_index"]
	individual_df = open_df[[col for col in columns if col in open_df.columns]].copy()
	individual_df["spatial_preference_index"] = pd.to_numeric(
		individual_df["spatial_preference_index"], errors="coerce"
	)
	individual_df = individual_df.dropna(subset=["spatial_preference_index"]).copy()
	individual_df["absolute_preference_index"] = individual_df["spatial_preference_index"].abs()
	individual_df["preference_threshold"] = float(primary_threshold)
	individual_df["preference_class"] = np.select(
		[
			individual_df["spatial_preference_index"] <= -primary_threshold,
			individual_df["spatial_preference_index"] >= primary_threshold,
		],
		["left", "right"],
		default="none",
	)
	individual_df["has_side_preference"] = individual_df["preference_class"] != "none"

	group_rows: list[dict[str, object]] = []
	stats_rows: list[dict[str, object]] = []
	conditions = sorted(
		individual_df["condition"].dropna().astype(str).unique().tolist(),
		key=_condition_sort_key,
	)
	for threshold in thresholds:
		classified = individual_df.copy()
		classified["preference_class"] = np.select(
			[
				classified["spatial_preference_index"] <= -threshold,
				classified["spatial_preference_index"] >= threshold,
			],
			["left", "right"],
			default="none",
		)
		classified["has_side_preference"] = classified["preference_class"] != "none"
		for condition in conditions:
			group = classified[classified["condition"].astype(str) == condition]
			counts = group["preference_class"].value_counts()
			n_total = int(len(group))
			n_preference = int(group["has_side_preference"].sum())
			group_rows.append(
				{
					"threshold": float(threshold),
					"condition": condition,
					"n_total": n_total,
					"n_left": int(counts.get("left", 0)),
					"n_none": int(counts.get("none", 0)),
					"n_right": int(counts.get("right", 0)),
					"n_with_preference": n_preference,
					"pct_with_preference": 100.0 * n_preference / n_total if n_total else math.nan,
					"median_absolute_pi": float(group["absolute_preference_index"].median()),
				}
			)

		if len(conditions) == 2:
			group_a = classified[classified["condition"].astype(str) == conditions[0]]
			group_b = classified[classified["condition"].astype(str) == conditions[1]]
			table = [
				[
					int(group_a["has_side_preference"].sum()),
					int((~group_a["has_side_preference"]).sum()),
				],
				[
					int(group_b["has_side_preference"].sum()),
					int((~group_b["has_side_preference"]).sum()),
				],
			]
			odds_ratio, fisher_p = stats.fisher_exact(table, alternative="two-sided")
			stats_rows.append(
				{
					"threshold": float(threshold),
					"comparison": f"{conditions[0]} vs {conditions[1]}",
					"group_a": conditions[0],
					"group_b": conditions[1],
					"metric": "side_preference_prevalence",
					"test": "Fisher exact test",
					"n_a": int(len(group_a)),
					"n_b": int(len(group_b)),
					"statistic": float(odds_ratio),
					"pvalue": float(fisher_p),
					"p_label": _p_to_marker(float(fisher_p)),
				}
			)

	if len(conditions) == 2:
		group_a = individual_df[individual_df["condition"].astype(str) == conditions[0]]
		group_b = individual_df[individual_df["condition"].astype(str) == conditions[1]]
		mw = stats.mannwhitneyu(
			group_a["absolute_preference_index"],
			group_b["absolute_preference_index"],
			alternative="two-sided",
			method="auto",
		)
		stats_rows.append(
			{
				"threshold": math.nan,
				"comparison": f"{conditions[0]} vs {conditions[1]}",
				"group_a": conditions[0],
				"group_b": conditions[1],
				"metric": "absolute_preference_index",
				"test": "Mann-Whitney U test",
				"n_a": int(len(group_a)),
				"n_b": int(len(group_b)),
				"statistic": float(mw.statistic),
				"pvalue": float(mw.pvalue),
				"p_label": _p_to_marker(float(mw.pvalue)),
			}
		)

	return individual_df, pd.DataFrame(group_rows), pd.DataFrame(stats_rows)

def _plot_open_spatial_preference_by_sex(sample_df: pd.DataFrame, *, output_path: Path) -> tuple[Path | None, list[dict[str, object]]]:
	if "sex" not in sample_df.columns:
		return None, []
	open_df = sample_df[sample_df["phase"].astype(str).str.lower() == "hab_open"].copy()
	open_df["sex"] = open_df["sex"].astype(str).str.lower().str.strip()
	open_df = open_df[open_df["sex"].isin(SEX_ORDER)].copy()
	if open_df.empty:
		return None, []

	condition_order = sorted(open_df["condition"].dropna().astype(str).unique().tolist(), key=_condition_sort_key)
	fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.9), constrained_layout=True)
	ax_time, ax_pref = axes
	stats_rows: list[dict[str, object]] = []
	group_gap = 1.05
	within_gap = 0.74

	bar_positions: list[float] = []
	bar_heights: list[float] = []
	bar_errors: list[float] = []
	bar_colors: list[str] = []
	bar_labels: list[str] = []
	pair_positions: dict[tuple[str, str], tuple[float, float]] = {}
	finite_time_values: list[float] = []
	xpos = 0.0

	for condition in condition_order:
		condition_df = open_df[open_df["condition"].astype(str) == condition].copy()
		for sex in SEX_ORDER:
			group = condition_df[condition_df["sex"] == sex].copy()
			if group.empty:
				continue
			left_vals = pd.to_numeric(group["left_zone_time_s"], errors="coerce")
			right_vals = pd.to_numeric(group["right_zone_time_s"], errors="coerce")
			color = _sex_color(sex)
			left_mean = float(left_vals.mean())
			right_mean = float(right_vals.mean())
			left_sem = _sem(left_vals)
			right_sem = _sem(right_vals)
			bar_positions.extend([xpos, xpos + within_gap])
			bar_heights.extend([left_mean, right_mean])
			bar_errors.extend([left_sem, right_sem])
			bar_colors.extend([color, color])
			bar_labels.extend(["L", "R"])
			pair_positions[(condition, sex)] = (xpos, xpos + within_gap)
			ax_time.text(xpos + within_gap / 2.0, -0.18, f"{condition}\n{_sex_label(sex)}", ha="center", va="top", fontsize=10, transform=ax_time.get_xaxis_transform())

			for xbase, vals in ((xpos, left_vals), (xpos + within_gap, right_vals)):
				points = vals.dropna().to_numpy(dtype=float)
				if points.size:
					jitter = np.linspace(-0.07, 0.07, points.size)
					ax_time.scatter(np.full(points.size, xbase) + jitter, points, s=28, color=color, edgecolors="black", linewidths=0.45, zorder=4)
					finite_time_values.extend([float(v) for v in points if np.isfinite(v)])
			for top in (left_mean + left_sem, right_mean + right_sem):
				if np.isfinite(top):
					finite_time_values.append(float(top))
			stat_row = _paired_ttest(left_vals, right_vals)
			stats_rows.append(
				{
					"phase": "hab_open",
					"sex": sex,
					"panel": "left_right_time_s",
					"metric": "left_zone_time_s vs right_zone_time_s",
					"comparison": f"{condition} {_sex_label(sex)}: L vs R",
					"test": stat_row["test"],
					"n": stat_row["n"],
					"statistic": stat_row["statistic"],
					"pvalue": stat_row["pvalue"],
				}
			)
			xpos += 2.0 + group_gap

	ax_time.bar(bar_positions, bar_heights, width=0.58, color=bar_colors, alpha=0.9, edgecolor="none")
	ax_time.errorbar(bar_positions, bar_heights, yerr=bar_errors, fmt="none", ecolor="black", elinewidth=1.1, capsize=3.5, capthick=1.1)
	ax_time.set_xticks(bar_positions)
	ax_time.set_xticklabels(bar_labels)
	ax_time.set_ylabel("Zone time (s)")
	ax_time.set_title("Open side occupancy by sex", fontsize=15)
	ax_time.grid(axis="y", alpha=0.22)
	ymax = max(finite_time_values, default=1.0)
	yrange = max(ymax, 1.0)
	for idx, key in enumerate(pair_positions):
		stat_row = next((row for row in stats_rows if row["panel"] == "left_right_time_s" and row["comparison"].startswith(f"{key[0]} {_sex_label(key[1])}:")), None)
		if stat_row is None:
			continue
		x1, x2 = pair_positions[key]
		_draw_sig_bar(
			ax_time,
			x1=x1,
			x2=x2,
			y=ymax + (0.08 + 0.06 * idx) * yrange,
			h=0.035 * yrange,
			label=_p_to_marker(float(stat_row["pvalue"])) if pd.notna(stat_row["pvalue"]) else "n/a",
		)
	ax_time.set_ylim(0.0, ymax + (0.26 + 0.06 * max(len(pair_positions) - 1, 0)) * yrange)

	pref_positions: list[float] = []
	pref_means: list[float] = []
	pref_sems: list[float] = []
	pref_colors: list[str] = []
	pref_labels: list[str] = []
	position_lookup: dict[tuple[str, str], float] = {}
	pref_finite: list[float] = [0.0]
	xpos = 0.0
	for condition in condition_order:
		condition_df = open_df[open_df["condition"].astype(str) == condition].copy()
		for sex in SEX_ORDER:
			group = condition_df[condition_df["sex"] == sex].copy()
			if group.empty:
				continue
			vals = pd.to_numeric(group["spatial_preference_index"], errors="coerce")
			mean = float(vals.mean())
			sem = _sem(vals)
			pref_positions.append(xpos)
			pref_means.append(mean)
			pref_sems.append(sem)
			pref_colors.append(_sex_color(sex))
			pref_labels.append(f"{condition}\n{_sex_label(sex)}")
			position_lookup[(condition, sex)] = xpos
			points = vals.dropna().to_numpy(dtype=float)
			if points.size:
				jitter = np.linspace(-0.07, 0.07, points.size)
				ax_pref.scatter(np.full(points.size, xpos) + jitter, points, s=28, color=_sex_color(sex), edgecolors="black", linewidths=0.45, zorder=4)
				pref_finite.extend([float(v) for v in points if np.isfinite(v)])
			if np.isfinite(mean):
				pref_finite.append(mean)
			if np.isfinite(mean) and np.isfinite(sem):
				pref_finite.extend([mean + sem, mean - sem])
			stat_row = _onesample_ttest(vals, popmean=0.0)
			stats_rows.append(
				{
					"phase": "hab_open",
					"sex": sex,
					"panel": "spatial_preference_index",
					"metric": "spatial_preference_index",
					"comparison": f"{condition} {_sex_label(sex)} vs 0",
					"test": stat_row["test"],
					"n": stat_row["n"],
					"popmean": stat_row["popmean"],
					"statistic": stat_row["statistic"],
					"pvalue": stat_row["pvalue"],
				}
			)
			xpos += 1.0
		xpos += 0.65

	ax_pref.bar(pref_positions, pref_means, width=0.62, color=pref_colors, alpha=0.9, edgecolor="none")
	ax_pref.errorbar(pref_positions, pref_means, yerr=pref_sems, fmt="none", ecolor="black", elinewidth=1.1, capsize=3.5, capthick=1.1)
	ax_pref.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
	ax_pref.set_xticks(pref_positions)
	ax_pref.set_xticklabels(pref_labels)
	ax_pref.set_ylabel("Spatial preference index")
	ax_pref.set_title("Right-side preference by sex", fontsize=15)
	ax_pref.grid(axis="y", alpha=0.22)
	pref_ymin = min(pref_finite)
	pref_ymax = max(pref_finite)
	yrange = max(pref_ymax - pref_ymin, 0.4)
	for idx, condition in enumerate(condition_order):
		m_vals = open_df[(open_df["condition"].astype(str) == condition) & (open_df["sex"] == "m")]["spatial_preference_index"]
		f_vals = open_df[(open_df["condition"].astype(str) == condition) & (open_df["sex"] == "f")]["spatial_preference_index"]
		stat_row = _welch_ttest(m_vals, f_vals)
		stats_rows.append(
			{
				"phase": "hab_open",
				"sex": "m_vs_f",
				"panel": "spatial_preference_index_sex_effect",
				"metric": "spatial_preference_index",
				"comparison": f"{condition}: M vs F",
				"test": stat_row["test"],
				"n_a": stat_row["n_a"],
				"n_b": stat_row["n_b"],
				"statistic": stat_row["statistic"],
				"pvalue": stat_row["pvalue"],
			}
		)
		if (condition, "m") in position_lookup and (condition, "f") in position_lookup:
			_draw_sig_bar(
				ax_pref,
				x1=float(position_lookup[(condition, "m")]),
				x2=float(position_lookup[(condition, "f")]),
				y=pref_ymax + (0.10 + 0.10 * idx) * yrange,
				h=0.04 * yrange,
				label=_p_to_marker(float(stat_row["pvalue"])) if pd.notna(stat_row["pvalue"]) else "n/a",
			)
	ax_pref.set_ylim(pref_ymin - 0.10 * yrange, pref_ymax + (0.34 + 0.10 * max(len(condition_order) - 1, 0)) * yrange)

	legend_handles = [
		Line2D([0], [0], marker="o", linestyle="", color=_sex_color("m"), markeredgecolor="black", label="Male"),
		Line2D([0], [0], marker="o", linestyle="", color=_sex_color("f"), markeredgecolor="black", label="Female"),
	]
	fig.legend(handles=legend_handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02), fontsize=10)
	fig.suptitle("Habituation open spatial preference - sex stratified", fontsize=18, y=1.08)
	fig.savefig(output_path, dpi=FIGURE_DPI, facecolor="white", bbox_inches="tight", pad_inches=0.14)
	plt.close(fig)
	return output_path, stats_rows

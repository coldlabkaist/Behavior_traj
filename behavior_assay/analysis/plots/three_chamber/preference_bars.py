from __future__ import annotations
from pathlib import Path
import sys
from analysis.paths import ROOT
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from analysis.core.three_chamber.preference_bars import _brown_forsythe_test
from analysis.core.three_chamber.preference_bars import _condition_sort_key
from analysis.core.three_chamber.preference_bars import _onesample_ttest
from analysis.core.three_chamber.preference_bars import _paired_ttest
from analysis.core.three_chamber.preference_bars import _roi_output_tag
from analysis.core.three_chamber.preference_bars import _sem
from analysis.core.three_chamber.preference_bars import _sex_color
from analysis.core.three_chamber.preference_bars import _sex_label
from analysis.core.three_chamber.preference_bars import _welch_ttest

from analysis.core.three_chamber.preference_bars import FIGURE_DPI

from analysis.core.three_chamber.preference_bars import PHASE_CONFIG

from analysis.core.three_chamber.preference_bars import CONDITION_COLORS

from analysis.core.three_chamber.preference_bars import SEX_ORDER

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
	ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="#3A3A3A", linewidth=1.2, clip_on=False)
	ax.text((x1 + x2) / 2.0, y + h, label, ha="center", va="bottom", fontsize=11)

def _draw_sig_line(ax: plt.Axes, *, x: float, y: float, half_width: float, label: str) -> None:
	ax.plot([x - half_width, x + half_width], [y, y], color="#3A3A3A", linewidth=1.2, clip_on=False)
	ax.text(x, y, label, ha="center", va="bottom", fontsize=11)

def _clean_axis(ax: plt.Axes) -> None:
	ax.grid(False)
	ax.set_facecolor("white")
	for side in ("top", "right"):
		ax.spines[side].set_visible(False)
	for side in ("left", "bottom"):
		ax.spines[side].set_linewidth(1.35)
		ax.spines[side].set_color("#3A3A3A")
	ax.tick_params(axis="both", width=1.25, length=5.0, color="#3A3A3A", labelsize=13)

def _use_zero_as_xaxis(ax: plt.Axes) -> None:
	ax.spines["bottom"].set_visible(False)
	ax.tick_params(axis="x", length=0)
	ax.axhline(0.0, color="#3A3A3A", linewidth=1.35, alpha=1.0, zorder=1)

def _save_figure(fig: plt.Figure, out_path: Path, *, pad_inches: float) -> None:
	save_kwargs = {
		"facecolor": "white",
		"bbox_inches": "tight",
		"pad_inches": pad_inches,
	}
	fig.savefig(out_path, dpi=FIGURE_DPI, **save_kwargs)
	fig.savefig(out_path.with_suffix(".svg"), format="svg", **save_kwargs)

def _plot_phase_bars(
	session_df: pd.DataFrame,
	*,
	phase: str,
	output_dir: Path,
	keypoint: str,
	roi_mode: str,
	radius_scale: float,
	file_suffix: str,
	shared_panel_ylims: dict[str, float] | None = None,
) -> tuple[Path, list[dict[str, object]]]:
	cfg = PHASE_CONFIG[phase]
	condition_order = sorted(session_df["condition"].dropna().astype(str).unique().tolist(), key=_condition_sort_key)
	stats_rows: list[dict[str, object]] = []
	fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.55), constrained_layout=False)
	fig.subplots_adjust(left=0.082, right=0.992, bottom=0.18, top=0.82, wspace=0.17)
	ax_time, ax_visit, ax_pref = axes
	pref_position = ax_pref.get_position()
	ax_pref.set_position(
		[
			pref_position.x0 + 0.035,
			pref_position.y0,
			pref_position.width - 0.035,
			pref_position.height,
		]
	)
	group_gap = 0.68
	within_gap = 0.58

	def _plot_paired_panel(
		ax: plt.Axes,
		*,
		left_col: str,
		right_col: str,
		ylabel: str,
		title: str,
		panel_name: str,
	) -> None:
		bar_positions: list[float] = []
		bar_heights: list[float] = []
		bar_errors: list[float] = []
		bar_colors: list[str] = []
		bar_labels: list[str] = []
		scatter_specs: list[tuple[np.ndarray, np.ndarray, str]] = []
		pair_positions: dict[str, tuple[float, float]] = {}
		pair_tops: dict[str, float] = {}
		finite_values: list[float] = []

		for idx, condition in enumerate(condition_order):
			base_x = idx * (within_gap + group_gap)
			group = session_df[session_df["condition"].astype(str) == condition].copy()
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

			left_points = left_vals.dropna().to_numpy(dtype=float)
			right_points = right_vals.dropna().to_numpy(dtype=float)
			jitter_left = np.linspace(-0.08, 0.08, max(len(left_points), 1))
			jitter_right = np.linspace(-0.08, 0.08, max(len(right_points), 1))
			scatter_specs.append((base_x + jitter_left[: len(left_points)], left_points, color))
			scatter_specs.append((base_x + within_gap + jitter_right[: len(right_points)], right_points, color))
			ax.text(base_x + within_gap / 2.0, -0.125, condition, ha="center", va="top", fontsize=13, transform=ax.get_xaxis_transform())

			paired_stats = _paired_ttest(left_vals, right_vals)
			stats_rows.append(
				{
					"phase": phase,
					"panel": panel_name,
					"comparison": f"{condition}: {cfg['left_label']} vs {cfg['right_label']}",
					"test": paired_stats["test"],
					"n": paired_stats["n"],
					"statistic": paired_stats["statistic"],
					"pvalue": paired_stats["pvalue"],
				}
			)

			for value in np.concatenate([left_points, right_points]):
				if np.isfinite(value):
					finite_values.append(float(value))
			for top in (left_mean + left_sem, right_mean + right_sem):
				if np.isfinite(top):
					finite_values.append(float(top))
			group_finite = [
				float(value)
				for value in np.concatenate([left_points, right_points, np.asarray([left_mean + left_sem, right_mean + right_sem])])
				if np.isfinite(value)
			]
			pair_tops[str(condition)] = max(group_finite, default=0.0)

		ax.bar(bar_positions, bar_heights, width=0.46, color=bar_colors, edgecolor="none")
		ax.errorbar(bar_positions, bar_heights, yerr=bar_errors, fmt="none", ecolor="#303030", elinewidth=1.2, capsize=3.5, capthick=1.2)
		for xs, ys, color in scatter_specs:
			if ys.size:
				ax.scatter(xs, ys, s=12, color=color, edgecolors="#202020", linewidths=0.8, zorder=4)
		ax.set_xticks(bar_positions)
		ax.set_xticklabels(bar_labels)
		ax.set_ylabel(ylabel, fontsize=14)
		ax.set_title(title, fontsize=15, pad=7)
		_clean_axis(ax)

		ymax = max(finite_values, default=0.0)
		yrange = max(ymax, 1.0)
		h = 0.025 * yrange
		annotation_tops: list[float] = []
		for condition in condition_order:
			stat_row = next((row for row in stats_rows if row["panel"] == panel_name and str(row["comparison"]).startswith(f"{condition}:")), None)
			if stat_row is None:
				continue
			x1, x2 = pair_positions[str(condition)]
			sig_y = pair_tops.get(str(condition), ymax) + 0.055 * yrange
			_draw_sig_bar(
				ax,
				x1=x1,
				x2=x2,
				y=sig_y,
				h=h,
				label=_p_to_marker(float(stat_row["pvalue"])) if pd.notna(stat_row["pvalue"]) else "n/a",
			)
			annotation_tops.append(sig_y + h)
		top_limit = max(max(annotation_tops, default=ymax) + 0.12 * yrange, ymax + 0.18 * yrange, 1.0)
		if shared_panel_ylims is not None and panel_name in shared_panel_ylims:
			top_limit = max(top_limit, float(shared_panel_ylims[panel_name]))
		ax.set_ylim(0.0, top_limit)

	_plot_paired_panel(
		ax_time,
		left_col=cfg["time_left_col"],
		right_col=cfg["time_right_col"],
		ylabel="Total Investigation Time (s)",
		title="Investigation Time",
		panel_name="investigation_time_s",
	)
	_plot_paired_panel(
		ax_visit,
		left_col=cfg["visit_left_col"],
		right_col=cfg["visit_right_col"],
		ylabel="Visit count",
		title="Visit Count",
		panel_name="visit_count",
	)

	pref_x = np.arange(len(condition_order), dtype=float)
	pref_means: list[float] = []
	pref_sems: list[float] = []
	pref_points: list[np.ndarray] = []
	for condition in condition_order:
		group = session_df[session_df["condition"].astype(str) == condition].copy()
		vals = pd.to_numeric(group[cfg["pref_col"]], errors="coerce")
		pref_means.append(float(vals.mean()))
		pref_sems.append(_sem(vals))
		pref_points.append(vals.dropna().to_numpy(dtype=float))
	ax_pref.bar(pref_x, pref_means, width=0.52, color=[CONDITION_COLORS.get(c, "#777777") for c in condition_order], edgecolor="none")
	ax_pref.errorbar(pref_x, pref_means, yerr=pref_sems, fmt="none", ecolor="#303030", elinewidth=1.2, capsize=3.5, capthick=1.2)
	for idx, condition in enumerate(condition_order):
		vals = pref_points[idx]
		if vals.size:
			jitter = np.linspace(-0.08, 0.08, vals.size)
			ax_pref.scatter(np.full(vals.size, pref_x[idx]) + jitter, vals, s=12, color=CONDITION_COLORS.get(condition, "#777777"), edgecolors="#202020", linewidths=0.8, zorder=4)
	ax_pref.set_xticks(pref_x)
	ax_pref.set_xticklabels(condition_order)
	ax_pref.set_ylabel("Preference Index", fontsize=14)
	ax_pref.set_title("Preference Index", fontsize=15, pad=7)
	_clean_axis(ax_pref)
	_use_zero_as_xaxis(ax_pref)
	pref_stat_rows: list[dict[str, object]] = []
	for idx, condition in enumerate(condition_order):
		group = session_df[session_df["condition"].astype(str) == condition].copy()
		pref_stats = _onesample_ttest(group[cfg["pref_col"]], popmean=0.0)
		pref_stat_rows.append(pref_stats)
		stats_rows.append(
			{
				"phase": phase,
				"panel": "preference_index",
				"comparison": f"{condition} vs 0",
				"test": pref_stats["test"],
				"n": pref_stats["n"],
				"popmean": pref_stats["popmean"],
				"statistic": pref_stats["statistic"],
				"pvalue": pref_stats["pvalue"],
			}
		)
	control_condition = next((condition for condition in condition_order if condition.lower() == "control"), None)
	vpa_condition = next((condition for condition in condition_order if condition.lower() == "vpa"), None)
	if control_condition is not None and vpa_condition is not None:
		control_values = session_df[session_df["condition"].astype(str) == control_condition][cfg["pref_col"]]
		vpa_values = session_df[session_df["condition"].astype(str) == vpa_condition][cfg["pref_col"]]
		variance_stats = _brown_forsythe_test(control_values, vpa_values)
		stats_rows.append(
			{
				"phase": phase,
				"panel": "preference_index_variance",
				"comparison": f"{control_condition} vs {vpa_condition}",
				"test": variance_stats["test"],
				"n_a": variance_stats["n_a"],
				"n_b": variance_stats["n_b"],
				"variance_a": variance_stats["variance_a"],
				"variance_b": variance_stats["variance_b"],
				"variance_ratio_b_over_a": variance_stats["variance_ratio_b_over_a"],
				"statistic": variance_stats["statistic"],
				"pvalue": variance_stats["pvalue"],
			}
		)
	pref_finite = [float(v) for vals in pref_points for v in vals if np.isfinite(v)]
	for mean, sem in zip(pref_means, pref_sems):
		if np.isfinite(mean):
			pref_finite.append(float(mean))
		if np.isfinite(mean) and np.isfinite(sem):
			pref_finite.append(float(mean + sem))
			pref_finite.append(float(mean - sem))
	pref_ymax = max(pref_finite, default=0.0)
	pref_ymin = min(pref_finite, default=0.0)
	pref_ymin = min(pref_ymin, 0.0)
	yrange = max(pref_ymax - pref_ymin, 0.2)
	pi_marker_y = pref_ymax + 0.10 * yrange
	for idx, pref_stats in enumerate(pref_stat_rows):
		_draw_sig_line(
			ax_pref,
			x=float(pref_x[idx]),
			y=pi_marker_y,
			half_width=0.27,
			label=_p_to_marker(float(pref_stats["pvalue"])) if pd.notna(pref_stats["pvalue"]) else "n/a",
		)
	ax_pref.set_ylim(pref_ymin - 0.08 * yrange, pref_ymax + 0.24 * yrange)

	fig.suptitle(cfg["title"].replace(" test", " Test"), fontsize=18, y=0.98)
	out_path = output_dir / f"preference_bars__{phase}__{keypoint}__{_roi_output_tag(roi_mode, radius_scale)}{file_suffix}.png"
	_save_figure(fig, out_path, pad_inches=0.08)
	plt.close(fig)
	return out_path, stats_rows

def _plot_phase_bars_by_sex(
	session_df: pd.DataFrame,
	*,
	phase: str,
	output_dir: Path,
	keypoint: str,
	roi_mode: str,
	radius_scale: float,
	file_suffix: str,
	shared_panel_ylims: dict[str, float] | None = None,
) -> tuple[Path | None, list[dict[str, object]]]:
	cfg = PHASE_CONFIG[phase]
	if "sex" not in session_df.columns:
		return None, []
	plot_df = session_df.copy()
	plot_df["sex"] = plot_df["sex"].astype(str).str.lower().str.strip()
	plot_df = plot_df[plot_df["sex"].isin(SEX_ORDER)].copy()
	if plot_df.empty:
		return None, []

	condition_order = sorted(plot_df["condition"].dropna().astype(str).unique().tolist(), key=_condition_sort_key)
	sex_order = [sex for sex in SEX_ORDER if sex in set(plot_df["sex"])]
	stats_rows: list[dict[str, object]] = []
	fig, axes = plt.subplots(1, 3, figsize=(17.4, 5.2), constrained_layout=True)
	ax_time, ax_visit, ax_pref = axes
	group_gap = 1.15
	within_gap = 0.78

	def _ordered_condition_sex() -> list[tuple[str, str]]:
		out: list[tuple[str, str]] = []
		for condition in condition_order:
			condition_df = plot_df[plot_df["condition"].astype(str) == condition]
			for sex in sex_order:
				if not condition_df[condition_df["sex"] == sex].empty:
					out.append((condition, sex))
		return out

	condition_sex_order = _ordered_condition_sex()

	def _plot_paired_panel(
		ax: plt.Axes,
		*,
		left_col: str,
		right_col: str,
		ylabel: str,
		title: str,
		panel_name: str,
	) -> None:
		bar_positions: list[float] = []
		bar_heights: list[float] = []
		bar_errors: list[float] = []
		bar_colors: list[str] = []
		bar_labels: list[str] = []
		pair_positions: dict[tuple[str, str], tuple[float, float]] = {}
		finite_values: list[float] = []

		for idx, (condition, sex) in enumerate(condition_sex_order):
			base_x = idx * (2.0 + group_gap)
			group = plot_df[(plot_df["condition"].astype(str) == condition) & (plot_df["sex"] == sex)].copy()
			left_vals = pd.to_numeric(group[left_col], errors="coerce")
			right_vals = pd.to_numeric(group[right_col], errors="coerce")
			color = _sex_color(sex)
			left_mean = float(left_vals.mean())
			right_mean = float(right_vals.mean())
			left_sem = _sem(left_vals)
			right_sem = _sem(right_vals)

			bar_positions.extend([base_x, base_x + within_gap])
			bar_heights.extend([left_mean, right_mean])
			bar_errors.extend([left_sem, right_sem])
			bar_colors.extend([color, color])
			bar_labels.extend([cfg["left_label"], cfg["right_label"]])
			pair_positions[(condition, sex)] = (base_x, base_x + within_gap)

			for xbase, vals in ((base_x, left_vals), (base_x + within_gap, right_vals)):
				points = vals.dropna().to_numpy(dtype=float)
				if points.size:
					jitter = np.linspace(-0.07, 0.07, points.size)
					ax.scatter(np.full(points.size, xbase) + jitter, points, s=24, color=color, edgecolors="#202020", linewidths=0.8, zorder=4)
					finite_values.extend([float(v) for v in points if np.isfinite(v)])
			for top in (left_mean + left_sem, right_mean + right_sem):
				if np.isfinite(top):
					finite_values.append(float(top))
			ax.text(base_x + within_gap / 2.0, -0.17, f"{condition}\n{_sex_label(sex)}", ha="center", va="top", fontsize=10, transform=ax.get_xaxis_transform())

			paired_stats = _paired_ttest(left_vals, right_vals)
			stats_rows.append(
				{
					"phase": phase,
					"sex": sex,
					"panel": panel_name,
					"metric": f"{left_col} vs {right_col}",
					"comparison": f"{condition} {_sex_label(sex)}: {cfg['left_label']} vs {cfg['right_label']}",
					"test": paired_stats["test"],
					"n": paired_stats["n"],
					"statistic": paired_stats["statistic"],
					"pvalue": paired_stats["pvalue"],
				}
			)

		ax.bar(bar_positions, bar_heights, width=0.58, color=bar_colors, alpha=0.9, edgecolor="none")
		ax.errorbar(bar_positions, bar_heights, yerr=bar_errors, fmt="none", ecolor="black", elinewidth=1.1, capsize=3.5, capthick=1.1)
		ax.set_xticks(bar_positions)
		ax.set_xticklabels(bar_labels)
		ax.set_ylabel(ylabel)
		ax.set_title(title, fontsize=15)
		_clean_axis(ax)

		ymax = max(finite_values, default=0.0)
		yrange = max(ymax, 1.0)
		base_y = ymax + 0.08 * yrange
		step = 0.07 * yrange
		for idx, (condition, sex) in enumerate(condition_sex_order):
			stat_row = next((row for row in stats_rows if row["panel"] == panel_name and row["comparison"].startswith(f"{condition} {_sex_label(sex)}:")), None)
			if stat_row is None:
				continue
			x1, x2 = pair_positions[(condition, sex)]
			_draw_sig_bar(
				ax,
				x1=x1,
				x2=x2,
				y=base_y + idx * step,
				h=0.035 * yrange,
				label=_p_to_marker(float(stat_row["pvalue"])) if pd.notna(stat_row["pvalue"]) else "n/a",
			)
		top_limit = max(ymax + 0.24 * yrange + len(condition_sex_order) * step, 1.0)
		if shared_panel_ylims is not None and panel_name in shared_panel_ylims:
			top_limit = max(top_limit, float(shared_panel_ylims[panel_name]))
		ax.set_ylim(0.0, top_limit)

		for condition in condition_order:
			for metric_col, metric_label in ((left_col, cfg["left_label"]), (right_col, cfg["right_label"])):
				m_vals = plot_df[(plot_df["condition"].astype(str) == condition) & (plot_df["sex"] == "m")][metric_col]
				f_vals = plot_df[(plot_df["condition"].astype(str) == condition) & (plot_df["sex"] == "f")][metric_col]
				sex_stats = _welch_ttest(m_vals, f_vals)
				stats_rows.append(
					{
						"phase": phase,
						"sex": "m_vs_f",
						"panel": f"{panel_name}_sex_effect",
						"metric": metric_col,
						"comparison": f"{condition}: M vs F ({metric_label})",
						"test": sex_stats["test"],
						"n_a": sex_stats["n_a"],
						"n_b": sex_stats["n_b"],
						"statistic": sex_stats["statistic"],
						"pvalue": sex_stats["pvalue"],
					}
				)

	_plot_paired_panel(
		ax_time,
		left_col=cfg["time_left_col"],
		right_col=cfg["time_right_col"],
		ylabel="Investigation time (s)",
		title="Investigation time by sex",
		panel_name="investigation_time_s",
	)
	_plot_paired_panel(
		ax_visit,
		left_col=cfg["visit_left_col"],
		right_col=cfg["visit_right_col"],
		ylabel="Visit count",
		title="Visit count by sex",
		panel_name="visit_count",
	)

	pref_positions: list[float] = []
	pref_means: list[float] = []
	pref_sems: list[float] = []
	pref_colors: list[str] = []
	pref_labels: list[str] = []
	pref_points: dict[tuple[str, str], np.ndarray] = {}
	pref_finite: list[float] = [0.0]
	for idx, (condition, sex) in enumerate(condition_sex_order):
		group = plot_df[(plot_df["condition"].astype(str) == condition) & (plot_df["sex"] == sex)].copy()
		vals = pd.to_numeric(group[cfg["pref_col"]], errors="coerce")
		xpos = float(idx)
		pref_positions.append(xpos)
		pref_means.append(float(vals.mean()))
		pref_sems.append(_sem(vals))
		pref_colors.append(_sex_color(sex))
		pref_labels.append(f"{condition}\n{_sex_label(sex)}")
		points = vals.dropna().to_numpy(dtype=float)
		pref_points[(condition, sex)] = points
		pref_finite.extend([float(v) for v in points if np.isfinite(v)])
		stat_row = _onesample_ttest(vals, popmean=0.0)
		stats_rows.append(
			{
				"phase": phase,
				"sex": sex,
				"panel": "preference_index",
				"metric": cfg["pref_col"],
				"comparison": f"{condition} {_sex_label(sex)} vs 0",
				"test": stat_row["test"],
				"n": stat_row["n"],
				"popmean": stat_row["popmean"],
				"statistic": stat_row["statistic"],
				"pvalue": stat_row["pvalue"],
			}
		)

	ax_pref.bar(pref_positions, pref_means, width=0.62, color=pref_colors, alpha=0.9, edgecolor="none")
	ax_pref.errorbar(pref_positions, pref_means, yerr=pref_sems, fmt="none", ecolor="black", elinewidth=1.1, capsize=3.5, capthick=1.1)
	for idx, (condition, sex) in enumerate(condition_sex_order):
		points = pref_points[(condition, sex)]
		if points.size:
			jitter = np.linspace(-0.07, 0.07, points.size)
			ax_pref.scatter(np.full(points.size, pref_positions[idx]) + jitter, points, s=24, color=_sex_color(sex), edgecolors="#202020", linewidths=0.8, zorder=4)
	for mean, sem in zip(pref_means, pref_sems):
		if np.isfinite(mean):
			pref_finite.append(float(mean))
		if np.isfinite(mean) and np.isfinite(sem):
			pref_finite.extend([float(mean + sem), float(mean - sem)])
	ax_pref.set_xticks(pref_positions)
	ax_pref.set_xticklabels(pref_labels)
	ax_pref.set_ylabel(cfg["pref_ylabel"])
	ax_pref.set_title("Preference index by sex", fontsize=15)
	_clean_axis(ax_pref)
	_use_zero_as_xaxis(ax_pref)

	pref_ymin = min(pref_finite, default=0.0)
	pref_ymax = max(pref_finite, default=0.0)
	pref_ymin = min(pref_ymin, 0.0)
	yrange = max(pref_ymax - pref_ymin, 0.25)
	for condition in condition_order:
		m_vals = plot_df[(plot_df["condition"].astype(str) == condition) & (plot_df["sex"] == "m")][cfg["pref_col"]]
		f_vals = plot_df[(plot_df["condition"].astype(str) == condition) & (plot_df["sex"] == "f")][cfg["pref_col"]]
		sex_stats = _welch_ttest(m_vals, f_vals)
		stats_rows.append(
			{
				"phase": phase,
				"sex": "m_vs_f",
				"panel": "preference_index_sex_effect",
				"metric": cfg["pref_col"],
				"comparison": f"{condition}: M vs F",
				"test": sex_stats["test"],
				"n_a": sex_stats["n_a"],
				"n_b": sex_stats["n_b"],
				"statistic": sex_stats["statistic"],
				"pvalue": sex_stats["pvalue"],
			}
		)
		m_idx = next((idx for idx, item in enumerate(condition_sex_order) if item == (condition, "m")), None)
		f_idx = next((idx for idx, item in enumerate(condition_sex_order) if item == (condition, "f")), None)
		if m_idx is not None and f_idx is not None:
			y = pref_ymax + (0.10 + 0.10 * condition_order.index(condition)) * yrange
			_draw_sig_bar(
				ax_pref,
				x1=float(pref_positions[m_idx]),
				x2=float(pref_positions[f_idx]),
				y=y,
				h=0.04 * yrange,
				label=_p_to_marker(float(sex_stats["pvalue"])) if pd.notna(sex_stats["pvalue"]) else "n/a",
			)
	ax_pref.set_ylim(pref_ymin - 0.10 * yrange, pref_ymax + (0.34 + 0.10 * max(len(condition_order) - 1, 0)) * yrange)

	legend_handles = [
		Line2D([0], [0], marker="o", linestyle="", color=_sex_color("m"), markeredgecolor="black", label="Male"),
		Line2D([0], [0], marker="o", linestyle="", color=_sex_color("f"), markeredgecolor="black", label="Female"),
	]
	fig.legend(handles=legend_handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02), fontsize=10)
	fig.suptitle(f"{cfg['title']} - sex stratified", fontsize=19, y=1.08)
	out_path = output_dir / f"preference_bars_by_sex__{phase}__{keypoint}__{_roi_output_tag(roi_mode, radius_scale)}{file_suffix}.png"
	_save_figure(fig, out_path, pad_inches=0.16)
	plt.close(fig)
	return out_path, stats_rows

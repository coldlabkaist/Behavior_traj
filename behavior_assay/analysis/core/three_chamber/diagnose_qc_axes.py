from __future__ import annotations
from pathlib import Path
import sys
from analysis.paths import ROOT, resolve_input_path
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

FIGURE_DPI = 300

PREFERRED_CONDITION_ORDER = ["Control", "control", "VPA", "vpa"]

CONDITION_COLORS = {
	"Control": "#304F78",
	"control": "#304F78",
	"VPA": "#C4475B",
	"vpa": "#C4475B",
}

SEX_ORDER = ["m", "f"]

PHASE_ORDER = ["soc", "nov"]

LOCO_PHASE_ORDER = ["hab_closed", "hab_open", "soc", "nov"]

PHASE_LABELS = {
	"soc": "Sociability",
	"nov": "Social novelty",
	"hab_closed": "Hab closed",
	"hab_open": "Hab open",
}

def _to_abs_path(path_str: str) -> Path:
    return resolve_input_path(path_str)

def _condition_sort_key(condition: str) -> tuple[int, str]:
	if condition in PREFERRED_CONDITION_ORDER:
		return (PREFERRED_CONDITION_ORDER.index(condition), condition)
	return (len(PREFERRED_CONDITION_ORDER), condition)

def _sem(values: pd.Series) -> float:
	arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
	if len(arr) <= 1:
		return 0.0
	return float(np.std(arr, ddof=1) / math.sqrt(len(arr)))

def _p_label(pvalue: float) -> str:
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

def _onesample(values: pd.Series, popmean: float = 0.0) -> dict[str, object]:
	arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
	if len(arr) < 2:
		return {"test": "onesample_ttest", "n": len(arr), "statistic": math.nan, "pvalue": math.nan, "popmean": popmean}
	stat = stats.ttest_1samp(arr, popmean, nan_policy="omit")
	return {"test": "onesample_ttest", "n": len(arr), "statistic": float(stat.statistic), "pvalue": float(stat.pvalue), "popmean": popmean}

def _welch(a: pd.Series, b: pd.Series) -> dict[str, object]:
	a_arr = pd.to_numeric(a, errors="coerce").dropna().to_numpy(dtype=float)
	b_arr = pd.to_numeric(b, errors="coerce").dropna().to_numpy(dtype=float)
	if len(a_arr) < 2 or len(b_arr) < 2:
		return {"test": "welch_ttest", "n_a": len(a_arr), "n_b": len(b_arr), "statistic": math.nan, "pvalue": math.nan}
	stat = stats.ttest_ind(a_arr, b_arr, equal_var=False, nan_policy="omit")
	return {"test": "welch_ttest", "n_a": len(a_arr), "n_b": len(b_arr), "statistic": float(stat.statistic), "pvalue": float(stat.pvalue)}

def _paired(a: pd.Series, b: pd.Series) -> dict[str, object]:
	pair = pd.DataFrame({"a": pd.to_numeric(a, errors="coerce"), "b": pd.to_numeric(b, errors="coerce")}).dropna()
	if len(pair) < 2:
		return {"test": "paired_ttest", "n": len(pair), "statistic": math.nan, "pvalue": math.nan}
	stat = stats.ttest_rel(pair["a"], pair["b"], nan_policy="omit")
	return {"test": "paired_ttest", "n": len(pair), "statistic": float(stat.statistic), "pvalue": float(stat.pvalue)}

def _date_from_session_key(session_key: object) -> str:
	parts = str(session_key).split("__")
	return parts[1] if len(parts) >= 2 else ""

def _add_preference_metrics(df: pd.DataFrame) -> pd.DataFrame:
	work = df.copy()
	work["date"] = work["session_key"].map(_date_from_session_key)
	work["cage_id"] = work["subject_id"].astype(str)
	soc = work["phase"].astype(str).eq("soc")
	nov = work["phase"].astype(str).eq("nov")
	work.loc[soc, "target_time_s"] = pd.to_numeric(work.loc[soc, "social_time_s"], errors="coerce")
	work.loc[soc, "opposite_time_s"] = pd.to_numeric(work.loc[soc, "empty_time_s"], errors="coerce")
	work.loc[soc, "target_visit_count"] = pd.to_numeric(work.loc[soc, "social_visit_count"], errors="coerce")
	work.loc[soc, "opposite_visit_count"] = pd.to_numeric(work.loc[soc, "empty_visit_count"], errors="coerce")
	work.loc[soc, "preference_index"] = pd.to_numeric(work.loc[soc, "social_preference_index"], errors="coerce")
	work.loc[nov, "target_time_s"] = pd.to_numeric(work.loc[nov, "novel_time_s"], errors="coerce")
	work.loc[nov, "opposite_time_s"] = pd.to_numeric(work.loc[nov, "familiar_time_s"], errors="coerce")
	work.loc[nov, "target_visit_count"] = pd.to_numeric(work.loc[nov, "novel_visit_count"], errors="coerce")
	work.loc[nov, "opposite_visit_count"] = pd.to_numeric(work.loc[nov, "familiar_visit_count"], errors="coerce")
	work.loc[nov, "preference_index"] = pd.to_numeric(work.loc[nov, "novel_preference_index"], errors="coerce")
	work["target_minus_opposite_time_s"] = work["target_time_s"] - work["opposite_time_s"]
	work["target_minus_opposite_visits"] = work["target_visit_count"] - work["opposite_visit_count"]
	return work

def _aggregate_preference(session_df: pd.DataFrame) -> pd.DataFrame:
	group_cols = ["condition", "date", "cage_id", "phase", "subject_id", "trial_id"]
	numeric_cols = [
		"target_time_s",
		"opposite_time_s",
		"target_visit_count",
		"opposite_visit_count",
		"preference_index",
		"target_minus_opposite_time_s",
		"target_minus_opposite_visits",
	]
	agg_map: dict[str, str] = {col: "mean" for col in numeric_cols}
	for col in ["sex", "keypoint", "roi_mode", "radius_scale"]:
		if col in session_df.columns:
			agg_map[col] = "first"
	sample_df = session_df.groupby(group_cols, dropna=False, as_index=False).agg(agg_map)
	counts = session_df.groupby(group_cols, dropna=False)["session_key"].count().reset_index(name="sessions_averaged")
	sample_df = sample_df.merge(counts, on=group_cols, how="left")
	sample_df["sample_key"] = sample_df.apply(
		lambda row: f"{row['condition']}__{row['date']}__{row['subject_id']}__id{int(row['trial_id'])}" if pd.notna(row["trial_id"]) else f"{row['condition']}__{row['date']}__{row['subject_id']}",
		axis=1,
	)
	return sample_df

def _aggregate_locomotion(session_df: pd.DataFrame) -> pd.DataFrame:
	work = session_df.copy()
	work["date"] = work["session_key"].map(_date_from_session_key)
	work["cage_id"] = work["subject_id"].astype(str)
	group_cols = ["condition", "date", "cage_id", "phase", "subject_id", "trial_id"]
	numeric_cols = [
		"total_distance_norm",
		"distance_per_min_norm",
		"mean_speed_norm_per_s",
		"duration_min",
		"spatial_preference_index",
		"left_zone_time_s",
		"right_zone_time_s",
	]
	agg_map: dict[str, str] = {col: "mean" for col in numeric_cols if col in work.columns}
	for col in ["sex", "keypoint"]:
		if col in work.columns:
			agg_map[col] = "first"
	sample_df = work.groupby(group_cols, dropna=False, as_index=False).agg(agg_map)
	counts = work.groupby(group_cols, dropna=False)["session_key"].count().reset_index(name="sessions_averaged")
	sample_df = sample_df.merge(counts, on=group_cols, how="left")
	sample_df["sample_key"] = sample_df.apply(
		lambda row: f"{row['condition']}__{row['date']}__{row['subject_id']}__id{int(row['trial_id'])}" if pd.notna(row["trial_id"]) else f"{row['condition']}__{row['date']}__{row['subject_id']}",
		axis=1,
	)
	return sample_df

def _summary_by(df: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
	rows = []
	for key, group in df.groupby(group_cols, dropna=False):
		key_values = key if isinstance(key, tuple) else (key,)
		row = dict(zip(group_cols, key_values))
		for metric in metric_cols:
			vals = pd.to_numeric(group[metric], errors="coerce").dropna()
			row[f"{metric}_n"] = int(len(vals))
			row[f"{metric}_mean"] = float(vals.mean()) if len(vals) else math.nan
			row[f"{metric}_sem"] = _sem(vals)
		rows.append(row)
	return pd.DataFrame(rows)

def _preference_stats(pref_df: pd.DataFrame) -> pd.DataFrame:
	rows = []
	for group_cols, axis_name in [
		(["phase", "condition"], "condition"),
		(["phase", "condition", "sex"], "condition_sex"),
		(["phase", "condition", "date"], "condition_date"),
		(["phase", "condition", "cage_id"], "condition_cage"),
	]:
		for key, group in pref_df.groupby(group_cols, dropna=False):
			key_values = key if isinstance(key, tuple) else (key,)
			base = dict(zip(group_cols, key_values))
			for metric in ["preference_index", "target_minus_opposite_time_s"]:
				test = _onesample(group[metric], 0.0)
				rows.append({**base, "axis": axis_name, "metric": metric, "comparison": "vs 0", **test, "p_label": _p_label(float(test["pvalue"])) if pd.notna(test["pvalue"]) else "n/a"})
	for phase, group in pref_df.groupby("phase", dropna=False):
		conditions = sorted(group["condition"].dropna().astype(str).unique(), key=_condition_sort_key)
		if len(conditions) >= 2:
			a, b = conditions[0], conditions[1]
			for metric in ["preference_index", "target_minus_opposite_time_s"]:
				test = _welch(group[group["condition"].astype(str).eq(a)][metric], group[group["condition"].astype(str).eq(b)][metric])
				rows.append({"axis": "condition_comparison", "phase": phase, "metric": metric, "comparison": f"{a} vs {b}", **test, "p_label": _p_label(float(test["pvalue"])) if pd.notna(test["pvalue"]) else "n/a"})
		for sex, sex_group in group.groupby("sex", dropna=False):
			conditions = sorted(sex_group["condition"].dropna().astype(str).unique(), key=_condition_sort_key)
			if len(conditions) >= 2:
				a, b = conditions[0], conditions[1]
				for metric in ["preference_index", "target_minus_opposite_time_s"]:
					test = _welch(sex_group[sex_group["condition"].astype(str).eq(a)][metric], sex_group[sex_group["condition"].astype(str).eq(b)][metric])
					rows.append({"axis": "condition_comparison_within_sex", "phase": phase, "sex": sex, "metric": metric, "comparison": f"{a} vs {b}", **test, "p_label": _p_label(float(test["pvalue"])) if pd.notna(test["pvalue"]) else "n/a"})
	return pd.DataFrame(rows)

def _locomotion_stats(loco_df: pd.DataFrame) -> pd.DataFrame:
	rows = []
	for phase, group in loco_df.groupby("phase", dropna=False):
		conditions = sorted(group["condition"].dropna().astype(str).unique(), key=_condition_sort_key)
		if len(conditions) >= 2:
			a, b = conditions[0], conditions[1]
			for metric in ["total_distance_norm", "distance_per_min_norm", "mean_speed_norm_per_s"]:
				test = _welch(group[group["condition"].astype(str).eq(a)][metric], group[group["condition"].astype(str).eq(b)][metric])
				rows.append({"axis": "condition_comparison", "phase": phase, "metric": metric, "comparison": f"{a} vs {b}", **test, "p_label": _p_label(float(test["pvalue"])) if pd.notna(test["pvalue"]) else "n/a"})
		for sex, sex_group in group.groupby("sex", dropna=False):
			conditions = sorted(sex_group["condition"].dropna().astype(str).unique(), key=_condition_sort_key)
			if len(conditions) >= 2:
				a, b = conditions[0], conditions[1]
				for metric in ["total_distance_norm", "distance_per_min_norm", "mean_speed_norm_per_s"]:
					test = _welch(sex_group[sex_group["condition"].astype(str).eq(a)][metric], sex_group[sex_group["condition"].astype(str).eq(b)][metric])
					rows.append({"axis": "condition_comparison_within_sex", "phase": phase, "sex": sex, "metric": metric, "comparison": f"{a} vs {b}", **test, "p_label": _p_label(float(test["pvalue"])) if pd.notna(test["pvalue"]) else "n/a"})
	open_df = loco_df[loco_df["phase"].eq("hab_open")].copy()
	for group_cols, axis_name in [(["condition"], "open_bias_condition"), (["condition", "sex"], "open_bias_condition_sex"), (["condition", "date"], "open_bias_condition_date")]:
		for key, group in open_df.groupby(group_cols, dropna=False):
			key_values = key if isinstance(key, tuple) else (key,)
			base = dict(zip(group_cols, key_values))
			test = _onesample(group["spatial_preference_index"], 0.0)
			rows.append({**base, "axis": axis_name, "phase": "hab_open", "metric": "spatial_preference_index", "comparison": "right bias vs 0", **test, "p_label": _p_label(float(test["pvalue"])) if pd.notna(test["pvalue"]) else "n/a"})
	return pd.DataFrame(rows)

def _write_report(
	path: Path,
	pref_df: pd.DataFrame,
	pref_stats: pd.DataFrame,
	loco_stats: pd.DataFrame,
	condition_date: pd.DataFrame,
) -> None:
	lines = [
		"# 3-chamber QC diagnosis",
		"",
		"## Design balance",
		"",
	]
	for _, row in condition_date.iterrows():
		lines.append(f"- {row['condition']} / {row['date']}: {int(row['n_samples'])} samples")
	lines.extend(["", "Note: date/batch is not fully crossed with condition, so date effects cannot be cleanly separated from condition effects.", ""])
	lines.append("## Preference group comparison")
	group_rows = pref_stats[pref_stats["axis"].eq("condition_comparison") & pref_stats["metric"].eq("preference_index")]
	for _, row in group_rows.iterrows():
		lines.append(f"- {row['phase']} {row['comparison']}: p={float(row['pvalue']):.4g} ({row['p_label']})")
	lines.extend(["", "## Sex-specific group comparison"])
	sex_rows = pref_stats[pref_stats["axis"].eq("condition_comparison_within_sex") & pref_stats["metric"].eq("preference_index")]
	for _, row in sex_rows.iterrows():
		lines.append(f"- {row['phase']} sex={row.get('sex', '')} {row['comparison']}: p={float(row['pvalue']):.4g} ({row['p_label']})")
	lines.extend(["", "## Locomotion group comparison"])
	loco_rows = loco_stats[loco_stats["axis"].eq("condition_comparison") & loco_stats["metric"].eq("distance_per_min_norm")]
	for _, row in loco_rows.iterrows():
		lines.append(f"- {row['phase']} {row['comparison']}: p={float(row['pvalue']):.4g} ({row['p_label']})")
	lines.extend(["", "## Open habituation bias"])
	open_rows = loco_stats[loco_stats["axis"].eq("open_bias_condition") & loco_stats["metric"].eq("spatial_preference_index")]
	for _, row in open_rows.iterrows():
		lines.append(f"- {row['condition']} right-bias vs 0: p={float(row['pvalue']):.4g} ({row['p_label']})")
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text("\n".join(lines), encoding="utf-8")


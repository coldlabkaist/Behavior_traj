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

CONDITION_ORDER = ["Control", "VPA"]

CONDITION_COLORS = {"Control": "#304F78", "VPA": "#C4475B"}

SEX_MARKERS = {"f": "o", "m": "s"}

def _to_abs_path(value: str) -> Path:
    return resolve_input_path(value)

def _clean(values: pd.Series) -> np.ndarray:
	return pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)

def _sem(values: pd.Series | np.ndarray) -> float:
	array = _clean(values) if isinstance(values, pd.Series) else np.asarray(values, dtype=float)
	array = array[np.isfinite(array)]
	return float(stats.sem(array)) if len(array) >= 2 else math.nan

def _p_label(pvalue: float) -> str:
	if not np.isfinite(pvalue):
		return "n/a"
	if pvalue < 0.0001:
		return "****"
	if pvalue < 0.001:
		return "***"
	if pvalue < 0.01:
		return "**"
	if pvalue < 0.05:
		return "*"
	return "n.s."

def _onesample(values: pd.Series, popmean: float = 0.0) -> dict[str, object]:
	array = _clean(values)
	if len(array) < 2:
		return {"test": "onesample_ttest", "n": len(array), "statistic": math.nan, "pvalue": math.nan}
	result = stats.ttest_1samp(array, popmean=popmean, nan_policy="omit")
	return {"test": "onesample_ttest", "n": len(array), "statistic": float(result.statistic), "pvalue": float(result.pvalue)}

def _welch(a: pd.Series, b: pd.Series) -> dict[str, object]:
	a_array = _clean(a)
	b_array = _clean(b)
	if len(a_array) < 2 or len(b_array) < 2:
		return {
			"test": "welch_ttest",
			"n_a": len(a_array),
			"n_b": len(b_array),
			"statistic": math.nan,
			"pvalue": math.nan,
			"hedges_g_b_minus_a": math.nan,
		}
	result = stats.ttest_ind(a_array, b_array, equal_var=False, nan_policy="omit")
	pooled_num = (len(a_array) - 1) * np.var(a_array, ddof=1) + (len(b_array) - 1) * np.var(b_array, ddof=1)
	pooled_den = len(a_array) + len(b_array) - 2
	pooled_sd = math.sqrt(pooled_num / pooled_den) if pooled_den > 0 and pooled_num > 0 else math.nan
	correction = 1.0 - 3.0 / (4.0 * (len(a_array) + len(b_array)) - 9.0)
	hedges_g = correction * (float(np.mean(b_array)) - float(np.mean(a_array))) / pooled_sd if np.isfinite(pooled_sd) else math.nan
	return {
		"test": "welch_ttest",
		"n_a": len(a_array),
		"n_b": len(b_array),
		"statistic": float(result.statistic),
		"pvalue": float(result.pvalue),
		"hedges_g_b_minus_a": hedges_g,
	}

def _interaction_contrast(sample_df: pd.DataFrame) -> dict[str, object]:
	groups = {
		(condition, sex): _clean(sample_df[(sample_df["condition"].eq(condition)) & (sample_df["sex"].eq(sex))]["preference_index"])
		for condition in CONDITION_ORDER
		for sex in ["f", "m"]
	}
	if any(len(values) < 2 for values in groups.values()):
		return {"test": "sex_by_condition_interaction_contrast", "n": len(sample_df), "statistic": math.nan, "pvalue": math.nan, "estimate": math.nan}
	weights = {("Control", "f"): -1.0, ("Control", "m"): 1.0, ("VPA", "f"): 1.0, ("VPA", "m"): -1.0}
	estimate = sum(weights[key] * float(np.mean(values)) for key, values in groups.items())
	variance_terms = {key: float(np.var(values, ddof=1)) / len(values) for key, values in groups.items()}
	variance = sum(variance_terms.values())
	if variance <= 0:
		return {"test": "sex_by_condition_interaction_contrast", "n": len(sample_df), "statistic": math.nan, "pvalue": math.nan, "estimate": estimate}
	statistic = estimate / math.sqrt(variance)
	df_num = variance**2
	df_den = sum((term**2) / (len(groups[key]) - 1) for key, term in variance_terms.items())
	df = df_num / df_den if df_den > 0 else math.nan
	pvalue = 2.0 * stats.t.sf(abs(statistic), df) if np.isfinite(df) else math.nan
	return {
		"test": "sex_by_condition_interaction_contrast",
		"n": len(sample_df),
		"statistic": statistic,
		"pvalue": pvalue,
		"estimate": estimate,
		"df": df,
	}

def _correlation(a: pd.Series, b: pd.Series, method: str = "spearman") -> dict[str, object]:
	pair = pd.DataFrame({"a": pd.to_numeric(a, errors="coerce"), "b": pd.to_numeric(b, errors="coerce")}).dropna()
	if len(pair) < 3 or pair["a"].nunique() < 2 or pair["b"].nunique() < 2:
		return {"test": f"{method}_correlation", "n": len(pair), "statistic": math.nan, "pvalue": math.nan}
	result = stats.spearmanr(pair["a"], pair["b"]) if method == "spearman" else stats.pearsonr(pair["a"], pair["b"])
	return {"test": f"{method}_correlation", "n": len(pair), "statistic": float(result.statistic), "pvalue": float(result.pvalue)}

def _trial_key(series: pd.Series) -> pd.Series:
	return pd.to_numeric(series, errors="coerce").astype("Int64")

def _attach_metadata(sample_df: pd.DataFrame, manifest_df: pd.DataFrame) -> pd.DataFrame:
	work = sample_df[sample_df["phase"].astype(str).eq("nov")].copy()
	work["trial_id"] = _trial_key(work["trial_id"])
	manifest = manifest_df[manifest_df["phase"].astype(str).eq("nov")].copy()
	manifest["trial_id"] = _trial_key(manifest["trial_id"])
	manifest["session_n"] = pd.to_numeric(manifest["session_n"], errors="coerce")
	if "session_n_values" in work.columns:
		session_values = set(work["session_n_values"].dropna().astype(str))
		if session_values == {"1"}:
			manifest = manifest[manifest["session_n"].eq(1)]
	keys = ["condition", "phase", "subject_id", "trial_id"]
	date_counts = manifest.groupby(keys, dropna=False)["date"].nunique()
	ambiguous = date_counts[date_counts > 1]
	if not ambiguous.empty:
		raise ValueError(f"Ambiguous manifest date matches for {len(ambiguous)} novelty samples")
	metadata = manifest[keys + ["date"]].drop_duplicates(keys)
	work = work.merge(metadata, on=keys, how="left", validate="one_to_one")
	if work["date"].isna().any():
		missing = work.loc[work["date"].isna(), keys]
		raise ValueError(f"Missing manifest date for {len(missing)} novelty samples")
	work["date"] = work["date"].astype(str)
	work["sex"] = work["sex"].astype(str).str.lower()
	work["cage_id"] = work["subject_id"].astype(str)
	work["preference_index"] = pd.to_numeric(work["novel_preference_index"], errors="coerce")
	work["target_minus_opposite_time_s"] = pd.to_numeric(work["novel_time_s"], errors="coerce") - pd.to_numeric(work["familiar_time_s"], errors="coerce")
	work["target_minus_opposite_visits"] = pd.to_numeric(work["novel_visit_count"], errors="coerce") - pd.to_numeric(work["familiar_visit_count"], errors="coerce")
	return work

def _attach_behavior_covariates(sample_df: pd.DataFrame, locomotion_path: Path) -> pd.DataFrame:
	if not locomotion_path.exists():
		return sample_df
	loco = pd.read_csv(locomotion_path)
	loco["trial_id"] = _trial_key(loco["trial_id"])
	keys = ["condition", "subject_id", "trial_id"]
	nov_cols = keys + ["distance_per_min_norm", "total_distance_norm"]
	nov = loco[loco["phase"].astype(str).eq("nov")][nov_cols].drop_duplicates(keys)
	nov = nov.rename(columns={"distance_per_min_norm": "nov_distance_per_min_norm", "total_distance_norm": "nov_total_distance_norm"})
	open_cols = keys + ["spatial_preference_index"]
	open_hab = loco[loco["phase"].astype(str).eq("hab_open")][open_cols].drop_duplicates(keys)
	open_hab = open_hab.rename(columns={"spatial_preference_index": "open_spatial_preference_index"})
	return sample_df.merge(nov, on=keys, how="left", validate="one_to_one").merge(open_hab, on=keys, how="left", validate="one_to_one")

def _group_summary(sample_df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
	rows: list[dict[str, object]] = []
	for key, group in sample_df.groupby(columns, dropna=False, sort=True):
		key_values = key if isinstance(key, tuple) else (key,)
		row = dict(zip(columns, key_values))
		for metric in ["preference_index", "target_minus_opposite_time_s", "target_minus_opposite_visits"]:
			values = _clean(group[metric])
			row[f"{metric}_n"] = len(values)
			row[f"{metric}_mean"] = float(np.mean(values)) if len(values) else math.nan
			row[f"{metric}_sd"] = float(np.std(values, ddof=1)) if len(values) >= 2 else math.nan
			row[f"{metric}_sem"] = _sem(values)
			test = _onesample(pd.Series(values))
			row[f"{metric}_vs_zero_p"] = test["pvalue"]
			row[f"{metric}_vs_zero_label"] = _p_label(float(test["pvalue"]))
		rows.append(row)
	return pd.DataFrame(rows)

def _cage_summary(sample_df: pd.DataFrame) -> pd.DataFrame:
	metric_cols = ["preference_index", "target_minus_opposite_time_s", "target_minus_opposite_visits"]
	return (
		sample_df.groupby(["condition", "date", "cage_id", "sex"], dropna=False, as_index=False)
		.agg(**{metric: (metric, "mean") for metric in metric_cols}, n_animals=("sample_key", "nunique"))
		.sort_values(["condition", "date", "cage_id"])
	)

def _overall_condition_test(sample_df: pd.DataFrame, metric: str) -> dict[str, object]:
	control = sample_df[sample_df["condition"].eq("Control")][metric]
	vpa = sample_df[sample_df["condition"].eq("VPA")][metric]
	result = _welch(control, vpa)
	return {
		"metric": metric,
		"comparison": "Control vs VPA",
		"mean_control": float(pd.to_numeric(control, errors="coerce").mean()),
		"mean_vpa": float(pd.to_numeric(vpa, errors="coerce").mean()),
		"difference_vpa_minus_control": float(pd.to_numeric(vpa, errors="coerce").mean() - pd.to_numeric(control, errors="coerce").mean()),
		**result,
		"p_label": _p_label(float(result["pvalue"])),
	}

def _leave_one_batch_out(sample_df: pd.DataFrame) -> pd.DataFrame:
	overall = _overall_condition_test(sample_df, "preference_index")
	rows: list[dict[str, object]] = [
		{
			"omitted_date": "ALL",
			"omitted_condition": "None",
			"omitted_n": 0,
			**overall,
			"shift_from_full_difference": 0.0,
		}
	]
	for date in sorted(sample_df["date"].unique()):
		omitted = sample_df[sample_df["date"].eq(date)]
		reduced = sample_df[~sample_df["date"].eq(date)]
		result = _overall_condition_test(reduced, "preference_index")
		rows.append(
			{
				"omitted_date": date,
				"omitted_condition": ",".join(sorted(omitted["condition"].unique())),
				"omitted_n": len(omitted),
				**result,
				"shift_from_full_difference": result["difference_vpa_minus_control"] - overall["difference_vpa_minus_control"],
			}
		)
	return pd.DataFrame(rows)

def _build_stats(sample_df: pd.DataFrame, cage_df: pd.DataFrame) -> pd.DataFrame:
	rows: list[dict[str, object]] = []
	for metric in ["preference_index", "target_minus_opposite_time_s", "target_minus_opposite_visits"]:
		rows.append({"axis": "condition", **_overall_condition_test(sample_df, metric)})
	for condition in CONDITION_ORDER:
		condition_df = sample_df[sample_df["condition"].eq(condition)]
		result = _onesample(condition_df["preference_index"])
		rows.append({"axis": "within_condition", "metric": "preference_index", "comparison": f"{condition} vs 0", **result, "p_label": _p_label(float(result["pvalue"]))})
		result = _welch(condition_df[condition_df["sex"].eq("f")]["preference_index"], condition_df[condition_df["sex"].eq("m")]["preference_index"])
		rows.append({"axis": "sex_within_condition", "metric": "preference_index", "comparison": f"{condition}: female vs male", **result, "p_label": _p_label(float(result["pvalue"]))})
	for sex in ["f", "m"]:
		sex_df = sample_df[sample_df["sex"].eq(sex)]
		result = _welch(sex_df[sex_df["condition"].eq("Control")]["preference_index"], sex_df[sex_df["condition"].eq("VPA")]["preference_index"])
		rows.append({"axis": "condition_within_sex", "metric": "preference_index", "comparison": f"Control vs VPA, sex={sex}", **result, "p_label": _p_label(float(result["pvalue"]))})
	interaction = _interaction_contrast(sample_df)
	rows.append({"axis": "sex_condition_interaction", "metric": "preference_index", "comparison": "(VPA F-M) - (Control F-M)", **interaction, "p_label": _p_label(float(interaction["pvalue"]))})
	result = _welch(cage_df[cage_df["condition"].eq("Control")]["preference_index"], cage_df[cage_df["condition"].eq("VPA")]["preference_index"])
	rows.append({"axis": "cage_mean_condition", "metric": "preference_index", "comparison": "Control vs VPA", **result, "p_label": _p_label(float(result["pvalue"]))})
	if "nov_distance_per_min_norm" in sample_df.columns:
		result = _welch(sample_df[sample_df["condition"].eq("Control")]["nov_distance_per_min_norm"], sample_df[sample_df["condition"].eq("VPA")]["nov_distance_per_min_norm"])
		rows.append({"axis": "locomotion_condition", "metric": "nov_distance_per_min_norm", "comparison": "Control vs VPA", **result, "p_label": _p_label(float(result["pvalue"]))})
		for condition in ["All"] + CONDITION_ORDER:
			group = sample_df if condition == "All" else sample_df[sample_df["condition"].eq(condition)]
			result = _correlation(group["preference_index"], group["nov_distance_per_min_norm"])
			rows.append({"axis": "locomotion_correlation", "metric": "preference_index_vs_nov_distance_per_min_norm", "comparison": condition, **result, "p_label": _p_label(float(result["pvalue"]))})
	if "open_spatial_preference_index" in sample_df.columns:
		for condition in ["All"] + CONDITION_ORDER:
			group = sample_df if condition == "All" else sample_df[sample_df["condition"].eq(condition)]
			result = _correlation(group["preference_index"], group["open_spatial_preference_index"])
			rows.append({"axis": "open_bias_correlation", "metric": "preference_index_vs_open_spatial_preference_index", "comparison": condition, **result, "p_label": _p_label(float(result["pvalue"]))})
	return pd.DataFrame(rows)

def _write_report(
	path: Path,
	sample_df: pd.DataFrame,
	batch_df: pd.DataFrame,
	sex_df: pd.DataFrame,
	leave_one_df: pd.DataFrame,
	stats_df: pd.DataFrame,
) -> None:
	overall = stats_df[(stats_df["axis"].eq("condition")) & (stats_df["metric"].eq("preference_index"))].iloc[0]
	interaction = stats_df[stats_df["axis"].eq("sex_condition_interaction")].iloc[0]
	dates_by_condition = sample_df.groupby("condition")["date"].agg(lambda values: sorted(set(values)))
	overlap = set(dates_by_condition.get("Control", [])) & set(dates_by_condition.get("VPA", []))
	max_shift_row = leave_one_df[~leave_one_df["omitted_date"].eq("ALL")].iloc[leave_one_df[~leave_one_df["omitted_date"].eq("ALL")]["shift_from_full_difference"].abs().argmax()]
	lines = [
		"# Social novelty cause-tracing QC",
		"",
		"Exploratory QC only. P-values in this report are unadjusted and should not be used for endpoint shopping.",
		"",
		"## Overall result",
		"",
		f"- Control mean PI: {float(overall['mean_control']):.3f}",
		f"- VPA mean PI: {float(overall['mean_vpa']):.3f}",
		f"- VPA - Control PI: {float(overall['difference_vpa_minus_control']):.3f}",
		f"- Welch p: {float(overall['pvalue']):.4g}; Hedges g: {float(overall['hedges_g_b_minus_a']):.3f}",
		"",
		"## Design confounding",
		"",
		f"- Dates shared by both conditions: {', '.join(sorted(overlap)) if overlap else 'none'}",
		"- With no shared dates, condition and date/batch effects cannot be separated statistically.",
		"",
		"## Batch/date means",
		"",
	]
	for row in batch_df.itertuples(index=False):
		lines.append(f"- {row.condition} / {row.date}: n={int(row.preference_index_n)}, PI={row.preference_index_mean:.3f}")
	lines.extend(
		[
			"",
			"## Sex means",
			"",
		]
	)
	for row in sex_df.itertuples(index=False):
		lines.append(f"- {row.condition} / sex={row.sex}: n={int(row.preference_index_n)}, PI={row.preference_index_mean:.3f}")
	lines.extend(
		[
			"",
			f"- Sex-by-condition interaction contrast: estimate={float(interaction['estimate']):.3f}, p={float(interaction['pvalue']):.4g}",
			"",
			"## Influence",
			"",
			f"- Largest leave-one-batch shift: omit {max_shift_row['omitted_date']} ({max_shift_row['omitted_condition']}), shift={float(max_shift_row['shift_from_full_difference']):+.3f}",
			"",
			"## Interpretation guardrails",
			"",
			"- A significant within-group PI tests novelty preference within that group; it does not establish a Control-VPA deficit.",
			"- ROI settings should be fixed from apparatus geometry before confirmatory analysis.",
			"- Batch and sex findings here are cause-tracing signals, not corrected confirmatory tests.",
		]
	)
	path.write_text("\n".join(lines), encoding="utf-8")


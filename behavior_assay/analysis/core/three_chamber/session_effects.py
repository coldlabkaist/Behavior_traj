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

PHASE_TITLES = {
	"soc": "Sociability test",
	"nov": "Social Novelty test",
}

PREFERRED_CONDITION_ORDER = ["Control", "control", "VPA", "vpa"]

CONDITION_COLORS = {
	"Control": "#304F78",
	"control": "#304F78",
	"VPA": "#C4475B",
	"vpa": "#C4475B",
}

CONDITION_OFFSETS = {
	"Control": -0.07,
	"control": -0.07,
	"VPA": 0.07,
	"vpa": 0.07,
}

METRICS = [
	("preference_index", "Preference index"),
	("target_time_s", "Target time (s)"),
	("opposite_time_s", "Opposite time (s)"),
	("delta_time_s", "Target - opposite time (s)"),
	("target_visit_count", "Target visits"),
	("opposite_visit_count", "Opposite visits"),
]

def _to_abs_path(path_str: str) -> Path:
    return resolve_input_path(path_str)

def _parse_phase_list(arg: str) -> list[str]:
	phases = [token.strip().lower() for token in str(arg).split(",") if token.strip()]
	if not phases:
		raise ValueError("At least one phase must be provided")
	return phases

def _condition_sort_key(condition: str) -> tuple[int, str]:
	if condition in PREFERRED_CONDITION_ORDER:
		return (PREFERRED_CONDITION_ORDER.index(condition), condition)
	return (len(PREFERRED_CONDITION_ORDER), condition)

def _condition_color(condition: str) -> str:
	return CONDITION_COLORS.get(str(condition), "#777777")

def _condition_offset(condition: str) -> float:
	return CONDITION_OFFSETS.get(str(condition), 0.0)

def _sem(values: pd.Series) -> float:
	arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
	if arr.size <= 1:
		return 0.0
	return float(np.std(arr, ddof=1) / math.sqrt(arr.size))

def _paired_ttest(n1: pd.Series, n2: pd.Series) -> dict[str, object]:
	pair_df = pd.DataFrame({"n1": pd.to_numeric(n1, errors="coerce"), "n2": pd.to_numeric(n2, errors="coerce")}).dropna()
	if len(pair_df) < 2:
		return {"test": "paired_ttest_n1_vs_n2", "n": len(pair_df), "statistic": math.nan, "pvalue": math.nan}
	stat = stats.ttest_rel(pair_df["n1"], pair_df["n2"], nan_policy="omit")
	return {"test": "paired_ttest_n1_vs_n2", "n": len(pair_df), "statistic": float(stat.statistic), "pvalue": float(stat.pvalue)}

def _default_input_path(keypoint: str, roi_mode: str, radius_scale: float) -> Path:
	return ROOT / "output" / "3chamber" / "barplots" / f"preference_session_summary__{keypoint}__{roi_mode}_r{radius_scale:.2f}.csv"

def _derive_session_date(session_key: object) -> str:
	text = str(session_key)
	parts = text.split("__")
	if len(parts) >= 2:
		return parts[1]
	return ""

def _prepare_session_table(df: pd.DataFrame) -> pd.DataFrame:
	work = df.copy()
	work["session_n"] = pd.to_numeric(work["session_n"], errors="coerce")
	work = work[work["session_n"].isin([1, 2])].copy()
	work["session_n"] = work["session_n"].astype(int)
	if "session_date" not in work.columns:
		work["session_date"] = work["session_key"].map(_derive_session_date) if "session_key" in work.columns else ""

	soc = work["phase"].eq("soc")
	nov = work["phase"].eq("nov")
	work.loc[soc, "preference_index"] = pd.to_numeric(work.loc[soc, "social_preference_index"], errors="coerce")
	work.loc[soc, "target_time_s"] = pd.to_numeric(work.loc[soc, "social_time_s"], errors="coerce")
	work.loc[soc, "opposite_time_s"] = pd.to_numeric(work.loc[soc, "empty_time_s"], errors="coerce")
	work.loc[soc, "target_visit_count"] = pd.to_numeric(work.loc[soc, "social_visit_count"], errors="coerce")
	work.loc[soc, "opposite_visit_count"] = pd.to_numeric(work.loc[soc, "empty_visit_count"], errors="coerce")
	work.loc[nov, "preference_index"] = pd.to_numeric(work.loc[nov, "novel_preference_index"], errors="coerce")
	work.loc[nov, "target_time_s"] = pd.to_numeric(work.loc[nov, "novel_time_s"], errors="coerce")
	work.loc[nov, "opposite_time_s"] = pd.to_numeric(work.loc[nov, "familiar_time_s"], errors="coerce")
	work.loc[nov, "target_visit_count"] = pd.to_numeric(work.loc[nov, "novel_visit_count"], errors="coerce")
	work.loc[nov, "opposite_visit_count"] = pd.to_numeric(work.loc[nov, "familiar_visit_count"], errors="coerce")
	work["delta_time_s"] = work["target_time_s"] - work["opposite_time_s"]
	work["delta_visit_count"] = work["target_visit_count"] - work["opposite_visit_count"]
	return work

def _pair_sessions(session_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
	pair_cols = ["condition", "phase", "session_date", "subject_id", "trial_id"]
	optional_cols = [col for col in ["sex", "keypoint", "roi_mode", "radius_scale"] if col in session_df.columns]
	metric_cols = [name for name, _label in METRICS] + ["delta_visit_count"]
	rows = []
	issues = []
	for key, group in session_df.groupby(pair_cols, dropna=False):
		key_map = dict(zip(pair_cols, key))
		n1 = group[group["session_n"].eq(1)]
		n2 = group[group["session_n"].eq(2)]
		if n1.empty or n2.empty:
			issues.append({**key_map, "issue": "missing_n1_or_n2", "n1_rows": len(n1), "n2_rows": len(n2)})
			continue
		if len(n1) > 1 or len(n2) > 1:
			issues.append({**key_map, "issue": "duplicate_session_rows_averaged", "n1_rows": len(n1), "n2_rows": len(n2)})
		row = key_map.copy()
		for col in optional_cols:
			values = group[col].dropna().unique()
			row[col] = values[0] if len(values) else ""
		for metric in metric_cols:
			row[f"{metric}_n1"] = pd.to_numeric(n1[metric], errors="coerce").mean()
			row[f"{metric}_n2"] = pd.to_numeric(n2[metric], errors="coerce").mean()
			row[f"{metric}_delta_n2_minus_n1"] = row[f"{metric}_n2"] - row[f"{metric}_n1"]
		rows.append(row)
	return pd.DataFrame(rows), pd.DataFrame(issues)


from __future__ import annotations
from pathlib import Path
import sys
from analysis.paths import ROOT
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

FIGURE_DPI = 1200

CONDITION_ORDER = ("Control", "VPA")

CONDITION_COLORS = {
	"Control": "#304F78",
	"VPA": "#C4475B",
}

PHASE_CONFIG = {
	"soc": {
		"title": "Sociability Test",
		"left_label": "E",
		"right_label": "S",
		"left_time": "empty_time_s",
		"right_time": "social_time_s",
		"preference": "social_preference_index",
	},
	"nov": {
		"title": "Social Novelty Test",
		"left_label": "F",
		"right_label": "N",
		"left_time": "familiar_time_s",
		"right_time": "novel_time_s",
		"preference": "novel_preference_index",
	},
}

def _path(value: str) -> Path:
	path = Path(value)
	return path if path.is_absolute() else ROOT / path

def _sem(values: pd.Series) -> float:
	array = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
	if array.size <= 1:
		return 0.0
	return float(np.std(array, ddof=1) / math.sqrt(array.size))

def _p_marker(pvalue: float) -> str:
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

def _paired_p(left: pd.Series, right: pd.Series) -> float:
	pairs = pd.concat(
		[pd.to_numeric(left, errors="coerce"), pd.to_numeric(right, errors="coerce")],
		axis=1,
	).dropna()
	if len(pairs) < 2:
		return math.nan
	return float(stats.ttest_rel(pairs.iloc[:, 0], pairs.iloc[:, 1]).pvalue)

def _one_sample_p(values: pd.Series) -> float:
	array = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
	if array.size < 2:
		return math.nan
	return float(stats.ttest_1samp(array, popmean=0.0).pvalue)

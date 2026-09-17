from __future__ import annotations
from pathlib import Path
import sys
from analysis.paths import ROOT
import itertools
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import ConvergenceWarning

try:
	import pingouin as pg
except ImportError:
	pg = None

PNDS = [10, 15, 20]

COLORS = {"Control": "#555555", "VPA": "#F28E1C"}

METRICS = {
	"avg_mom_dist_to_cluster_mm": "Mean mother distance to initial pup cluster (mm)",
	"time_mom_close_to_cluster_sec": "Time mother close to initial cluster (s)",
	"total_stable_proximity_time_sec": "Total stable proximity time (s)",
	"avg_stable_proximity_bout_duration_sec": "Mean stable proximity bout duration (s)",
	"avg_mom_dist_to_stray_pup_mm": "Mean mother distance to stray pup (mm)",
	"total_dispersion_duration_sec": "Total dispersion duration (s)",
	"total_path_length_mm": "Mother total path length (mm)",
	"avg_velocity_mm_s": "Mother mean velocity (mm/s)",
	"pct_time_stationary": "Mother stationary time (%)",
	"pct_time_active": "Mother active time (%)",
	"avg_speed_while_moving_mm_s": "Mother speed while moving (mm/s)",
	"avg_pups_per_cluster": "Mean pups per cluster",
	"avg_mom_first_approach_latency_sec": "Mother first-approach latency (s)",
	"mom_approach_success_rate_pct": "Mother approach success within window (%)",
	"n_approach_bouts": "Number of mother-to-cluster approach bouts",
	"avg_bout_duration_sec": "Mean approach bout duration (s)",
	"approach_bouts_per_active_min": "Mother-to-cluster bouts per active minute",
	"pct_active_steps_toward_cluster_when_far": "Active steps directed toward cluster while far (%)",
}

def between_group_tests(df: pd.DataFrame, metric: str) -> pd.DataFrame:
	rows: list[dict[str, object]] = []
	for pnd in PNDS:
		subset = df[df["pnd"] == pnd]
		control = pd.to_numeric(
			subset.loc[subset["condition"] == "Control", metric],
			errors="coerce",
		).dropna()
		vpa = pd.to_numeric(
			subset.loc[subset["condition"] == "VPA", metric],
			errors="coerce",
		).dropna()
		if len(control) >= 2 and len(vpa) >= 2:
			t_stat, p_value = stats.ttest_ind(vpa, control, equal_var=False)
		else:
			t_stat = p_value = np.nan
		rows.append(
			{
				"metric": metric,
				"pnd": pnd,
				"n_control": len(control),
				"n_vpa": len(vpa),
				"mean_control": control.mean() if len(control) else np.nan,
				"mean_vpa": vpa.mean() if len(vpa) else np.nan,
				"welch_t": t_stat,
				"p_value": p_value,
			}
		)
	output = pd.DataFrame(rows)
	valid = output["p_value"].notna()
	output["p_holm_within_metric"] = np.nan
	if valid.any():
		output.loc[valid, "p_holm_within_metric"] = multipletests(
			output.loc[valid, "p_value"],
			method="holm",
		)[1]
	return output

def stable_proximity_tests(df: pd.DataFrame) -> pd.DataFrame:
	rows: list[dict[str, object]] = []
	for pnd in PNDS:
		subset = df[df["pnd"] == pnd]
		control = subset[subset["condition"] == "Control"]
		vpa = subset[subset["condition"] == "VPA"]
		control_time = pd.to_numeric(
			control["total_stable_proximity_time_sec"],
			errors="coerce",
		).dropna()
		vpa_time = pd.to_numeric(
			vpa["total_stable_proximity_time_sec"],
			errors="coerce",
		).dropna()
		if len(control_time) >= 2 and len(vpa_time) >= 2:
			u_stat, p_mann_whitney = stats.mannwhitneyu(
				control_time,
				vpa_time,
				alternative="two-sided",
			)
		else:
			u_stat = p_mann_whitney = np.nan
		control_any = int((pd.to_numeric(control["n_stable_proximity_bouts"], errors="coerce") > 0).sum())
		vpa_any = int((pd.to_numeric(vpa["n_stable_proximity_bouts"], errors="coerce") > 0).sum())
		contingency = [
			[control_any, len(control) - control_any],
			[vpa_any, len(vpa) - vpa_any],
		]
		if len(control) and len(vpa):
			fisher_odds_ratio, p_fisher = stats.fisher_exact(contingency)
		else:
			fisher_odds_ratio = p_fisher = np.nan
		rows.append(
			{
				"pnd": pnd,
				"n_control": len(control_time),
				"n_vpa": len(vpa_time),
				"median_time_control_sec": control_time.median(),
				"median_time_vpa_sec": vpa_time.median(),
				"mann_whitney_u": u_stat,
				"p_mann_whitney": p_mann_whitney,
				"control_sessions_with_bout": control_any,
				"vpa_sessions_with_bout": vpa_any,
				"fisher_odds_ratio": fisher_odds_ratio,
				"p_fisher_any_bout": p_fisher,
			}
		)
	output = pd.DataFrame(rows)
	for p_column, adjusted_column in (
		("p_mann_whitney", "p_holm_mann_whitney"),
		("p_fisher_any_bout", "p_holm_fisher_any_bout"),
	):
		valid = output[p_column].notna()
		output[adjusted_column] = np.nan
		if valid.any():
			output.loc[valid, adjusted_column] = multipletests(
				output.loc[valid, p_column],
				method="holm",
			)[1]
	return output

def within_group_tests(df: pd.DataFrame, metric: str) -> pd.DataFrame:
	rows: list[dict[str, object]] = []
	for condition in ("Control", "VPA"):
		wide = (
			df[df["condition"] == condition]
			.pivot(index="subject_id", columns="pnd", values=metric)
			.reindex(columns=PNDS)
		)
		condition_rows: list[dict[str, object]] = []
		for first, second in itertools.combinations(PNDS, 2):
			paired = wide[[first, second]].apply(pd.to_numeric, errors="coerce").dropna()
			if len(paired) >= 2:
				t_stat, p_value = stats.ttest_rel(paired[second], paired[first])
			else:
				t_stat = p_value = np.nan
			condition_rows.append(
				{
					"metric": metric,
					"condition": condition,
					"pnd_first": first,
					"pnd_second": second,
					"n_pairs": len(paired),
					"paired_t": t_stat,
					"p_value": p_value,
				}
			)
		valid_indices = [
			index for index, value in enumerate(condition_rows)
			if pd.notna(value["p_value"])
		]
		if valid_indices:
			adjusted = multipletests(
				[condition_rows[index]["p_value"] for index in valid_indices],
				method="holm",
			)[1]
			for index, value in zip(valid_indices, adjusted):
				condition_rows[index]["p_holm_within_condition_metric"] = value
		for row in condition_rows:
			row.setdefault("p_holm_within_condition_metric", np.nan)
		rows.extend(condition_rows)
	return pd.DataFrame(rows)

def longitudinal_model(df: pd.DataFrame, metric: str) -> pd.DataFrame:
	subset = df[["subject_id", "condition", "pnd", metric]].copy()
	subset[metric] = pd.to_numeric(subset[metric], errors="coerce")
	subset = subset.dropna(subset=[metric])
	subset["pnd"] = pd.to_numeric(subset["pnd"], errors="raise").astype(int)
	subset["condition"] = subset["condition"].astype(str)
	subset["subject_id"] = subset["subject_id"].astype(str)
	counts = subset.groupby("subject_id")["pnd"].nunique()
	complete_subjects = counts[counts == len(PNDS)].index
	subset = subset[subset["subject_id"].isin(complete_subjects)]
	if subset["subject_id"].nunique() < 4 or subset["condition"].nunique() < 2:
		return pd.DataFrame(
			[{"metric": metric, "model_type": "not_run", "term": "insufficient complete repeated measures"}]
		)
	if pg is not None:
		try:
			result = pg.mixed_anova(
				data=subset,
				dv=metric,
				within="pnd",
				subject="subject_id",
				between="condition",
			).rename(columns={"Source": "term"})
			result.insert(0, "metric", metric)
			result.insert(1, "model_type", "mixed_anova_pingouin")
			result["n_complete_subjects"] = subset["subject_id"].nunique()
			return result
		except Exception:
			pass

	formula = f"{metric} ~ C(condition) * C(pnd)"
	try:
		with warnings.catch_warnings():
			warnings.simplefilter("error", ConvergenceWarning)
			warnings.filterwarnings("error", message=".*singular.*", category=UserWarning)
			fit = smf.mixedlm(
				formula,
				data=subset,
				groups=subset["subject_id"],
			).fit(reml=False, method="lbfgs", disp=False)
		if not bool(getattr(fit, "converged", True)):
			raise RuntimeError("mixed model did not converge")
		model_type = "mixedlm_random_subject_intercept"
	except Exception:
		fit = smf.ols(formula, data=subset).fit(
			cov_type="cluster",
			cov_kwds={"groups": subset["subject_id"]},
		)
		model_type = "ols_subject_cluster_robust"
	confidence = fit.conf_int()
	rows: list[dict[str, object]] = []
	for term in fit.params.index:
		if term == "Group Var":
			continue
		rows.append(
			{
				"metric": metric,
				"model_type": model_type,
				"term": term,
				"estimate": fit.params[term],
				"std_error": fit.bse[term],
				"statistic": fit.tvalues[term],
				"p_value": fit.pvalues[term],
				"ci_low": confidence.loc[term, 0],
				"ci_high": confidence.loc[term, 1],
				"n_complete_subjects": subset["subject_id"].nunique(),
				"n_rows": len(subset),
			}
		)
	return pd.DataFrame(rows)

def group_summary(df: pd.DataFrame) -> pd.DataFrame:
	rows: list[dict[str, object]] = []
	for metric in METRICS:
		if metric not in df:
			continue
		for (condition, pnd), group in df.groupby(["condition", "pnd"]):
			values = pd.to_numeric(group[metric], errors="coerce").dropna()
			rows.append(
				{
					"metric": metric,
					"condition": condition,
					"pnd": pnd,
					"n": len(values),
					"mean": values.mean(),
					"sd": values.std(ddof=1),
					"sem": values.sem(),
					"median": values.median(),
					"q25": values.quantile(0.25),
					"q75": values.quantile(0.75),
				}
			)
	return pd.DataFrame(rows)

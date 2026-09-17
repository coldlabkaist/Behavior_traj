from __future__ import annotations
from pathlib import Path
import sys
from analysis.paths import ROOT
import warnings
import numpy as np
import pandas as pd
import patsy
from scipy import stats
from statsmodels.formula.api import mixedlm
from statsmodels.stats.multitest import multipletests

PNDS = (10, 15, 20)

CONDITIONS = ("Control", "VPA")

METRIC = "pct_time_body_scale_proximity"

def _stars(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"

def _validated_complete_data(metrics: pd.DataFrame) -> pd.DataFrame:
    required = {"subject_id", "condition", "pnd", METRIC}
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data = metrics.loc[:, sorted(required)].copy()
    data["pnd"] = pd.to_numeric(data["pnd"], errors="raise").astype(int)
    data[METRIC] = pd.to_numeric(data[METRIC], errors="raise")
    data = data[
        data["condition"].isin(CONDITIONS) & data["pnd"].isin(PNDS)
    ].copy()
    if data.duplicated(["subject_id", "pnd"]).any():
        duplicates = data.loc[
            data.duplicated(["subject_id", "pnd"], keep=False),
            ["subject_id", "pnd"],
        ]
        raise ValueError(f"Duplicate subject/PND rows found:\n{duplicates}")

    condition_counts = data.groupby("subject_id")["condition"].nunique()
    if not condition_counts.eq(1).all():
        raise ValueError("Each subject_id must belong to exactly one condition")

    pnd_sets = data.groupby("subject_id")["pnd"].agg(lambda x: set(x))
    complete = pnd_sets.eq(set(PNDS))
    if not complete.all():
        incomplete = pnd_sets.loc[~complete]
        raise ValueError(
            "Repeated-measures ANOVA requires all three PNDs per cage; "
            f"incomplete subjects: {incomplete.to_dict()}"
        )

    return data.sort_values(["condition", "subject_id", "pnd"]).reset_index(
        drop=True
    )

def split_plot_anova(data: pd.DataFrame) -> pd.DataFrame:
    """Two-way mixed ANOVA: condition between cages, PND within cages."""
    n_subjects = data["subject_id"].nunique()
    n_groups = len(CONDITIONS)
    n_pnds = len(PNDS)
    grand_mean = float(data[METRIC].mean())

    group_n = data.groupby("condition")["subject_id"].nunique()
    group_mean = data.groupby("condition")[METRIC].mean()
    pnd_mean = data.groupby("pnd")[METRIC].mean()
    cell_mean = data.groupby(["condition", "pnd"])[METRIC].mean()
    subject_mean = data.groupby("subject_id")[METRIC].mean()
    subject_condition = data.groupby("subject_id")["condition"].first()

    ss_group = n_pnds * sum(
        group_n.loc[group] * (group_mean.loc[group] - grand_mean) ** 2
        for group in CONDITIONS
    )
    ss_subject_group = n_pnds * sum(
        (
            subject_mean.loc[subject]
            - group_mean.loc[subject_condition.loc[subject]]
        )
        ** 2
        for subject in subject_mean.index
    )
    ss_pnd = n_subjects * sum(
        (pnd_mean.loc[pnd] - grand_mean) ** 2 for pnd in PNDS
    )
    ss_interaction = sum(
        group_n.loc[group]
        * (
            cell_mean.loc[(group, pnd)]
            - group_mean.loc[group]
            - pnd_mean.loc[pnd]
            + grand_mean
        )
        ** 2
        for group in CONDITIONS
        for pnd in PNDS
    )

    residuals = []
    for row in data.itertuples(index=False):
        residuals.append(
            getattr(row, METRIC)
            - subject_mean.loc[row.subject_id]
            - cell_mean.loc[(row.condition, row.pnd)]
            + group_mean.loc[row.condition]
        )
    ss_error = float(np.square(residuals).sum())

    df_group = n_groups - 1
    df_subject_group = n_subjects - n_groups
    df_pnd = n_pnds - 1
    df_interaction = (n_groups - 1) * (n_pnds - 1)
    df_error = (n_subjects - n_groups) * (n_pnds - 1)

    ms_group = ss_group / df_group
    ms_subject_group = ss_subject_group / df_subject_group
    ms_pnd = ss_pnd / df_pnd
    ms_interaction = ss_interaction / df_interaction
    ms_error = ss_error / df_error

    rows = [
        {
            "effect": "Group",
            "ss": ss_group,
            "df1": df_group,
            "df2": df_subject_group,
            "ms": ms_group,
            "error_term": "Subject(Group)",
            "F": ms_group / ms_subject_group,
            "p_value": stats.f.sf(ms_group / ms_subject_group, df_group, df_subject_group),
            "partial_eta_squared": ss_group / (ss_group + ss_subject_group),
        },
        {
            "effect": "PND",
            "ss": ss_pnd,
            "df1": df_pnd,
            "df2": df_error,
            "ms": ms_pnd,
            "error_term": "PND x Subject(Group)",
            "F": ms_pnd / ms_error,
            "p_value": stats.f.sf(ms_pnd / ms_error, df_pnd, df_error),
            "partial_eta_squared": ss_pnd / (ss_pnd + ss_error),
        },
        {
            "effect": "Group x PND",
            "ss": ss_interaction,
            "df1": df_interaction,
            "df2": df_error,
            "ms": ms_interaction,
            "error_term": "PND x Subject(Group)",
            "F": ms_interaction / ms_error,
            "p_value": stats.f.sf(
                ms_interaction / ms_error,
                df_interaction,
                df_error,
            ),
            "partial_eta_squared": ss_interaction / (ss_interaction + ss_error),
        },
    ]
    result = pd.DataFrame(rows)
    result["significance"] = result["p_value"].map(_stars)
    return result

def tukey_emm_pairwise(
    data: pd.DataFrame,
    anova: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Tukey-adjusted comparisons among six Group x PND marginal means.

    A random-intercept model retains the repeated observations within cage.
    The studentized-range adjustment treats all six marginal means as the
    Tukey family. The repeated-measures error degrees of freedom from the
    split-plot ANOVA are used for the simultaneous inference.
    """
    model_data = data.copy()
    model_data["pnd"] = model_data["pnd"].astype(str)
    formula = (
        f"{METRIC} ~ "
        "C(condition, Treatment(reference='Control')) * "
        "C(pnd, Treatment(reference='10'))"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = mixedlm(
            formula,
            model_data,
            groups=model_data["subject_id"],
        ).fit(reml=True, method="powell", maxiter=1000)
    if not fit.converged:
        raise RuntimeError("Random-intercept model did not converge")

    fixed_names = list(fit.fe_params.index)
    beta = fit.fe_params.to_numpy(dtype=float)
    fixed_cov = fit.cov_params().loc[fixed_names, fixed_names].to_numpy(
        dtype=float
    )

    cell_rows: list[dict[str, object]] = []
    cell_designs: dict[tuple[str, int], np.ndarray] = {}
    for condition in CONDITIONS:
        for pnd in PNDS:
            new_data = pd.DataFrame(
                {"condition": [condition], "pnd": [str(pnd)]}
            )
            design = patsy.build_design_matrices(
                [fit.model.data.design_info],
                new_data,
                return_type="dataframe",
            )[0]
            vector = design.loc[:, fixed_names].to_numpy(dtype=float)[0]
            estimate = float(vector @ beta)
            standard_error = float(np.sqrt(vector @ fixed_cov @ vector))
            cell_designs[(condition, pnd)] = vector
            cell_rows.append(
                {
                    "condition": condition,
                    "pnd": pnd,
                    "estimated_marginal_mean": estimate,
                    "standard_error": standard_error,
                    "n": int(
                        (
                            (data["condition"] == condition)
                            & (data["pnd"] == pnd)
                        ).sum()
                    ),
                }
            )

    df_error = int(
        anova.loc[anova["effect"] == "Group x PND", "df2"].iloc[0]
    )
    n_means = len(CONDITIONS) * len(PNDS)
    q_critical = float(
        stats.studentized_range.ppf(0.95, n_means, df_error) / np.sqrt(2.0)
    )

    pair_rows: list[dict[str, object]] = []
    ordered_cells = [(condition, pnd) for condition in CONDITIONS for pnd in PNDS]
    emm_lookup = {
        (row["condition"], int(row["pnd"])): float(
            row["estimated_marginal_mean"]
        )
        for row in cell_rows
    }
    for first_index, first in enumerate(ordered_cells):
        for second in ordered_cells[first_index + 1 :]:
            contrast = cell_designs[first] - cell_designs[second]
            difference = emm_lookup[first] - emm_lookup[second]
            standard_error = float(
                np.sqrt(contrast @ fixed_cov @ contrast)
            )
            t_value = difference / standard_error
            p_raw = float(2.0 * stats.t.sf(abs(t_value), df_error))
            p_tukey = float(
                stats.studentized_range.sf(
                    abs(t_value) * np.sqrt(2.0),
                    n_means,
                    df_error,
                )
            )
            pair_rows.append(
                {
                    "condition_1": first[0],
                    "pnd_1": first[1],
                    "condition_2": second[0],
                    "pnd_2": second[1],
                    "estimate_1": emm_lookup[first],
                    "estimate_2": emm_lookup[second],
                    "difference_1_minus_2": difference,
                    "standard_error": standard_error,
                    "t_value": t_value,
                    "df": df_error,
                    "p_raw_two_sided": p_raw,
                    "p_tukey_hsd": p_tukey,
                    "simultaneous_ci_low": difference
                    - q_critical * standard_error,
                    "simultaneous_ci_high": difference
                    + q_critical * standard_error,
                    "reject_0_05": p_tukey < 0.05,
                    "significance": _stars(p_tukey),
                    "family": "All 6 Group x PND estimated marginal means",
                }
            )

    all_pairs = pd.DataFrame(pair_rows)
    same_pnd = all_pairs[
        (all_pairs["condition_1"] == "Control")
        & (all_pairs["condition_2"] == "VPA")
        & (all_pairs["pnd_1"] == all_pairs["pnd_2"])
    ].copy()
    same_pnd["pnd"] = same_pnd["pnd_1"].astype(int)
    same_pnd = same_pnd.sort_values("pnd").reset_index(drop=True)
    same_pnd["p_holm_across_3_pnd"] = multipletests(
        same_pnd["p_raw_two_sided"].to_numpy(dtype=float),
        method="holm",
    )[1]
    same_pnd["significance_holm"] = same_pnd[
        "p_holm_across_3_pnd"
    ].map(_stars)

    model_info = pd.DataFrame(
        [
            {
                "model": "Random-intercept repeated-measures model",
                "fixed_effects": "Group * PND",
                "random_effect": "Cage intercept",
                "estimation": "REML",
                "optimizer": "Powell",
                "converged": bool(fit.converged),
                "n_subjects": data["subject_id"].nunique(),
                "n_observations": len(data),
                "random_intercept_variance": float(fit.cov_re.iloc[0, 0]),
                "residual_variance": float(fit.scale),
                "tukey_family_size": n_means,
                "tukey_error_df": df_error,
            }
        ]
    )
    return pd.DataFrame(cell_rows), all_pairs, same_pnd, model_info

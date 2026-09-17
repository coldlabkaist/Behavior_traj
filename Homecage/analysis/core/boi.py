"""core / boi.

Published-analysis functions. Inputs and current sources are recorded in Final manifests.
"""
from __future__ import annotations

import numpy as np

import pandas as pd

from sklearn.linear_model import Ridge

from sklearn.metrics import mean_absolute_error

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

WEEKS = np.arange(3, 9, dtype=int)


FACTORS = ("spacing", "orientation", "dynamics")


FEATURES = tuple(
    [f"{factor}_median" for factor in FACTORS]
    + [f"{factor}_iqr" for factor in FACTORS]
)


ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)


def _model(alpha: float) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=float(alpha))),
        ]
    )


def _summarize(data: pd.DataFrame) -> pd.DataFrame:
    required = {"condition", "cage_id", "sex", "week", *FACTORS}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    rows: list[dict] = []
    for keys, current in data.groupby(
        ["condition", "cage_id", "sex", "week"], sort=True
    ):
        condition, cage_id, sex, week = keys
        row = {
            "condition": str(condition).lower(),
            "cage_id": str(cage_id),
            "sex": str(sex).lower(),
            "week": int(week),
            "n_windows": int(len(current)),
        }
        for factor in FACTORS:
            values = current[factor].to_numpy(dtype=float)
            q25, median, q75 = np.quantile(values, [0.25, 0.5, 0.75])
            row[f"{factor}_median"] = float(median)
            row[f"{factor}_iqr"] = float(q75 - q25)
        rows.append(row)
    result = pd.DataFrame(rows).sort_values(
        ["condition", "cage_id", "week"]
    ).reset_index(drop=True)
    if result[list(FEATURES)].isna().any().any():
        raise RuntimeError("Non-finite summary feature detected")
    return result


def _logo_mae(control: pd.DataFrame, alpha: float) -> float:
    predictions = np.full(len(control), np.nan, dtype=float)
    for cage_id in sorted(control["cage_id"].unique()):
        train = control["cage_id"].ne(cage_id)
        test = ~train
        model = _model(alpha)
        model.fit(control.loc[train, FEATURES], control.loc[train, "week"])
        predictions[test] = model.predict(control.loc[test, FEATURES])
    return float(mean_absolute_error(control["week"], predictions))


def _choose_alpha(control: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    rows = [
        {"alpha": float(alpha), "control_logo_mae": _logo_mae(control, alpha)}
        for alpha in ALPHAS
    ]
    table = pd.DataFrame(rows).sort_values(["control_logo_mae", "alpha"])
    selected = float(table.iloc[0]["alpha"])
    table["selected"] = table["alpha"].eq(selected)
    return selected, table


def _crossfit(
    summaries: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    control = summaries.loc[summaries["condition"].eq("control")].reset_index(drop=True)
    vpa = summaries.loc[summaries["condition"].eq("vpa")].reset_index(drop=True)
    rows: list[dict] = []
    audits: list[dict] = []
    alpha_rows: list[pd.DataFrame] = []
    for held_out in sorted(control["cage_id"].unique()):
        train = control.loc[control["cage_id"].ne(held_out)].reset_index(drop=True)
        test = control.loc[control["cage_id"].eq(held_out)].reset_index(drop=True)
        selected_alpha, alpha_table = _choose_alpha(train)
        alpha_table.insert(0, "held_out_control_cage", held_out)
        alpha_rows.append(alpha_table)
        model = _model(selected_alpha)
        model.fit(train.loc[:, FEATURES], train["week"])

        control_prediction = model.predict(test.loc[:, FEATURES])
        for record, prediction in zip(test.to_dict("records"), control_prediction):
            rows.append(
                {
                    **record,
                    "reference_fold": held_out,
                    "predicted_control_week": float(prediction),
                }
            )
        vpa_prediction = model.predict(vpa.loc[:, FEATURES])
        for record, prediction in zip(vpa.to_dict("records"), vpa_prediction):
            rows.append(
                {
                    **record,
                    "reference_fold": held_out,
                    "predicted_control_week": float(prediction),
                }
            )
        ridge = model.named_steps["ridge"]
        scaler = model.named_steps["scale"]
        audits.append(
            {
                "held_out_control_cage": held_out,
                "selected_alpha": selected_alpha,
                "n_training_control_cages": int(train["cage_id"].nunique()),
                "n_training_control_cage_weeks": int(len(train)),
                "intercept_standardized": float(ridge.intercept_),
                **{
                    f"coefficient_standardized__{feature}": float(value)
                    for feature, value in zip(FEATURES, ridge.coef_)
                },
                **{
                    f"training_mean__{feature}": float(value)
                    for feature, value in zip(FEATURES, scaler.mean_)
                },
                **{
                    f"training_scale__{feature}": float(value)
                    for feature, value in zip(FEATURES, scaler.scale_)
                },
            }
        )

    fold_scores = pd.DataFrame(rows)
    control_scores = fold_scores.loc[fold_scores["condition"].eq("control")].copy()
    vpa_scores = (
        fold_scores.loc[fold_scores["condition"].eq("vpa")]
        .groupby(["condition", "cage_id", "sex", "week"], as_index=False)
        .agg(
            n_windows=("n_windows", "first"),
            predicted_control_week=("predicted_control_week", "mean"),
            reference_sd=("predicted_control_week", "std"),
            reference_low=("predicted_control_week", lambda x: float(np.quantile(x, 0.025))),
            reference_high=("predicted_control_week", lambda x: float(np.quantile(x, 0.975))),
            n_reference_folds=("reference_fold", "nunique"),
        )
    )
    control_scores["reference_sd"] = np.nan
    control_scores["reference_low"] = np.nan
    control_scores["reference_high"] = np.nan
    control_scores["n_reference_folds"] = 1
    scores = pd.concat(
        [
            control_scores[
                [
                    "condition", "cage_id", "sex", "week", "n_windows",
                    "predicted_control_week", "reference_sd", "reference_low",
                    "reference_high", "n_reference_folds",
                ]
            ],
            vpa_scores,
        ],
        ignore_index=True,
    ).sort_values(["condition", "cage_id", "week"]).reset_index(drop=True)
    scores["developmental_score_unanchored"] = (
        scores["predicted_control_week"] - 3.0
    ) / 5.0
    control_scores_only = scores.loc[scores["condition"].eq("control")]
    control_start = float(
        control_scores_only.loc[
            control_scores_only["week"].eq(int(WEEKS[0])), "predicted_control_week"
        ].mean()
    )
    control_end = float(
        control_scores_only.loc[
            control_scores_only["week"].eq(int(WEEKS[-1])), "predicted_control_week"
        ].mean()
    )
    if control_end <= control_start:
        raise RuntimeError("Control endpoint score calibration is not increasing")
    scores["developmental_score"] = (
        scores["predicted_control_week"] - control_start
    ) / (control_end - control_start)
    scores["score_control_3w_reference"] = control_start
    scores["score_control_8w_reference"] = control_end
    return scores, fold_scores, pd.concat(alpha_rows, ignore_index=True).merge(
        pd.DataFrame(audits), on="held_out_control_cage", how="left"
    )


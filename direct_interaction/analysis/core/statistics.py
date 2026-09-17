"""Week-wise Welch tests, category-wise BH correction and mouse coverage."""
from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

METRICS = [
    ("total_seconds", "Total Duration (s)"),
    ("n_bouts", "Bout count"),
]


def compute_week_stats(df: pd.DataFrame, category: str, p_mode: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric, _label in METRICS:
        col = f"{category}_{metric}"
        if col not in df.columns:
            continue
        for week in sorted(df["week"].dropna().astype(int).unique()):
            week_df = df[df["week"].astype(int) == int(week)]
            cont = pd.to_numeric(
                week_df.loc[week_df["condition"] == "cont", col],
                errors="coerce",
            ).dropna()
            exp = pd.to_numeric(
                week_df.loc[week_df["condition"] == "exp", col],
                errors="coerce",
            ).dropna()
            t_stat = math.nan
            p_value = math.nan
            degrees_of_freedom = math.nan
            if len(cont) >= 2 and len(exp) >= 2:
                result = ttest_ind(cont, exp, equal_var=False, nan_policy="omit")
                t_stat = float(result.statistic)
                p_value = float(result.pvalue)
                degrees_of_freedom = float(result.df)
            rows.append(
                {
                    "Category": category,
                    "metric": metric,
                    "week": int(week),
                    "n_cont": int(len(cont)),
                    "n_exp": int(len(exp)),
                    "mean_cont": float(cont.mean()) if len(cont) else math.nan,
                    "mean_exp": float(exp.mean()) if len(exp) else math.nan,
                    "delta_exp_minus_cont": (
                        float(exp.mean() - cont.mean())
                        if len(cont) and len(exp)
                        else math.nan
                    ),
                    "test": "welch_ttest",
                    "t": t_stat,
                    "df": degrees_of_freedom,
                    "p_value": p_value,
                }
            )
    stats = pd.DataFrame(rows)
    stats["q_value"] = benjamini_hochberg(stats["p_value"])
    stats["plot_p_value"] = stats["q_value"] if p_mode == "figure-bh" else stats["p_value"]
    stats["stars"] = stats["plot_p_value"].map(p_marker)
    return stats


def write_coverage(df: pd.DataFrame, output_dir: Path) -> None:
    coverage = (
        df.groupby(["condition", "mouse_id"])["week"]
        .nunique()
        .reset_index(name="n_weeks")
        .groupby(["condition", "n_weeks"])
        .size()
        .reset_index(name="n_mice")
    )
    coverage.to_csv(output_dir / "individual_trajectory_mouse_coverage.csv", index=False)


def sem(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if len(values) <= 1:
        return math.nan
    return float(values.std(ddof=1) / math.sqrt(len(values)))


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    p = pd.to_numeric(p_values, errors="coerce")
    adjusted = pd.Series(np.nan, index=p.index, dtype=float)
    valid = p.dropna()
    n = len(valid)
    if n == 0:
        return adjusted
    order = valid.sort_values().index
    ranked = valid.loc[order].to_numpy(dtype=float)
    raw_adjusted = ranked * n / np.arange(1, n + 1)
    monotone = np.minimum.accumulate(raw_adjusted[::-1])[::-1]
    adjusted.loc[order] = np.clip(monotone, 0, 1)
    return adjusted


def p_marker(p_value: object) -> str:
    p = pd.to_numeric(pd.Series([p_value]), errors="coerce").iloc[0]
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


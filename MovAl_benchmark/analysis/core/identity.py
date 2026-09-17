"""Statistics for the original, paired 17-video identity-switch experiment."""
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests
from .paths import ROOT

IDENTITY_COLUMNS = ["raw_sleap", "raw_dlc", "raw_yolo",
                    "seg_cont_sleap", "seg_cont_dlc", "seg_cont_yolo"]
COMPARISONS = [("raw_sleap", "raw_yolo"), ("raw_dlc", "raw_yolo"),
               ("seg_cont_sleap", "seg_cont_yolo"), ("seg_cont_dlc", "seg_cont_yolo")]


def load_data():
    data = pd.read_excel(ROOT / "Fig2C/data/Mode_comp.xlsx")
    data.columns = data.columns.str.strip()
    if len(data) != 17 or not data.videos.is_unique:
        raise ValueError("Expected 17 unique paired videos")
    if not np.isfinite(data[IDENTITY_COLUMNS].to_numpy(float)).all():
        raise ValueError("Incomplete paired observations")
    return data


def statistics(data):
    long = data.melt(id_vars="videos", var_name="cell", value_name="frequency")
    long["condition"] = long.cell.map(lambda c: "Raw" if c.startswith("raw_") else "Seg Cont")
    long["method"] = long.cell.str.split("_").str[-1]
    table = AnovaRM(long, "frequency", "videos", within=["condition", "method"]).fit().anova_table
    anova = table.rename_axis("effect").reset_index().rename(columns={
        "F Value": "F", "Num DF": "df1", "Den DF": "df2", "Pr > F": "P"})
    pairs = []
    for test, reference in COMPARISONS:
        result = ttest_rel(data[test], data[reference])
        pairs.append(dict(test_column=test, reference_column=reference,
                          t=float(result.statistic), df=int(result.df), P=float(result.pvalue)))
    pairs = pd.DataFrame(pairs)
    pairs["q"] = multipletests(pairs.P, method="fdr_bh")[1]
    return anova, pairs


def caption(anova, pairs):
    labels = ["input condition", "method", "input condition × method"]
    a = "; ".join(f"{label}, F({int(row.df1)},{int(row.df2)}) = {row.F:.2f}, P = "
                  + format(row.P, ".5f" if label == labels[-1] else ".6f")
                  for label, row in zip(labels, anova.itertuples()))
    labels = ["Raw SLEAP", "Raw DLC", "Seg-Cont SLEAP", "Seg-Cont DLC"]
    p = "; ".join(f"{label}, t({int(row.df)}) = {row.t:.2f}, q = {row.q:.4f}"
                  for label, row in zip(labels, pairs.itertuples()))
    return ("(C) Identity-switch frequency. (Top) Representative images showing an identity-switch event. "
            "(Bottom) Identity-switch frequencies across methods and input conditions for 17 videos. "
            "Points represent individual videos, and box plots show the distributions "
            f"(two-way repeated-measures ANOVA: {a}; two-sided paired t-tests with "
            "Benjamini–Hochberg correction for comparisons with MovAl under the corresponding "
            f"input condition: {p}). *q < 0.05, **q < 0.01; ns, not significant.\n")

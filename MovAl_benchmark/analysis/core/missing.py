
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests


from .paths import ROOT
DATA_PATH = ROOT / "Fig2B/data/tracking_miss_long.csv"

KEYPOINT_ORDER = ["Nose", "Ear_L", "Ear_R", "Body_C", "Tail"]
CONDITION_ORDER = [
    "Raw SLEAP",
    "Seg-Cont SLEAP",
    "Raw DLC",
    "Seg-Cont DLC",
    "Raw MovAl",
    "Seg-Cont MovAl",
]


def load_video_level_data() -> pd.DataFrame:
    """Build one tracking-miss estimate per video, key point, and condition."""
    long = pd.read_csv(DATA_PATH)
    keys = ["video", "method", "input", "track", "keypoint"]
    if long.duplicated(keys).any() or long[keys + ["na_percent"]].isna().any().any():
        raise ValueError("Missing or duplicate track-level observations")
    # Keep Seg-only source observations in the file, but use the two paper inputs.
    selected = long[long["input"].isin(["Raw", "Seg-Cont"])].copy()
    out = selected.groupby(["video", "keypoint", "input", "method"], as_index=False)["na_percent"].mean()
    out["condition"] = out["input"] + " " + out["method"]
    expected = 17 * len(KEYPOINT_ORDER) * len(CONDITION_ORDER)
    if len(out) != expected or out["video"].nunique() != 17:
        raise ValueError(f"Expected 17 videos and {expected} video-level rows")
    if set(out["condition"]) != set(CONDITION_ORDER) or set(out["keypoint"]) != set(KEYPOINT_ORDER):
        raise ValueError("Unexpected analysis conditions or keypoints")
    return out


def descriptive_stats(data: pd.DataFrame) -> pd.DataFrame:
    summary = (
        data.groupby(["keypoint", "condition"])["na_percent"]
        .agg(mean="mean", sd="std", n="count")
        .reset_index()
    )
    summary["sem"] = summary["sd"] / np.sqrt(summary["n"])
    summary["keypoint"] = pd.Categorical(
        summary["keypoint"], KEYPOINT_ORDER, ordered=True
    )
    summary["condition"] = pd.Categorical(
        summary["condition"], CONDITION_ORDER, ordered=True
    )
    return summary.sort_values(["keypoint", "condition"])


def repeated_anova(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keypoint in KEYPOINT_ORDER:
        sub = data[data["keypoint"] == keypoint].copy()
        table = AnovaRM(
            sub,
            depvar="na_percent",
            subject="video",
            within=["input", "method"],
        ).fit().anova_table
        for effect, values in table.iterrows():
            rows.append(
                {
                    "keypoint": keypoint,
                    "effect": effect,
                    "F": float(values["F Value"]),
                    "df_num": float(values["Num DF"]),
                    "df_den": float(values["Den DF"]),
                    "p": float(values["Pr > F"]),
                }
            )
    return pd.DataFrame(rows)


def holm_posthoc(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pairs = [
        (CONDITION_ORDER[i], CONDITION_ORDER[j])
        for i in range(len(CONDITION_ORDER))
        for j in range(i + 1, len(CONDITION_ORDER))
    ]
    for keypoint in KEYPOINT_ORDER:
        wide = (
            data[data["keypoint"] == keypoint]
            .pivot(index="video", columns="condition", values="na_percent")
            .reindex(columns=CONDITION_ORDER)
        )
        p_values = []
        local_rows = []
        for condition_1, condition_2 in pairs:
            stat = ttest_rel(
                wide[condition_1].to_numpy(),
                wide[condition_2].to_numpy(),
                nan_policy="raise",
            )
            diff = wide[condition_1] - wide[condition_2]
            row = {
                "keypoint": keypoint,
                "condition_1": condition_1,
                "condition_2": condition_2,
                "mean_difference_1_minus_2": float(diff.mean()),
                "sem_difference": float(diff.std(ddof=1) / np.sqrt(len(diff))),
                "t": float(stat.statistic),
                "df": int(len(diff) - 1),
                "p_raw": float(stat.pvalue),
            }
            local_rows.append(row)
            p_values.append(stat.pvalue)
        reject, p_holm, _, _ = multipletests(p_values, method="holm")
        for row, adjusted, rejected in zip(local_rows, p_holm, reject):
            row["p_holm"] = float(adjusted)
            row["significant_holm_0.05"] = bool(rejected)
            rows.append(row)
    return pd.DataFrame(rows)


def plot_summary(long_data):
    """Mean and SEM across 17 videos after averaging their three tracks."""
    selected = long_data[long_data.input.isin(["Raw", "Seg-Cont"])]
    video = selected.groupby(["video", "keypoint", "method", "input"], as_index=False).na_percent.mean()
    return video.groupby(["keypoint", "method", "input"], as_index=False).na_percent.agg(
        mean="mean", sem="sem", n="count")



from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests


from .paths import ROOT

DATA_ROOT = ROOT.parent / "data/csv/Fig2BCD_FigS1"
METHOD_FOLDERS = {
    "Raw SLEAP": "Raw_sleap",
    "Seg-Cont SLEAP": "SegCont_sleap",
    "Raw MovAl": "Raw_moval",
    "Seg-Cont MovAl": "SegCont_moval",
}
METHOD_ORDER = list(METHOD_FOLDERS)
KEYPOINTS = ["Body_C", "Nose", "Tail"]
BANDS = {
    "Low": (0.00, 0.05),
    "Mid": (0.05, 0.15),
    "High": (0.15, 0.50),
}
TRACKS = ["track_0", "track_1", "track_2"]
FRAMES = np.arange(9000)


def load_pose_csv(path: Path, method: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "frame.idx" in df.columns:
        df = df.rename(columns={"frame.idx": "frame_idx"})
        df["frame_idx"] = df["frame_idx"] - 1
    df = df[df["frame_idx"].between(0, 8999)].copy()
    full_index = pd.MultiIndex.from_product(
        [FRAMES, TRACKS], names=["frame_idx", "track"]
    ).to_frame(index=False)
    return full_index.merge(df, on=["frame_idx", "track"], how="left")


def summarize_file(path: Path, method: str) -> list[dict]:
    df = load_pose_csv(path, method)
    video_id = path.stem.removesuffix("_1000model")
    rows: list[dict] = []
    for track, track_df in df.groupby("track", sort=True):
        track_df = track_df.sort_values("frame_idx")
        for keypoint in KEYPOINTS:
            x = pd.to_numeric(track_df[f"{keypoint}.x"], errors="coerce").to_numpy()
            y = pd.to_numeric(track_df[f"{keypoint}.y"], errors="coerce").to_numpy()
            dx = np.diff(x)
            dy = np.diff(y)
            jitter = np.sqrt(dx**2 + dy**2)
            jitter[~np.isfinite(dx) | ~np.isfinite(dy)] = np.nan
            # Preserve the original notebook definition: zero and missing displacements
            # are omitted before log10 transformation and FFT.
            jitter = jitter[np.isfinite(jitter) & (jitter > 0)]
            if jitter.size < 10:
                continue
            signal = np.log10(jitter)
            signal = signal - np.mean(signal)
            frequency = np.fft.rfftfreq(signal.size, d=1.0)
            power = np.abs(np.fft.rfft(signal)) ** 2
            for band, (low, high) in BANDS.items():
                mask = (frequency >= low) & (frequency < high)
                rows.append(
                    {
                        "video": video_id,
                        "track": track,
                        "keypoint": keypoint,
                        "band": band,
                        "method": method,
                        "band_power": float(power[mask].sum()),
                        "n_displacements": int(jitter.size),
                    }
                )
    return rows


def compute_band_power() -> pd.DataFrame:
    rows: list[dict] = []
    for method, folder in METHOD_FOLDERS.items():
        files = sorted((DATA_ROOT / folder).glob("*.csv"))
        if len(files) != 17:
            raise RuntimeError(f"Expected 17 files for {method}, found {len(files)}")
        for path in files:
            rows.extend(summarize_file(path, method))
    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError("No band-power observations were generated")
    return result


def repeated_measures_statistics(video_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    anova_rows: list[dict] = []
    posthoc_rows: list[dict] = []

    for keypoint in KEYPOINTS:
        for band in BANDS:
            subset = video_df[
                (video_df["keypoint"] == keypoint) & (video_df["band"] == band)
            ].copy()
            wide = subset.pivot(index="video", columns="method", values="band_power")
            wide = wide.reindex(columns=METHOD_ORDER).dropna()
            long = wide.reset_index().melt(
                id_vars="video", var_name="condition", value_name="band_power"
            )
            long["method"] = long["condition"].map(
                {
                    "Raw SLEAP": "SLEAP",
                    "Seg-Cont SLEAP": "SLEAP",
                    "Raw MovAl": "MovAl",
                    "Seg-Cont MovAl": "MovAl",
                }
            )
            long["input"] = long["condition"].map(
                {
                    "Raw SLEAP": "Raw",
                    "Seg-Cont SLEAP": "Seg-Cont",
                    "Raw MovAl": "Raw",
                    "Seg-Cont MovAl": "Seg-Cont",
                }
            )
            fit = AnovaRM(
                long,
                depvar="band_power",
                subject="video",
                within=["method", "input"],
            ).fit()
            for effect, row in fit.anova_table.iterrows():
                anova_rows.append(
                    {
                        "keypoint": keypoint,
                        "band": band,
                        "effect": effect,
                        "n_videos": len(wide),
                        "F": float(row["F Value"]),
                        "df_num": float(row["Num DF"]),
                        "df_den": float(row["Den DF"]),
                        "P": float(row["Pr > F"]),
                    }
                )

            band_pairs: list[dict] = []
            for first, second in combinations(METHOD_ORDER, 2):
                test = ttest_rel(wide[first], wide[second], nan_policy="omit")
                band_pairs.append(
                    {
                        "keypoint": keypoint,
                        "band": band,
                        "group1": first,
                        "group2": second,
                        "n_videos": len(wide),
                        "mean_group1": float(wide[first].mean()),
                        "mean_group2": float(wide[second].mean()),
                        "mean_difference": float((wide[first] - wide[second]).mean()),
                        "t": float(test.statistic),
                        "df": len(wide) - 1,
                        "P": float(test.pvalue),
                    }
                )
            adjusted = multipletests(
                [row["P"] for row in band_pairs], method="holm"
            )[1]
            for row, p_adjusted in zip(band_pairs, adjusted):
                row["P_holm"] = float(p_adjusted)
                posthoc_rows.append(row)

    return pd.DataFrame(anova_rows), pd.DataFrame(posthoc_rows)


def video_summary(track_df):
    return track_df.groupby(["video", "keypoint", "band", "method"], as_index=False).agg(
        band_power=("band_power", "mean"), n_tracks=("track", "nunique"))


def descriptive_summary(video_df):
    return video_df.groupby(["keypoint", "band", "method"], as_index=False).agg(
        n_videos=("video", "nunique"), mean_power=("band_power", "mean"),
        sd_power=("band_power", "std"),
        se_power=("band_power", lambda x: x.std(ddof=1) / np.sqrt(x.count())))


def representative_traces(video="240508_5w_031_m_hc_first", track="track_0"):
    """The original notebook selected video index 10 and track_0.

    Resolve that index to the explicit alphabetically ordered filename.
    """
    parts = []
    for method, folder in METHOD_FOLDERS.items():
        matches = [p for p in (DATA_ROOT / folder).glob("*.csv")
                   if p.stem.removesuffix("_1000model") == video]
        if len(matches) != 1:
            raise ValueError(f"Cannot uniquely resolve representative video for {method}")
        df = load_pose_csv(matches[0], method)
        df = df[df.track == track].sort_values("frame_idx")
        for kp in KEYPOINTS:
            xy = df[[f"{kp}.x", f"{kp}.y"]].replace(0, np.nan).to_numpy(float)
            delta = np.diff(xy, axis=0)
            distance = np.sqrt((delta ** 2).sum(axis=1))
            valid = np.isfinite(distance) & (distance > 0)
            parts.append(pd.DataFrame({"method": method, "keypoint": kp,
                "frame_idx": df.frame_idx.to_numpy()[1:][valid],
                "jitter_log": np.log10(distance[valid]), "video": video, "track": track}))
    return pd.concat(parts, ignore_index=True)


def plot_summary(track_data):
    # Match the saved notebook's bar/error-bar aggregation over video/track pairs.
    return track_data.groupby(["keypoint", "band", "method"], as_index=False).agg(
        mean_power=("band_power", "mean"), se_power=("band_power", "sem"),
        n_tracks=("track", "size"))



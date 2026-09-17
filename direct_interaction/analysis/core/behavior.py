"""A/B prediction probabilities to the Fig5D and FigS9AB mouse table."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from core.metadata import MODELS, build_metadata

MODEL_BEHAVIORS = {
    "modelA": ["Approach", "Following", "Mounting", "Facing"],
    "modelB": ["Nose-Head", "Nose-Body", "Nose-Anogenital"],
}
CATEGORIES = {
    "Attentive": ["Approach", "Facing", "Following"],
    "Prosocial": ["Nose-Head", "Nose-Body", "Nose-Anogenital", "Mounting"],
}
CATEGORIES["Social"] = CATEGORIES["Attentive"] + CATEGORIES["Prosocial"]


def build_individual_behavior_table(
    result_dir: Path,
    *,
    threshold: float = 0.5,
    fps: int = 30,
    max_gap_frames: int = 15,
    min_bout_frames: int = 15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Union category frames, merge short gaps, then filter short bouts.

    Shorter model predictions are padded with negative frames, matching the
    paper's max_zero_pad policy. Social is measured from the seven-behavior
    union, not by adding the two component categories' measurements.
    """
    samples = [sample for sample in build_metadata(Path(result_dir))
               if sample.source_schema == "b6_vpa_pnd_id"]
    rows, qc_rows = [], []
    for sample in samples:
        predictions = {
            model: _read_thresholded_behaviors(
                Path(result_dir) / model / getattr(sample, f"{model}_file_name"),
                MODEL_BEHAVIORS[model], threshold,
            )
            for model in MODELS
        }
        frame_count = max(len(frame) for frame in predictions.values())
        merged = pd.DataFrame({
            behavior: np.pad(frame[behavior].to_numpy(dtype="int8"),
                             (0, frame_count - len(frame)))
            for frame in predictions.values() for behavior in frame
        })
        row = {
            "video": Path(sample.representative_file_name).stem,
            "mouse_id": f"{sample.cage_number}_id{sample.id_number}",
            "cage_number": sample.cage_number,
            "id_number": int(sample.id_number),
            "condition": sample.condition,
            "sex": sample.sex,
            "week": sample.week,
            "pnd": sample.pnd,
            "source_schema": sample.source_schema,
            "frame_count": frame_count,
            "video_seconds": frame_count / fps,
            "fps": fps,
            "max_gap_frames": max_gap_frames,
            "min_bout_frames": min_bout_frames,
        }
        measurements = {}
        for category, members in CATEGORIES.items():
            union = _category_union_vector(merged, members)
            ranges = _filter_min_bout_frames(
                _merged_bout_ranges_r_style(union, max_gap_frames), min_bout_frames,
            )
            durations = ranges[:, 1] - ranges[:, 0] + 1
            measurements[category] = {
                "total_seconds": float(durations.sum() / fps),
                "n_bouts": int(len(durations)),
                "mean_seconds_per_bout": float(durations.mean() / fps) if len(durations) else math.nan,
            }
        for metric in ["total_seconds", "n_bouts", "mean_seconds_per_bout"]:
            for category in CATEGORIES:
                row[f"{category}_{metric}"] = measurements[category][metric]
        rows.append(row)
        qc = {
            "sample_key": sample.sample_key,
            "file_name": sample.representative_file_name,
            "frame_count_match": len({len(frame) for frame in predictions.values()}) == 1,
            "written_frame_count": frame_count,
            "frame_policy": "max_zero_pad",
        }
        for model, frame in predictions.items():
            qc[f"{model}_file_name"] = getattr(sample, f"{model}_file_name")
            qc[f"{model}_frame_count"] = len(frame)
            qc[f"{model}_padded_frames"] = frame_count - len(frame)
        qc_rows.append(qc)
    if not rows:
        raise ValueError("No B6 observations found in the A/B prediction folders")
    data = pd.DataFrame(rows).sort_values(["mouse_id", "week"]).reset_index(drop=True)
    return data, pd.DataFrame(qc_rows)


def _read_thresholded_behaviors(
    path: Path,
    behaviors: list[str],
    threshold: float,
) -> pd.DataFrame:
    usecols = [f"Probability_{behavior}" for behavior in behaviors]
    df = pd.read_csv(path, usecols=usecols)
    out = pd.DataFrame()
    for behavior in behaviors:
        probability = pd.to_numeric(
            df[f"Probability_{behavior}"],
            errors="coerce",
        ).fillna(0)
        out[behavior] = (probability >= threshold).astype("int8")
    return out



def _category_union_vector(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    if not cols:
        return np.zeros(len(df), dtype=bool)
    values = df.loc[:, cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    return (values.to_numpy() != 0).any(axis=1)



def _merged_bout_ranges_r_style(binary: np.ndarray, max_gap_frames: int) -> np.ndarray:
    x = np.asarray(binary).astype(bool)
    idx = np.flatnonzero(x)
    if len(idx) == 0:
        return np.empty((0, 2), dtype=int)

    run_breaks = np.r_[0, np.flatnonzero(np.diff(idx) > 1) + 1]
    run_starts = idx[run_breaks]
    run_ends = np.r_[idx[run_breaks[1:] - 1], idx[-1]]

    merged: list[tuple[int, int]] = []
    cur_start = int(run_starts[0])
    cur_end = int(run_ends[0])
    for start, end in zip(run_starts[1:], run_ends[1:]):
        gap = int(start) - cur_end
        if gap <= max_gap_frames:
            cur_end = int(end)
        else:
            merged.append((cur_start, cur_end))
            cur_start = int(start)
            cur_end = int(end)
    merged.append((cur_start, cur_end))
    return np.asarray(merged, dtype=int)



def _filter_min_bout_frames(ranges: np.ndarray, min_bout_frames: int) -> np.ndarray:
    if min_bout_frames <= 1 or ranges.size == 0:
        return ranges
    durations = ranges[:, 1] - ranges[:, 0] + 1
    return ranges[durations >= min_bout_frames]


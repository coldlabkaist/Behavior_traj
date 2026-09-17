"""Pooled coordinate RMSE using all five clips and both animal label formats.

Take the square root of full-precision MSE; round only the displayed RMSE.
No reassignment or optimization of animal identities is applied.
"""
import numpy as np
import pandas as pd
from .paths import ROOT

DATA_ROOT = ROOT / "Fig2E/data"
SOURCE_ROOT = ROOT.parent / "data/csv/Fig2E"
KEYPOINTS = ["Nose", "Ear_L", "Ear_R", "Body_C", "Tail"]
CONDITIONS = [("SLEAP", "Raw"), ("SLEAP", "Seg-Cont"),
              ("DLC", "Raw"), ("DLC", "Seg-Cont"),
              ("MovAl", "Raw"), ("MovAl", "Seg-Cont")]
SOURCES = [
    ("Raw_sleap", "SLEAP", "Raw", 1920, 1080),
    ("SegCont_sleap", "SLEAP", "Seg-Cont", 1280, 720),
    ("raw_dlc", "DLC", "Raw", 1920, 1080),
    ("SegCont_dlc", "DLC", "Seg-Cont", 1920, 1080),
    ("Raw_moval", "MovAl", "Raw", 1920, 1080),
    ("SegCont_moval", "MovAl", "Seg-Cont", 1920, 1080),
]


def load_summary():
    data = pd.read_csv(DATA_ROOT / "rmse_summary.csv")
    if len(data) != 30 or data.duplicated(["keypoint", "method", "input"]).any():
        raise ValueError("Expected 30 unique heatmap cells")
    np.testing.assert_allclose(data.rmse, np.sqrt(data.mse), rtol=1e-12, atol=1e-12)
    return data


def load_coordinates(path, sx=1, sy=1):
    data = pd.read_csv(path)
    if "frame.idx" in data:
        data = data.rename(columns={"frame.idx": "frame_idx"})
        data["frame_idx"] -= 1
    data = data[data.frame_idx.between(0, 249)].copy()
    data["track"] = data.track.replace({f"animal_{i+1}": f"track_{i}" for i in range(3)})
    if not data.track.isin(["track_0", "track_1", "track_2"]).all():
        raise ValueError(f"Unexpected animal labels in {path}")
    if data.duplicated(["frame_idx", "track"]).any():
        raise ValueError(f"Duplicate frame/track observations in {path}")
    for kp in KEYPOINTS:
        data[f"{kp}.x"] = data[f"{kp}.x"].replace(0, np.nan) / sx
        data[f"{kp}.y"] = data[f"{kp}.y"].replace(0, np.nan) / sy
    return data


def calculate_from_coordinates():
    gt_paths = {p.stem.removeprefix("Contoured_"): p for p in (SOURCE_ROOT / "gt").glob("*.csv")}
    if len(gt_paths) != 5:
        raise ValueError("Expected five ground-truth clips")
    records = []
    for folder, method, condition, sx, sy in SOURCES:
        paths = sorted((SOURCE_ROOT / folder).glob("*.csv"))
        if {p.stem for p in paths} != set(gt_paths):
            raise ValueError(f"Prediction/ground-truth clip mismatch: {folder}")
        errors = {kp: [] for kp in KEYPOINTS}
        for path in paths:
            pred = load_coordinates(path, sx, sy)
            gt = load_coordinates(gt_paths[path.stem])
            merged = pred.merge(gt, on=["frame_idx", "track"], suffixes=("_pred", "_gt"),
                                validate="one_to_one")
            for kp in KEYPOINTS:
                xy = merged[[f"{kp}.x_pred", f"{kp}.y_pred", f"{kp}.x_gt", f"{kp}.y_gt"]].dropna().to_numpy()
                errors[kp].extend(((xy[:, :2] - xy[:, 2:]) ** 2).sum(axis=1))
        for kp, values in errors.items():
            if not values:
                raise ValueError(f"No paired coordinates for {folder}/{kp}")
            mse = float(np.mean(values))
            records.append(dict(keypoint=kp, method=method, input=condition, n_pairs=len(values),
                                mse=mse, rmse=float(np.sqrt(mse))))
    return pd.DataFrame(records)

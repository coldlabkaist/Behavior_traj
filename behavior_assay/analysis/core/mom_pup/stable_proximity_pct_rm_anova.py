from __future__ import annotations
from pathlib import Path
import sys
from analysis.paths import ROOT
import numpy as np
import pandas as pd

SOURCE_TIME = "total_stable_proximity_time_sec"

SOURCE_METRIC = "pct_time_stable_proximity"

def derive_percentage(raw: pd.DataFrame) -> pd.DataFrame:
    required = {
        SOURCE_TIME,
        "n_valid_frames_mom_cluster",
        "fps",
    }
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    result = raw.copy()
    stable_time = pd.to_numeric(result[SOURCE_TIME], errors="raise")
    valid_frames = pd.to_numeric(
        result["n_valid_frames_mom_cluster"], errors="raise"
    )
    fps = pd.to_numeric(result["fps"], errors="raise")
    valid_time = valid_frames / fps
    if (valid_time <= 0).any():
        raise ValueError("All sessions must have positive valid observation time")
    if (stable_time > valid_time + 1e-9).any():
        raise ValueError("Stable proximity time exceeds valid observation time")

    result["valid_mom_cluster_time_sec"] = valid_time
    result[SOURCE_METRIC] = 100.0 * stable_time / valid_time
    if not np.isfinite(result[SOURCE_METRIC]).all():
        raise ValueError("Non-finite stable proximity percentages found")
    return result

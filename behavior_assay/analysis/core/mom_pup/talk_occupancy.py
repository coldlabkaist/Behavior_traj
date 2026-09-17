from __future__ import annotations
from pathlib import Path
import sys
from analysis.paths import ROOT
from pathlib import Path
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
from analysis.core.mom_pup.spatial import _hist2d_counts

CONTROL = "#304F78"

VPA = "#C4475B"

PNDS = (10, 15, 20)

SELECTIONS = {
	("Control", 10): "B6_VPA_08",
	("VPA", 10): "B6_VPA_15",
	("Control", 15): "B6_VPA_10",
	("VPA", 15): "B6_VPA_11",
	("Control", 20): "B6_VPA_08",
	("VPA", 20): "B6_VPA_05",
}

def _load_selected_sessions(manifest_path: Path) -> dict[tuple[str, int], pd.DataFrame]:
	manifest = pd.read_csv(manifest_path)
	manifest["pnd"] = pd.to_numeric(manifest["pnd"], errors="coerce")
	manifest["include"] = manifest["include"].astype(str).str.lower().eq("true")

	sessions: dict[tuple[str, int], pd.DataFrame] = {}
	for key, subject_id in SELECTIONS.items():
		condition, pnd = key
		matched = manifest[
			(manifest["subject_id"] == subject_id)
			& (manifest["condition"] == condition)
			& (manifest["pnd"] == pnd)
			& manifest["include"]
		]
		if len(matched) != 1:
			raise ValueError(
				f"Expected one included manifest row for {condition}, PND {pnd}, "
				f"{subject_id}; found {len(matched)}."
			)
		raw_path = Path(matched.iloc[0]["raw_path"])
		df = pd.read_csv(raw_path, usecols=["track", "Body_C.x", "Body_C.y"])
		df["track"] = df["track"].astype(str)
		sessions[key] = df
	return sessions

def _session_histograms(
	df: pd.DataFrame,
	*,
	bins: int,
) -> tuple[np.ndarray, np.ndarray]:
	mom = df.loc[df["track"] == "track_0", ["Body_C.x", "Body_C.y"]].dropna()
	pup = df.loc[df["track"] != "track_0", ["Body_C.x", "Body_C.y"]].dropna()
	pup_h = _hist2d_counts(
		pup["Body_C.x"].to_numpy(dtype=float),
		pup["Body_C.y"].to_numpy(dtype=float),
		bins,
		(0.0, 1.0),
		(0.0, 1.0),
	)
	mom_h = _hist2d_counts(
		mom["Body_C.x"].to_numpy(dtype=float),
		mom["Body_C.y"].to_numpy(dtype=float),
		bins,
		(0.0, 1.0),
		(0.0, 1.0),
	)
	return pup_h, mom_h

def _shared_log_norm(
	histograms: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]],
	*,
	vmax_quantile: float,
) -> LogNorm:
	positive = np.concatenate(
		[
			h.ravel()[h.ravel() > 0]
			for pair in histograms.values()
			for h in pair
		]
	)
	vmax = float(np.quantile(positive, vmax_quantile))
	return LogNorm(vmin=1.0, vmax=max(vmax, 2.0))

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np


@dataclass
class Schema:
	# Keypoint names without channels; channels are (x,y,score)
	keypoints: List[str]
	# Column aliases mapping non-standard names to standard ones
	alias_map: Dict[str, str] = field(default_factory=dict)
	# Pandas dtype map for required columns
	dtype_map: Dict[str, str] = field(default_factory=lambda: {
		"frame_idx": "Int64",
		"track": "string",
		"instance.score": "float32",
	})
	# NA tokens for reading
	na_values: List[str] = field(default_factory=lambda: ["", "NA", "NaN", "nan", "None"])
	# Coordinate mode: "unit" means [0,1]; "pixel" requires image_size
	coord_mode: str = "unit"

	@property
	def required_columns(self) -> List[str]:
		cols = ["frame_idx", "track", "instance.score"]
		for kp in self.keypoints:
			cols.extend([f"{kp}.x", f"{kp}.y", f"{kp}.score"])
		return cols

	def coerce_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
		for col, dtype in self.dtype_map.items():
			if col in df.columns:
				try:
					df[col] = df[col].astype(dtype)
				except Exception:
					df[col] = pd.to_numeric(df[col], errors="coerce")
		return df

	def normalize_coords(self, df: pd.DataFrame, coord_mode: str, image_size: Tuple[int, int] | None) -> pd.DataFrame:
		# If current data is in unit coords but user requests pixel, or vice versa, perform conversion
		# For now assume input already in requested mode; conversion hooks can be added later
		return df


def normalize_columns(df: pd.DataFrame, alias_map: Dict[str, str]) -> pd.DataFrame:
	if not alias_map:
		return df
	columns = {c: alias_map.get(c, c) for c in df.columns}
	return df.rename(columns=columns)


def infer_keypoints(df: pd.DataFrame) -> List[str]:
	bases: List[str] = []
	for col in df.columns:
		if "." not in col:
			continue
		name, chan = col.rsplit(".", 1)
		if chan in {"x", "y", "score"}:
			bases.append(name)
	return sorted(set(bases))


def validate_or_raise(df: pd.DataFrame, schema: Schema) -> dict:
	report = {"rows": len(df), "missing_columns": [], "na_rates": {}, "keypoints": []}
	# Columns
	for col in ["frame_idx", "track", "instance.score"]:
		if col not in df.columns:
			report["missing_columns"].append(col)
	# Keypoints present
	report["keypoints"] = infer_keypoints(df)
	# NA rates for required basics
	for col in [c for c in df.columns if c.endswith(('.x', '.y', '.score'))] + ["instance.score"]:
		report["na_rates"][col] = float(pd.isna(df[col]).mean()) if col in df.columns else 1.0
	if report["missing_columns"]:
		raise ValueError(f"Missing required columns: {report['missing_columns']}")
	return report

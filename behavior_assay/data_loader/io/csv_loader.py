from __future__ import annotations

from pathlib import Path
from typing import Tuple
import pandas as pd

from .schema import Schema, normalize_columns, validate_or_raise


def read_pose_csv(
	path: str | Path,
	schema: Schema,
	*,
	coord_mode: str | None = None,
	image_size: tuple[int, int] | None = None,
	usecols: list[str] | None = None,
	chunksize: int | None = None,
) -> Tuple[pd.DataFrame, dict]:
	# Basic read; can expand to chunksize later
	df = pd.read_csv(path, usecols=usecols, na_values=schema.na_values)
	df = normalize_columns(df, schema.alias_map)
	df = schema.coerce_dtypes(df)
	if coord_mode is None:
		coord_mode = schema.coord_mode
	df = schema.normalize_coords(df, coord_mode, image_size)

	report = validate_or_raise(df, schema)
	df = df.sort_values([c for c in ["frame_idx", "track"] if c in df.columns]).reset_index(drop=True)
	return df, report

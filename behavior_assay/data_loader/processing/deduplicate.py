from __future__ import annotations

import pandas as pd


def deduplicate_frame_track(
	df: pd.DataFrame,
	policy: str = "highest_instance_score",
) -> pd.DataFrame:
	"""
	Ensure at most one row per (frame_idx, track).
	policy options:
	- highest_instance_score: keep row with max instance.score
	- highest_kp_score_sum: keep row with max sum of all <kp>.score columns
	"""
	if "frame_idx" not in df.columns or "track" not in df.columns:
		raise ValueError("DataFrame must contain 'frame_idx' and 'track'")

	gcols = ["frame_idx", "track"]
	work = df.copy()

	if policy == "highest_instance_score":
		if "instance.score" not in work.columns:
			# if missing, fallback to first occurrence
			return (
				work.sort_values(gcols)
				.drop_duplicates(subset=gcols, keep="first")
				.reset_index(drop=True)
			)
		idx = work.groupby(gcols)["instance.score"].idxmax()
		return work.loc[idx].sort_values(gcols).reset_index(drop=True)

	elif policy == "highest_kp_score_sum":
		score_cols = [c for c in work.columns if c.endswith(".score")]
		if not score_cols:
			return (
				work.sort_values(gcols)
				.drop_duplicates(subset=gcols, keep="first")
				.reset_index(drop=True)
			)
		work["_kp_score_sum"] = work[score_cols].sum(axis=1, numeric_only=True)
		idx = work.groupby(gcols)["_kp_score_sum"].idxmax()
		out = work.loc[idx].drop(columns=["_kp_score_sum"], errors="ignore")
		return out.sort_values(gcols).reset_index(drop=True)

	else:
		# default fallback: first occurrence per group
		return (
			work.sort_values(gcols)
			.drop_duplicates(subset=gcols, keep="first")
			.reset_index(drop=True)
		)

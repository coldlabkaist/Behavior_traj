from __future__ import annotations
from pathlib import Path
import sys
from analysis.paths import ROOT, resolve_input_path
from pathlib import Path
import pandas as pd

KEYPOINTS = ["Nose", "Body_C", "Ear_L", "Ear_R", "Neck", "Tail"]

DEFAULT_INTERP_KEYPOINTS = ["Body_C", "Nose"]

SUMMARY_COLUMNS = [
	"date",
	"condition",
	"session_key",
	"subject_id",
	"sex",
	"phase",
	"phase_detail",
	"hab_type",
	"trial_id",
	"session_n",
	"social_side",
	"empty_side",
	"familiar_side",
	"novel_side",
	"heatmap_target_label",
	"heatmap_target_side",
	"mirror_lr_for_target_right",
	"layout_raw",
	"pose_path",
	"output_path",
	"rows_in",
	"rows_after_dedup",
	"rows_after_pad",
	"tracks",
	"frame_min",
	"frame_max",
	"dense_frame_count",
	"duplicate_rows_removed",
	"padded_rows_added",
	"interpolated_keypoints",
	"Body_C_filled_x",
	"Body_C_filled_y",
	"Body_C_long_gaps",
	"Body_C_remaining_nan_x",
	"Body_C_remaining_nan_y",
	"Nose_filled_x",
	"Nose_filled_y",
	"Nose_long_gaps",
	"Nose_remaining_nan_x",
	"Nose_remaining_nan_y",
	"method",
]

ISSUE_COLUMNS = [
	"date",
	"condition",
	"session_key",
	"subject_id",
	"phase",
	"phase_detail",
	"hab_type",
	"trial_id",
	"session_n",
	"pose_path",
	"issue",
	"detail",
]

def _to_abs_path(path_str: str) -> Path:
    return resolve_input_path(path_str)

def _dense_frame_values(df: pd.DataFrame) -> list[int]:
	frame_values = pd.to_numeric(df["frame_idx"], errors="coerce").dropna().astype(int)
	if frame_values.empty:
		return []
	return list(range(int(frame_values.min()), int(frame_values.max()) + 1))

def _parse_keypoints(arg: str) -> list[str]:
	keypoints = [token.strip() for token in str(arg).split(",") if token.strip()]
	if not keypoints:
		raise ValueError("At least one interpolation keypoint must be provided")
	return keypoints


from pathlib import Path
import sys
import pandas as pd
from data_loader.io.csv_loader import read_pose_csv
from data_loader.io.schema import Schema
from data_loader.processing.deduplicate import deduplicate_frame_track
from data_loader.processing.interpolate import interpolate_keypoints, pad_frame_track_grid

def preprocess_sessions(args):
    manifest_path = _to_abs_path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f'Manifest not found: {manifest_path}')
    output_dir = _to_abs_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_df = pd.read_csv(manifest_path)
    if manifest_df.empty:
        print('Manifest is empty.')
        return 0
    schema = Schema(keypoints=KEYPOINTS)
    interp_keypoints = _parse_keypoints(args.keypoints)
    summary_rows: list[dict[str, object]] = []
    issue_rows: list[dict[str, object]] = []
    for row in manifest_df.to_dict(orient='records'):
        pose_rel = str(row.get('pose_path', '')).strip()
        meta = {'date': row.get('date', ''), 'condition': row.get('condition', ''), 'session_key': row.get('session_key', ''), 'subject_id': row.get('subject_id', ''), 'sex': row.get('sex', ''), 'phase': row.get('phase', ''), 'phase_detail': row.get('phase_detail', row.get('phase', '')), 'hab_type': row.get('hab_type', ''), 'trial_id': row.get('trial_id', pd.NA), 'session_n': row.get('session_n', pd.NA), 'social_side': row.get('social_side', ''), 'empty_side': row.get('empty_side', ''), 'familiar_side': row.get('familiar_side', ''), 'novel_side': row.get('novel_side', ''), 'heatmap_target_label': row.get('heatmap_target_label', ''), 'heatmap_target_side': row.get('heatmap_target_side', ''), 'mirror_lr_for_target_right': row.get('mirror_lr_for_target_right', pd.NA), 'layout_raw': row.get('layout_raw', ''), 'pose_path': pose_rel}
        if not pose_rel:
            issue_rows.append({**meta, 'issue': 'missing_pose_path', 'detail': ''})
            continue
        pose_path = _to_abs_path(pose_rel)
        if not pose_path.exists():
            issue_rows.append({**meta, 'issue': 'pose_file_not_found', 'detail': pose_rel})
            continue
        try:
            (df, _) = read_pose_csv(pose_path, schema)
        except Exception as exc:
            issue_rows.append({**meta, 'issue': 'read_pose_failed', 'detail': str(exc)})
            continue
        rows_in = len(df)
        if rows_in == 0:
            issue_rows.append({**meta, 'issue': 'empty_pose_csv', 'detail': ''})
            continue
        duplicate_mask = df.duplicated(['frame_idx', 'track'])
        duplicate_rows_removed = int(duplicate_mask.sum())
        if duplicate_rows_removed or args.dedup_policy:
            df = deduplicate_frame_track(df, policy=args.dedup_policy)
        rows_after_dedup = len(df)
        frame_values = _dense_frame_values(df)
        if not frame_values:
            issue_rows.append({**meta, 'issue': 'missing_frame_idx', 'detail': ''})
            continue
        tracks = sorted(df['track'].dropna().astype(str).unique().tolist())
        if not tracks:
            issue_rows.append({**meta, 'issue': 'missing_track', 'detail': ''})
            continue
        if args.no_pad:
            padded = df.sort_values(['frame_idx', 'track']).reset_index(drop=True)
        else:
            padded = pad_frame_track_grid(df, frames=frame_values, tracks=tracks)
        rows_after_pad = len(padded)
        padded_rows_added = rows_after_pad - rows_after_dedup
        try:
            (filled, summary) = interpolate_keypoints(padded, keypoints=interp_keypoints, method=args.method, max_gap=args.max_gap, smooth_window=args.smooth_window if args.smooth_window > 1 else None, score_threshold=args.score_threshold, kalman_q_pos=args.kalman_q_pos, kalman_q_vel=args.kalman_q_vel, kalman_r_base=args.kalman_r_base, kalman_max_predict=args.kalman_max_predict, kalman_gate_mahalanobis_sq=None if args.kalman_gate <= 0 else float(args.kalman_gate))
        except Exception as exc:
            issue_rows.append({**meta, 'issue': 'interpolation_failed', 'detail': str(exc)})
            continue
        date_dir = output_dir / str(row.get('date', 'unknown_date'))
        date_dir.mkdir(parents=True, exist_ok=True)
        out_file = date_dir / f'interp__{args.method}__{pose_path.name}'
        filled.to_csv(out_file, index=False)
        summary_rows.append({**meta, 'output_path': out_file.relative_to(ROOT).as_posix(), 'rows_in': rows_in, 'rows_after_dedup': rows_after_dedup, 'rows_after_pad': rows_after_pad, 'tracks': len(tracks), 'frame_min': frame_values[0], 'frame_max': frame_values[-1], 'dense_frame_count': len(frame_values), 'duplicate_rows_removed': duplicate_rows_removed, 'padded_rows_added': padded_rows_added, 'interpolated_keypoints': ','.join(interp_keypoints), 'Body_C_filled_x': summary.get('Body_C_filled_x', pd.NA), 'Body_C_filled_y': summary.get('Body_C_filled_y', pd.NA), 'Body_C_long_gaps': summary.get('Body_C_long_gaps', pd.NA), 'Body_C_remaining_nan_x': summary.get('Body_C_remaining_nan_x', pd.NA), 'Body_C_remaining_nan_y': summary.get('Body_C_remaining_nan_y', pd.NA), 'Nose_filled_x': summary.get('Nose_filled_x', pd.NA), 'Nose_filled_y': summary.get('Nose_filled_y', pd.NA), 'Nose_long_gaps': summary.get('Nose_long_gaps', pd.NA), 'Nose_remaining_nan_x': summary.get('Nose_remaining_nan_x', pd.NA), 'Nose_remaining_nan_y': summary.get('Nose_remaining_nan_y', pd.NA), 'method': summary['method']})
    summary_df = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    issues_df = pd.DataFrame(issue_rows, columns=ISSUE_COLUMNS)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(['date', 'subject_id', 'phase', 'phase_detail', 'trial_id', 'session_n', 'layout_raw'], na_position='last').reset_index(drop=True)
    if not issues_df.empty:
        issues_df = issues_df.sort_values(['date', 'subject_id', 'phase', 'phase_detail', 'trial_id', 'session_n', 'issue'], na_position='last').reset_index(drop=True)
    summary_path = output_dir / 'preprocess_summary.csv'
    issues_path = output_dir / 'preprocess_issues.csv'
    summary_df.to_csv(summary_path, index=False)
    issues_df.to_csv(issues_path, index=False)
    print(f'Sessions in manifest: {len(manifest_df)}')
    print(f'Preprocessed sessions: {len(summary_df)}')
    print(f"Sessions with preprocessing issues: {(issues_df['session_key'].nunique() if not issues_df.empty else 0)}")
    if not summary_df.empty:
        if 'Body_C_filled_x' in summary_df.columns:
            print(f"Total filled Body_C.x: {int(summary_df['Body_C_filled_x'].fillna(0).sum())}")
            print(f"Total filled Body_C.y: {int(summary_df['Body_C_filled_y'].fillna(0).sum())}")
        if 'Nose_filled_x' in summary_df.columns:
            print(f"Total filled Nose.x: {int(summary_df['Nose_filled_x'].fillna(0).sum())}")
            print(f"Total filled Nose.y: {int(summary_df['Nose_filled_y'].fillna(0).sum())}")
        print(f"Total padded rows added: {int(summary_df['padded_rows_added'].sum())}")
    print(f'Wrote {summary_path.relative_to(ROOT).as_posix()}')
    print(f'Wrote {issues_path.relative_to(ROOT).as_posix()}')
    return 0

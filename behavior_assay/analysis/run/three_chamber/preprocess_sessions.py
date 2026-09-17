from __future__ import annotations
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from analysis.core.three_chamber.preprocess_sessions import preprocess_sessions
import argparse
import pandas as pd
from data_loader.io.csv_loader import read_pose_csv
from data_loader.io.schema import Schema
from data_loader.processing.deduplicate import deduplicate_frame_track
from data_loader.processing.interpolate import interpolate_keypoints, pad_frame_track_grid
from analysis.core.three_chamber.preprocess_sessions import _dense_frame_values
from analysis.core.three_chamber.preprocess_sessions import _parse_keypoints
from analysis.core.three_chamber.preprocess_sessions import _to_abs_path
from analysis.core.three_chamber.preprocess_sessions import KEYPOINTS
from analysis.core.three_chamber.preprocess_sessions import DEFAULT_INTERP_KEYPOINTS
from analysis.core.three_chamber.preprocess_sessions import SUMMARY_COLUMNS
from analysis.core.three_chamber.preprocess_sessions import ISSUE_COLUMNS

def main(argv: list[str] | None=None) -> int:
    parser = argparse.ArgumentParser(description='Preprocess 3-chamber pose CSVs listed in the session manifest.')
    parser.add_argument('manifest', nargs='?', default='output/3chamber/manifest/session_manifest.csv', help='Manifest CSV produced by build_manifest.py')
    parser.add_argument('--output-dir', default='output/3chamber/preprocessed', help='Directory to write per-date preprocessed CSVs and summary tables')
    parser.add_argument('--dedup-policy', default='highest_instance_score', choices=['highest_instance_score', 'highest_kp_score_sum', 'first'], help='Policy for duplicate (frame_idx, track) rows')
    parser.add_argument('--no-pad', action='store_true', help='Skip dense frame padding before interpolation')
    parser.add_argument('--method', default='linear', choices=['linear', 'kalman'], help='Interpolation method')
    parser.add_argument('--keypoints', default=','.join(DEFAULT_INTERP_KEYPOINTS), help=f"Comma-separated keypoints to interpolate (default: {','.join(DEFAULT_INTERP_KEYPOINTS)})")
    parser.add_argument('--max-gap', type=int, default=0, help='Max gap length to interpolate for linear mode; 0 means fill all interior gaps')
    parser.add_argument('--smooth-window', type=int, default=0, help='Optional moving-average window after linear interpolation')
    parser.add_argument('--score-threshold', type=float, default=0.0, help='Kalman: ignore measurements with score below threshold')
    parser.add_argument('--kalman-q-pos', type=float, default=0.001, help='Kalman process noise for position')
    parser.add_argument('--kalman-q-vel', type=float, default=0.01, help='Kalman process noise for velocity')
    parser.add_argument('--kalman-r-base', type=float, default=0.01, help='Kalman base measurement noise')
    parser.add_argument('--kalman-max-predict', type=int, default=30, help='Kalman max consecutive prediction-only steps')
    parser.add_argument('--kalman-gate', type=float, default=0.0, help='Kalman Mahalanobis^2 gate threshold; 0 disables')
    args = parser.parse_args(argv)
    return preprocess_sessions(args)
if __name__ == '__main__':
    raise SystemExit(main())

"""Prepare tracking inputs or inspect/re-encode videos in a separate work folder."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import pandas as pd

from .tracking import (find_video_for_csv, list_csv_files, normalize_stem,
                       parse_drop_bodyparts, parse_expected_tracks, process_one_pair)
from .video import diagnose_one, rewrite_video, validate_output

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ["prepare", "videos"]:
        sub = commands.add_parser(command)
        sub.add_argument("--csv-dir", type=Path, default=PROJECT_ROOT.parent / "data/csv/Fig5D_FigS9AB/tracking")
        sub.add_argument("--video-dir", type=Path, default=PROJECT_ROOT.parent / "data/video/Fig5D_FigS9AB")
        sub.add_argument("--output", type=Path,
                         default=PROJECT_ROOT / "work" / ("prepared" if command == "prepare" else "video_check"))
        sub.add_argument("--recursive", action="store_true")
        sub.add_argument("--keep-original-names", action="store_true")
        sub.add_argument("--limit", type=int)
        sub.add_argument("--codec", default="XVID")
        if command == "prepare":
            sub.add_argument("--expected-tracks", type=parse_expected_tracks, default=(0, 1))
            sub.add_argument("--drop-bodyparts", type=parse_drop_bodyparts, default=("Neck",))
            sub.add_argument("--source-width", type=int, default=1920)
            sub.add_argument("--source-height", type=int, default=1080)
            sub.add_argument("--seek-validation", choices=["none", "quick", "deep"], default="deep")
            sub.add_argument("--dry-run", action="store_true",
                             help="Inspect the transformation; write only a report in the work folder")
        else:
            sub.add_argument("--rewrite", action="store_true",
                             help="Write corrected copies after checking the source timeline")
            sub.add_argument("--backend", choices=["opencv", "ffmpeg"], default="opencv")
            sub.add_argument("--ffmpeg", type=Path, help="FFmpeg executable; otherwise found on PATH")
    return parser.parse_args()


def main():
    args = parse_args()
    csv_dir, video_dir, output = args.csv_dir.resolve(), args.video_dir.resolve(), args.output.resolve()
    if not csv_dir.is_dir() or not video_dir.is_dir():
        raise FileNotFoundError("Both CSV and video input directories must exist")
    if (PROJECT_ROOT / "work").resolve() not in output.parents:
        raise ValueError("Output must be a subdirectory of direct_interaction/work")
    for source in [csv_dir, video_dir]:
        if output == source or source in output.parents or output in source.parents:
            raise ValueError("Output and input directories must be separate")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose a new output folder")
    files = list_csv_files(csv_dir, args.recursive)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        files = files[:args.limit]
    if not files:
        raise ValueError("No input CSV files found")
    ffmpeg = None
    if args.command == "videos" and args.rewrite and args.backend == "ffmpeg":
        ffmpeg = str(args.ffmpeg) if args.ffmpeg else shutil.which("ffmpeg")
        if not ffmpeg or not Path(ffmpeg).is_file():
            raise FileNotFoundError("Provide --ffmpeg or put FFmpeg on PATH")
    # Resolve every pair and name collision before producing any video or CSV.
    pairs, used = [], set()
    for csv_path in files:
        video_path = find_video_for_csv(csv_path, video_dir, args.recursive, args.keep_original_names)
        if video_path is None:
            raise FileNotFoundError(f"No video for {csv_path.name}")
        stem = csv_path.stem if args.keep_original_names else normalize_stem(csv_path.stem)
        if stem in used:
            raise ValueError(f"Output name collision: {stem}")
        used.add(stem)
        pairs.append((csv_path, video_path, stem))
    output.mkdir(parents=True, exist_ok=True)
    reports = []
    for csv_path, video_path, stem in pairs:
        try:
            if args.command == "prepare":
                row = process_one_pair(csv_path, video_path, output, args, stem)
            else:
                row = diagnose_one(csv_path, video_path)
                row["normalized_file"] = stem
                if args.rewrite:
                    allowed = {"OK_NO_DELETE", "OK_DELETE_THESE_SEEK_FRAMES", "SEEK_MISMATCH"}
                    if row["status"] not in allowed:
                        raise ValueError(f"Source timeline needs review: {row['status']}")
                    destination = output / "video" / f"{stem}.avi"
                    rewrite_video(video_path, destination, int(row["csv_frame_count"]), args.codec, ffmpeg)
                    validation = validate_output(destination, int(row["csv_frame_count"]))
                    row["input_status"] = row["status"]
                    row.update(validation)
                    row["output_video"] = str(destination)
                    row["status"] = "OK" if validation["validation_ok"] else "VALIDATION_FAILED"
        except Exception as exc:
            row = {"file": csv_path.name, "status": "ERROR", "error": str(exc)}
        reports.append(row)
        print(f"{csv_path.name}: {row['status']}", flush=True)
    pd.DataFrame(reports).to_csv(output / "report.csv", index=False)
    print(f"Report: {output / 'report.csv'}")
    return 0 if all(str(row["status"]).startswith(("OK", "DRY_RUN_OK")) for row in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())

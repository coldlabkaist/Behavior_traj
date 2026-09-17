# Data preparation and video correction

These tools prepare matched tracking/video inputs for SiMBA. Paper statistics and
figures are calculated from existing predictions by the separate
[analysis workflow](../analysis/README.md).

- `tracking.py`: match files, normalize names, convert coordinate units, remove selected body parts, and fill missing rows.
- `video.py`: compare sequential decoding with random frame access, diagnose frame alignment, and re-encode videos.
- `__main__.py`: input/output paths, execution options, and processing reports.

## Run

Run from `direct_interaction/`. The default inputs are
`../data/csv/Fig5D_FigS9AB/tracking` and `../data/video/Fig5D_FigS9AB`.
The tools were validated with Python 3.9.25 and [requirements.txt](requirements.txt).

To inspect planned conversions and save only a report:

```powershell
python -B -m preprocessing prepare --dry-run --output work/prepare_check
```

To prepare CSV/video pairs for a new dataset:

```powershell
python -B -m preprocessing prepare --csv-dir INPUT_CSV --video-dir INPUT_VIDEO --output work/prepared
```

Defaults select tracks 0 and 1, remove Neck, convert coordinates to video dimensions
when needed, apply linear interpolation within each track, and sequentially decode
and re-encode video using XVID. Scores for inserted rows are set to zero. Original
inputs are not modified.

Use `--drop-bodyparts ""` to retain all body parts and `--keep-original-names` to
preserve file names. Default name normalization removes the predict prefix, timestamp
suffix, and trailing `_cut`; a name collision stops execution.

To diagnose current video/tracking frame alignment:

```powershell
python -B -m preprocessing videos --output work/video_check
```

To create new video copies that preserve the sequential decoding timeline:

```powershell
python -B -m preprocessing videos --rewrite --output work/video_rewrite
```

OpenCV is the default encoding backend. Select the FFmpeg implementation with
`--backend ffmpeg`; if FFmpeg is not on PATH, specify `--ffmpeg EXE_PATH`.
A video/tracking frame-count mismatch is reported as an error, rather than corrected
by automatic trimming. Seek ranges in diagnostic reports describe mismatches;
they are not instructions to delete additional frames from sequential video.

Outputs are restricted to new directories under `direct_interaction/work/`.
`prepare` writes `csv/`, `video/`, and `report.csv`; video correction writes `video/`
and `report.csv`. Diagnostics and dry runs write reports only. Use `--limit 1` to
check a single pair. Errors or results requiring further review produce exit code 1.

Video diagnostics accept both long-form `track, frame_idx` CSV files and SiMBA pose
CSV files with three header rows. Coordinate preparation accepts long-form CSV inputs.

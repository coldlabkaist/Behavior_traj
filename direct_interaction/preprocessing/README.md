# Data preparation

Prepare tracking/video pairs for SiMBA. Use Python 3.9 and
[requirements.txt](requirements.txt); run from `direct_interaction/`.
Default inputs are `../data/csv/Fig5D_FigS9AB/tracking` and
`../data/video/Fig5D_FigS9AB`.

```powershell
python -m pip install -r preprocessing/requirements.txt
python -B -m preprocessing prepare --dry-run --output work/prepare_check
python -B -m preprocessing prepare --csv-dir INPUT_CSV --video-dir INPUT_VIDEO --output work/prepared
```

Defaults select tracks 0/1, remove Neck, normalize names and coordinate units, interpolate
tracking, and re-encode video. Original files are preserved. Use `--help` for options.

## Video alignment

```powershell
python -B -m preprocessing videos --output work/video_check
python -B -m preprocessing videos --rewrite --output work/video_rewrite
```

Diagnostics check frame alignment; `--rewrite` creates new copies using sequential
decoding. Frame-count mismatches are reported, not automatically trimmed.
Outputs must use new folders under `work/`. Use `--limit 1` to check one pair.
For paper statistics and figures, use the [analysis workflow](../analysis/README.md).

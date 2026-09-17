# Preprocessing

Tracking quality summaries and Body_C interpolation. Run from `behavior_assay/`:

```powershell
python -B -m preprocessing.report_data --help
python -B -m preprocessing.interpolate_body_c --help
```

`report_data` summarizes missing observations and duplicate frames.
`interpolate_body_c` handles duplicates, missing frame rows, and Body_C interpolation.
For paper figures and statistics, use the [analysis runner](../analysis/run/README.md).

# Preprocessing

Tools for tracking-CSV quality summaries and Body_C interpolation.

- `report_data.py`: summarize missing observations and duplicate frames by file and animal.
- `interpolate_body_c.py`: handle duplicates, fill missing frame rows, and interpolate Body_C.

Run from the `behavior_assay` directory:

```powershell
python -B -m preprocessing.report_data --help
python -B -m preprocessing.interpolate_body_c --help
```

For default paper figure and statistical reproduction, use the
[analysis runner](../analysis/run/README.md).

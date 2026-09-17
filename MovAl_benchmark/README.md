# MovAl tracking benchmark

Tracking miss, identity switch, jitter, and RMSE analyses for Fig2B–E, FigS1, and Tables S1–S2.

## Setup and run

Use Python 3.12. Run from the repository root:

```powershell
python -m pip install -r MovAl_benchmark/analysis/requirements.txt
python -B -m MovAl_benchmark.analysis.run.reproduce
python -B -m MovAl_benchmark.analysis.run.verify
```

## Inputs

Supply these files separately; paths below are relative to `MovAl_benchmark/`:

- Fig2B: `Fig2B/data/tracking_miss_long.csv`.
- Fig2C: `Fig2C/data/Mode_comp.xlsx`.
- Jitter: model folders under `../data/csv/Fig2BCD_FigS1/`.
- RMSE: ground-truth and prediction CSVs under `../data/csv/Fig2E/`.
- Verification: reference results in `Fig2B/`, `Fig2C/`, `Fig2D_FigS1_TableS2/`, and `Fig2E/`.

Outputs are written to `MovAl_benchmark/work/reproduced`. Use `--output-root <path>`
with both commands to change the destination. Select a panel with `--figure Fig2B`,
`Fig2C`, `Fig2D`, `FigS1`, or `Fig2E`. Full verification requires all panel outputs.
The verifier returns exit code 1 if a comparison fails.

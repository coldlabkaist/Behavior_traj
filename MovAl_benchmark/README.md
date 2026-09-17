# MovAl tracking benchmark

> This repository contains code only. Supply the raw data and reference results
> separately, using the paths below, before running reproduction or verification.

Tracking-miss, identity-switch, jitter, and RMSE analyses for Fig2B–E, FigS1, and Tables S1–S2.

## Organization

- `analysis/core`: input loading, aggregation, paired statistics, FFT, and coordinate RMSE.
- `analysis/plots`: shared styles and PNG/SVG generation.
- `analysis/run`: reproduction and comparison with reference results.
- `Fig2B`, `Fig2C`, `Fig2D_FigS1_TableS2`, `Fig2E`: separate panel packages containing figure, stat, and data folders.
- `../data/csv/Fig2BCD_FigS1/{Raw_*,SegCont_*}`: raw tracking CSV files by method.
- `reference`: local figure-assembly reference images, excluded from Git.

`work` contains regenerated outputs and local investigation records; it is not a required input.

## Run

Activate a Python 3.12 environment and run from the `Behavior_traj` repository root:

```powershell
python -m pip install -r MovAl_benchmark/analysis/requirements.txt
python -B -m MovAl_benchmark.analysis.run.reproduce
python -B -m MovAl_benchmark.analysis.run.verify
```

The default output is `MovAl_benchmark/work/reproduced`. To use another location,
pass `--output-root <path>` to both commands. Select a panel with `--figure Fig2B`,
`Fig2C`, `Fig2D`, `FigS1`, or `Fig2E`. Full verification requires outputs for all panels.

## Inputs and calculations

### Fig2B and Table S1

The input is `Fig2B/data/tracking_miss_long.csv`. Of the Raw, Seg, and Seg-Cont input
conditions, the paper uses Raw and Seg-Cont. For each keypoint, first average the
three animals within each video. Use the 17 video-level observations to calculate
mean ± SEM, a 2 × 3 repeated-measures ANOVA, and 15 paired t-tests with Holm correction.
Table S1 statistics are stored in `Fig2B/stat`. The current reference summary is
`plot_summary.csv`; `saved_plot_summary.csv` is a historical reference only.

### Fig2C

Use the original experimental values in `Fig2C/data/Mode_comp.xlsx`, without replacing
them with the distance-analysis recount. Calculate a 2 × 3 repeated-measures ANOVA
and four paired t-tests against MovAl within the corresponding input conditions,
with Benjamini–Hochberg (BH) correction. Figures are saved as
`figure/identity_switch_frequency.png` and `.svg` within the panel package.

### Fig2D, FigS1, and Table S2

Use 68 CSV files from the Raw/SegCont SLEAP and MovAl directories under
`../data/csv/Fig2BCD_FigS1`. For frames 0–8999, compute consecutive position differences
for each of three tracks. Apply FFT to the mean-centered log10 values of valid,
positive displacements. Sum power in the Low [0, 0.05), Mid [0.05, 0.15), and High
[0.15, 0.5) frequency bands.

Outputs contain 1,836 individual-level power rows, 612 video-level power rows,
27 ANOVA rows, and 54 paired-comparison rows. Statistics use video-level means and
retain the original plot aggregation definitions. Band-power plots for `Body_C`,
`Nose`, and `Tail`, together with example displacement traces, are also generated.

### Fig2E

Use five ground-truth and 30 prediction CSV files under `../data/csv/Fig2E`. Adjust
frame indices, treat zero coordinates as missing, and normalize by each model's
image dimensions. Pool valid prediction–ground-truth coordinate pairs across the
five clips, average squared Euclidean distances for each keypoint, and take the square root.
DLC animal labels are matched by name; animals are not reassigned to minimize error.
Intermediate MSE values are not rounded; only the displayed RMSE is rounded to four
decimal places. No inferential statistics are calculated for this panel.

## Verification

All 18 reproduction/comparison checks passed in the validated environment. Fig2B/C/E
PNGs matched the reference figures pixel for pixel. Jitter verification compares
saved values and calculation definitions; it does not reproduce the layout of the
historical figure assembled from six conditions. The verifier returns exit code 1
when a comparison fails.

Manuscript edits are not automated. Local notes under `work/manuscript_update` and
`work/before_current_data_update` are not required for reproduction. Preserve supplied
reference figures, statistics, and inputs separately from regenerated outputs.

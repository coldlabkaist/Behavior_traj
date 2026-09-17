# Direct interaction paper reproduction

Panels: **Fig5D (Social), FigS9A (Attentive), and FigS9B (Prosocial)**.
Place separately supplied reference results in `output/final`; they are not included
in this repository.

## Run

Use Python 3.12.4 and the versions in [requirements.txt](requirements.txt). Reference
PNGs were generated with Matplotlib 3.8.4; other versions may render text and lines
differently. The SiMBA classifier environment is separate from this analysis environment.

Run from `direct_interaction/analysis/`:

```powershell
python -B -m run.reproduce
```

The runner calculates individual-level values from prediction probabilities in
`../../data/csv/Fig5D_FigS9AB/predictions/{modelA,modelB}`, then generates statistics
and figures. The default output is `work/direct_interaction/`. If it already exists,
use `--output-dir work/new_run`. Inputs and reference results are not overwritten.

To regenerate statistics and figures from the shared individual-level table:

```powershell
python -B -m run.reproduce --input-csv output/final/Fig5D/data/individual_behavior_values.csv --output-dir work/from_final
```

## Calculation settings

- Models A/B, probability threshold ≥0.5, 124 B6 observations from 56 mice, 30 fps.
- Attentive: Approach, Facing, Following.
- Prosocial: Nose-Head, Nose-Body, Nose-Anogenital, Mounting.
- Social: the union of all seven behaviors, rather than the sum of category durations or bout counts.
- Within each category, combine positive frames and merge gaps shorter than 15 frames.
  Include only merged bouts lasting at least 15 frames in duration and bout counts.
- Compare Control and VPA at each week using two-sided Welch t-tests. Apply BH correction
  within each category across six comparisons: three weeks × two metrics.
- Figures combine both sexes and show mean ± SEM and individual trajectories, with
  mean-focused axes, a 9.4 × 4.2 inch canvas, 300 dpi, and PNG/SVG exports.

## Modules

- [core/behavior.py](core/behavior.py): combine prediction probabilities, form category unions, calculate bouts, and construct individual-level tables.
- [core/metadata.py](core/metadata.py): map file names to animal, condition, and age.
- [core/statistics.py](core/statistics.py): Welch tests, BH correction, SEM, and mouse counts by observed week.
- [plots/trajectory.py](plots/trajectory.py): individual trajectories and mean/SEM plots for the three categories.
- [plots/style.py](plots/style.py): shared colors, fonts, and PNG/SVG saving.
- [run/reproduce.py](run/reproduce.py): paper defaults, execution options, and output management.
- [run/verify.py](run/verify.py): compare regenerated data, statistics, and PNGs with reference results.

`core` and `plots` contain reusable functions; `run` provides the command-line entry points.
`../../data/csv/Fig5D_FigS9AB/predictions/` contains prediction inputs, `output/final/`
contains reference results, and `work/` contains regenerated outputs.

The canonical source collection uses repository-root `data/video/Fig5D_FigS9AB` and
`data/csv/Fig5D_FigS9AB/{tracking,predictions}`. Historical CSV files from before
preprocessing are not required by this reproduction workflow. Only the seven paper
behaviors are read from prediction CSVs; individual values are calculated without
writing intermediate CSV files.

To verify a run generated from the shared table:

```powershell
python -B -m run.verify --output-dir work/from_final
```

All checks, including PNG comparisons, pass in the matching validated environment.

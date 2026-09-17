# Direct interaction analysis

Run from `direct_interaction/analysis/` after completing the [setup](../README.md).

```powershell
python -B -m run.reproduce
python -B -m run.verify
```

Use `--output-dir work/new_run` if the default output already exists.
To regenerate from the separately supplied individual-level table:

```powershell
python -B -m run.reproduce --input-csv output/final/Fig5D/data/individual_behavior_values.csv --output-dir work/from_final
python -B -m run.verify --output-dir work/from_final
```

## Analysis defaults

- B6, models A/B, probability threshold ≥0.5, 30 fps.
- Attentive: Approach, Facing, Following.
- Prosocial: Nose-Head, Nose-Body, Nose-Anogenital, Mounting.
- Social: the union of all seven behaviors.
- Merge negative gaps shorter than 15 frames; retain merged bouts of at least 15 frames.
- Weekly Control/VPA comparisons: two-sided Welch t-tests, with BH correction across
  six comparisons per category (three weeks × two metrics).

Figures show individual trajectories and mean ± SEM. Reference PNGs use Matplotlib
3.8.4; other rendering environments may differ. Inputs and reference results are preserved.

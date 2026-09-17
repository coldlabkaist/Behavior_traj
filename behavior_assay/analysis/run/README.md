# Behavior assay run options

Run from the repository root after installing the
[project requirements](../../README.md) and supplying `behavior_assay/output/Final/`.

```powershell
python behavior_assay/analysis/run/reproduce.py --panels Fig5F Fig5H
python behavior_assay/analysis/run/reproduce.py --mode stats
python behavior_assay/analysis/run/reproduce.py --panels FigS10A FigS10B --output-dir behavior_assay/output/oft_check
```

- `--mode all` (default): recalculate and verify statistics, then generate figures.
- `--mode figures`: generate figures after checking their statistics.
- `--final-dir`: select another input package.
- `--output-dir`: select another output location.

A statistical mismatch stops the run. `verification.json` records the comparisons.
The Fig5E detection example is copied from the supplied figure; occupancy maps are regenerated.

## Rebuild three-chamber density maps

Supply raw tracking and ROI pins under `data/csv/Fig5BC_FigS8ABCD`, together with the
Final input metadata, then run:

```powershell
python behavior_assay/analysis/run/prepare_density.py
```

This rebuilds the numerical density maps. The default reproduction runner uses the
saved maps and does not require this step on every run.

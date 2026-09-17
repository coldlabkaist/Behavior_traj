# Execution guide

Run all commands from `Homecage/`. See the [project README](../../README.md) for
installation, DAE training, and Movie S3, and the
[R environment guide](../core/statistics/README.md) for R packages and versions.
All data, fitted models, and reference results mentioned here are supplied separately.

## Default reproduction

```powershell
python -B -m analysis.run.reproduce
python -B -m analysis.run.figures
```

Inputs are read from `analysis/output/Final` and DAE checkpoints from `checkpoints`.
Numerical checks are written to `analysis/work/paper`; PNG/SVG figures are written to
`analysis/work/paper_figures`. Select another destination with `--output`. Final files
are not overwritten.

- `reproduce` checks BOI, GMM posteriors, transitions, occupancy, age effects in the
  soft joint distribution, S3 training history, S4 factor analysis, S5 saved repeat
  results, S6 entropy, Fig5G/I Spearman correlations, and saved R statistics for Fig4EFG/S7.
- `figures` draws the paper panels from saved final inputs, excluding Illustrator assembly.
- Set the R executable with `--rscript`; the default is `C:/Program Files/R/R-4.4.1/bin/Rscript.exe`.

To generate selected panels:

```powershell
python -B -m analysis.run.figures --panels Fig4EFG FigS7AB
```

## Refitting and raw-data latent extraction

```powershell
python -B -m analysis.run.reproduce --refit-projection
python -B -m analysis.run.reproduce --refit-s5 --output analysis/work/s5_refit
python -B -m analysis.run.extract_latent
python -B -m analysis.run.probe
Rscript analysis/run/fit_models.R
Rscript analysis/run/refit_sex_df.R
```

- `--refit-projection` retrains the MLP from Final 64-dimensional latent data using
  12 Control cages, then checks the three-dimensional coordinates and fold history.
  BOI inputs pass through a CSV round trip using `%.10g` formatting.
- `--refit-s5` repeats GMM fitting from the Final codebook and shifted coordinates.
  It does not regenerate shifted latent data from raw inputs.
- `extract_latent` uses raw data and the saved checkpoint with batch size 256 and
  four PyTorch threads, then checks z, relation values, and metadata against Final.
  Use a new `--output` directory if a previous output exists. See the
  [latent-extraction limitation](../../README.md#latent-extraction) before using this command.
- `probe` retrains the five-fold nonlinear probe for S3B.
- `fit_models.R` refits the heteroscedastic BOI models and compares them with Final RDS/CSV files.
- `refit_sex_df.R` checks approximate degrees of freedom for sex endpoints with seed 20260914 and extra.iter 200.

These refitting commands are not part of the default verification run. Numerical
agreement and pixel-level figure agreement are separate checks. Refitting never
automatically replaces Final files.

## Panel-to-module mapping

| Panel | Calculation modules | Plot modules |
|---|---|---|
| Fig4B | latent_cache, projection | distribution |
| Fig4C/D | motifs, transitions | motifs, transitions |
| Fig4EFG | boi, statistics | boi |
| Fig5G/I | rank_tests | associations |
| FigS3AB | nonlinear_probe | validation |
| FigS4AB | factor_analysis | validation |
| FigS5AB | motif_stability, temporal_shift | stability |
| FigS6AB | transitions | transitions |
| FigS7AB | statistics | boi |

Calculation modules live in `analysis/core`; plot modules live in `analysis/plots`.
`core/archive.py` reads and checks Final inputs, `core/verification.py` performs
numerical comparisons, and `plots/panels.py` selects panels. Shared colors and fonts
are defined in `plots/style.py`; PNG/SVG export is handled by `plots/export.py`.

S5 requires the codebook, compact shifted coordinates, and cage-level repeat results.
Keep these shared inputs together with the panel-specific Final `data` folders.
Historical paths in manifests record provenance.

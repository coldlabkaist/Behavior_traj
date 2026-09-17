# Additional Homecage commands

Run from `Homecage/` after supplying the inputs listed in the
[project README](../../README.md).

## Selected figures

```powershell
python -B -m analysis.run.figures --panels Fig4EFG FigS7AB
```

## Training and refitting

```powershell
python train.py --config cfg/train_config.yaml
python -B -m analysis.run.reproduce --refit-projection
python -B -m analysis.run.reproduce --refit-s5 --output analysis/work/s5_refit
python -B -m analysis.run.probe
Rscript analysis/run/fit_models.R
Rscript analysis/run/refit_sex_df.R
```

These commands train the DAE or refit the MLP projection, S5 GMMs, S3B probe, and R models.
S5 refitting requires saved shifted coordinates; it does not recreate them from raw data.

## Raw-data latent extraction

```powershell
python -B -m analysis.run.extract_latent --device cpu
```

Extraction compares its output with Final. See the
[known limitation](../../README.md#latent-extraction). Use a new `--output` directory
when rerunning. Refitted or extracted values do not automatically replace Final inputs.

## Movie S3

```powershell
python -B -m analysis.run.movie_s3 --video-root PATH_TO_AVI_DIRECTORY
```

Requires original AVI files, Final pose data, and FFmpeg/ffprobe on PATH. Add
`--check-only` to validate inputs. The default output is `analysis/work/movie_s3/Movie_S3.mp4`.

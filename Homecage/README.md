# Homecage analysis

> This repository contains code only. Raw data, Final results, and trained models
> must be supplied separately and placed at the paths below.

Run all commands from `Behavior_traj/Homecage/`.

## Environment

The reference environment uses Python 3.9, PyTorch 2.1.2, and R 4.4.1.

```powershell
cd Homecage
python -m pip install -r requirements.txt
Rscript analysis/run/install_r_packages.R
```

See the [R statistics guide](analysis/core/statistics/README.md) for package versions.
If `Rscript` is not on PATH, use its full executable path.

## Required inputs

| Task | Inputs |
|---|---|
| Reproduce paper values and figures | Complete `analysis/output/Final/` package and `checkpoints/best_clean.pth` |
| Extract latent representations from raw data | `../data/csv/Fig4B-G_FigS3-S7/{cont,experiments,roi}/`, checkpoint, `cfg/train_config.yaml`, and `cfg/animal.yaml` |
| Train the DAE | `../data/csv/Fig4B-G_FigS3-S7/{reference,roi}/`, `cfg/train_config.yaml`, and `cfg/animal.yaml` |
| Generate Movie S3 | Original AVI files, raw CSV files, Final pose data, and `analysis/resources/movie_s3.json` |

<a id="reproduction"></a>
## Reproduce paper values and figures

```powershell
python -B -m analysis.run.reproduce
python -B -m analysis.run.figures
```

Numerical verification outputs are written to `analysis/work/paper/`; PNG and SVG
figures are written to `analysis/work/paper_figures/`. Final inputs are preserved.
Final Illustrator assembly is not included. Set the R executable with
`reproduce --rscript "path/to/Rscript.exe"` if necessary.

The default commands use saved latent arrays and fitted models. See the
[detailed execution guide](analysis/run/README.md) for panel selection and commands
to refit the MLP, S5 models, S3 probe, and R models.

<a id="latent-extraction"></a>
## Extract latent representations from raw data

```powershell
python -B -m analysis.run.extract_latent --device cpu
python -B -m analysis.run.reproduce --refit-projection
```

Extraction writes to `analysis/work/latent_extraction/` and compares the result with
Final latent data. `--refit-projection` retrains the MLP using Final latent data and
compares the resulting three-dimensional coordinates with the reference values.
For another extraction run, select an empty location with
`--output analysis/work/new_extraction`.

In the current environment, raw-data extraction reproduces the Final metadata but
does not exactly reproduce the latent values (maximum absolute difference approximately
0.00372). Use the saved Final latent data for downstream paper reproduction.

The implementation specifies the following reproduction settings:

- CPU, PyTorch 2.1.2, four threads, inference batch size 256.
- Control followed by VPA, with fixed file/window order, shuffle=False, and augmentation=False.
- Non-overlapping 30-frame windows and the configured ROI, quality-control, and normalization rules.
- MLP seed 42, at most 300 Control windows per cage-week, 12 folds, and the existing early-stopping rule.
- BOI coordinates saved to CSV with `%.10g` formatting and then read back.

## Train the DAE

```powershell
python train.py --config cfg/train_config.yaml
```

Use a single process. Training outputs follow `training.checkpoint_dir` in the configuration.

## Movie S3

```powershell
python -B -m analysis.run.movie_s3 --video-root D:/data_vid --video-root D:/Moval_proj/hc_more/raw_videos
```

Replace the example `--video-root` paths with your original AVI directories. Multiple
roots are searched in the specified order. FFmpeg and ffprobe must be on PATH.
Arial is used when available, with a fallback font otherwise. Add `--check-only` to
check selected scenes and input paths without generating the movie.

The [movie code](analysis/plots/movie_s3.py) assembles the 40 scenes in the
[selection configuration](analysis/resources/movie_s3.json) into ten sets. The default
output is `analysis/work/movie_s3/Movie_S3.mp4`: 30 seconds, 10 fps, 3840 × 1200 pixels.
If the output already exists, choose a different `--output` path.

## Configuration and data loading

Use `cfg/train_config.yaml` together with `cfg/animal.yaml`. New training runs follow
`training.checkpoint_dir`; paper reproduction uses `checkpoints/best_clean.pth`.
`datasets/pose_io.py` reads CSV and ROI inputs, `preprocessing.py` handles interpolation
and quality control, and `dataset.py` constructs 30-frame training/inference windows.
Import the public classes with `from datasets import PoseReader, MousePoseDataset`.

The configuration/checkpoint paths `data/cont`, `data/experiments`, `data/reference`,
and `data/roi` are stable source identifiers. `datasets/paths.py` resolves them to the
root raw-data folder, so the saved Final metadata does not need to be modified.

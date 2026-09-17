# MovAl behavior analyses

Analysis, plotting, preprocessing, and training code for the MovAl behavioral study.
The repository contains four projects, with shared conventions for calculations,
plotting, and command-line entry points.

**This repository contains code, configuration, and documentation only.** Raw data,
analysis tables, statistics, figures, latent arrays, and trained models are supplied
separately. Place the required files at the documented paths before running an analysis.

## Projects

| Project | Analysis | Paper panels |
|---|---|---|
| [Homecage](Homecage/README.md) | DAE latent representations, motifs, and BOI | Fig4B–G, Fig5G/I, FigS3–S7 |
| [MovAl_benchmark](MovAl_benchmark/README.md) | Tracking miss, identity switch, jitter, and RMSE | Fig2B–E, FigS1, Tables S1–S2 |
| [direct_interaction](direct_interaction/README.md) | Direct interaction from SiMBA predictions | Fig5D, FigS9A–B |
| [behavior_assay](behavior_assay/README.md) | Three-chamber, mother–pup, and open-field assays | Fig5B/C/E/F/H, FigS8, FigS10 |

## Organization and execution

Calculations are organized in `core`, plotting in `plots`, and command-line entry
points in `run`. Each project README specifies the working directory, environment,
required inputs, and supported reproduction steps. Final Illustrator assembly and
manuscript editing are outside the automated workflow.

Homecage and behavior_assay were validated with Python 3.9; MovAl_benchmark and the
direct_interaction analysis use Python 3.12. Use each project's requirements in a
separate environment. Homecage statistical verification also requires R. The direct
interaction preprocessing tools have their own environment requirements.

## Separate data packages

Place raw inputs under `data/csv/<panel group>` at the repository root. Videos, when
supplied, belong under `data/video/<panel group>`. Panels that share inputs use one
folder to avoid duplicate copies.

| Panel group | Inputs |
|---|---|
| `Fig2BCD_FigS1` | Benchmark tracking by method and input condition |
| `Fig2E` | Ground-truth and predicted coordinates for RMSE |
| `Fig4B-G_FigS3-S7` | Homecage Control/VPA tracking, DAE reference data, and ROI definitions |
| `Fig5BC_FigS8ABCD` | Three-chamber tracking and ROI definitions |
| `Fig5D_FigS9AB` | Direct interaction tracking, SiMBA predictions, and original videos |
| `Fig5EFH` | Mother–pup tracking |
| `FigS10AB` | Open-field tracking |

The local source collection contains 1,913 CSV files and 151 direct interaction
videos. Its `data/manifest.csv` records file sizes, SHA256 hashes, and original
locations. These files are not included in this repository. Original videos and
SiMBA projects/models are available on request.

Some commands also require curated analysis inputs, saved latent arrays, fitted
models, checkpoints, or reference results for verification. Restore those separate
files to the following locations:

- `Homecage/analysis/output/Final/` and `Homecage/checkpoints/`
- `MovAl_benchmark/Fig2B/`, `MovAl_benchmark/Fig2C/`,
  `MovAl_benchmark/Fig2D_FigS1_TableS2/`, and `MovAl_benchmark/Fig2E/`
- `direct_interaction/analysis/output/final/`
- `behavior_assay/output/Final/`

Panel packages use `figure`, `stat`, and `data` subfolders. Table S1 tracking-miss
statistics belong in `MovAl_benchmark/Fig2B/stat`. Historical paths in saved manifests
and checkpoints act as source identifiers; the loaders resolve them to the current
raw-data locations where needed.

Cloning this repository alone does not provide all inputs needed to reproduce the
paper. See the project READMEs for the required packages and paths. New runs write
to local `work` or `reproduced` folders, leaving the supplied reference results intact.

## Version control

All four projects are ordinary subfolders of one Git repository. The root
`.gitignore` excludes raw data, results, models, environments, and temporary outputs.
The published history contains code, configuration, and documentation only.
Git LFS is not required for this repository.

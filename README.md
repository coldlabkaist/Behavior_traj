# MovAl behavior analyses

Analysis and plotting code for the MovAl behavioral study.

| Project | Analysis | Paper panels |
|---|---|---|
| [Homecage](Homecage/README.md) | DAE latent representations, motifs, and BOI | Fig4B–G, Fig5G/I, FigS3–S7 |
| [MovAl_benchmark](MovAl_benchmark/README.md) | Tracking miss, identity switch, jitter, and RMSE | Fig2B–E, FigS1, Tables S1–S2 |
| [direct_interaction](direct_interaction/README.md) | Direct interaction from SiMBA predictions | Fig5D, FigS9A–B |
| [behavior_assay](behavior_assay/README.md) | Three-chamber, mother–pup, and open-field assays | Fig5B/C/E/F/H, FigS8, FigS10 |

## Getting started

Follow each project's README for installation, input paths, and commands. Use separate
environments: Python 3.9 for Homecage and behavior_assay, and Python 3.12 for the benchmark
and direct interaction analyses. Homecage statistics also require R.

Calculations, plotting, and command-line entry points are organized in `core`, `plots`,
and `run`, respectively.

## Data

This repository contains code and configuration only. Data, results, and trained models
are supplied separately and must be placed at the paths listed in each project README.
Raw CSV files belong under `data/csv/<panel group>` and videos under
`data/video/<panel group>`. Original videos and SiMBA models are available on request.

## License

Code in this repository is released under the [MIT License](LICENSE).

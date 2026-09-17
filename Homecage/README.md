# Homecage

DAE latent, motif, and BOI analyses for Fig4B–G, Fig5G/I, and FigS3–S7.

## Setup

Use Python 3.9, PyTorch 2.1.2, and R 4.4.1. From the repository root:

```powershell
cd Homecage
python -m pip install -r requirements.txt
Rscript analysis/run/install_r_packages.R
```

## Inputs

Data and models are supplied separately. Paths below are relative to `Homecage/`:

- Paper reproduction: `analysis/output/Final/` and `checkpoints/best_clean.pth`.
- Raw tracking and ROI: `../data/csv/Fig4B-G_FigS3-S7/{cont,experiments,roi}/`.
- DAE training data: `../data/csv/Fig4B-G_FigS3-S7/reference/`.

<a id="reproduction"></a>
## Run

```powershell
python -B -m analysis.run.reproduce
python -B -m analysis.run.figures
```

Outputs are written to `analysis/work/paper/` and `analysis/work/paper_figures/`.
Use `--rscript "path/to/Rscript.exe"` with `reproduce` if R is installed elsewhere.
Final inputs are preserved; final figure assembly is not automated.

<a id="latent-extraction"></a>
Use saved Final latent data for paper reproduction. Raw-data extraction currently
matches the metadata but not the latent values exactly.

See [additional commands](analysis/run/README.md) for training, refitting, and Movie S3,
and [R dependencies](analysis/core/statistics/README.md) for the statistical environment.

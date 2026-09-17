# Behavior assays

> This repository contains code only. Raw data and Final results must be supplied
> separately and placed at the documented paths before running the analyses.

Paper analyses for three-chamber, mother–pup, and open-field assays.

| Assay | Panels |
|---|---|
| Three-chamber | Fig5B/C, FigS8A–D |
| Mother–pup | Fig5E/F/H |
| Open field | FigS10A/B |

- `../data/csv/Fig5BC_FigS8ABCD`, `Fig5EFH`, `FigS10AB`: raw tracking and ROI inputs by assay.
- `data_loader`, `preprocessing`: CSV loading and preprocessing.
- `analysis/core`, `analysis/plots`, `analysis/run`: assay calculations, plotting, and commands.
- `analysis/tools`: ROI and contact-review GUIs.
- `output/Final`: separately supplied panel figures, statistics, data, and shared inputs.

Activate a Python 3.9 environment and run from the repository root:

```powershell
python -m pip install -r behavior_assay/requirements.txt
python -B behavior_assay/analysis/run/reproduce.py
```

Inputs are read from Final; regenerated outputs are written to `output/reproduced`
within this project. The supported workflow is **final individual/cage-level data
and numerical maps → statistics and figures**. It does not recalculate every metric
from raw tracking in one command. Regenerated outputs may be removed after comparison.

See the [execution guide](analysis/run/README.md) for inputs and options.
The Fig5E detection example uses an image from the external figure-editing source.

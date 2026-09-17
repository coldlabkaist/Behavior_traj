# Behavior assays

Three-chamber (Fig5B/C, FigS8A–D), mother–pup (Fig5E/F/H), and open-field (FigS10A/B) analyses.

## Setup and run

Use Python 3.9. Run from the repository root:

```powershell
python -m pip install -r behavior_assay/requirements.txt
python -B behavior_assay/analysis/run/reproduce.py
```

## Inputs and outputs

Supply `behavior_assay/output/Final/` separately. The default runner regenerates
statistics and figures from final individual/cage-level values and numerical maps;
it does not recalculate every metric from raw tracking.

Raw tracking and ROI inputs belong under repository-root `data/csv/` in
`Fig5BC_FigS8ABCD`, `Fig5EFH`, and `FigS10AB`.

Outputs are saved to `behavior_assay/output/reproduced`; Final inputs are preserved.
See [run options](analysis/run/README.md) and [preprocessing tools](preprocessing/README.md).

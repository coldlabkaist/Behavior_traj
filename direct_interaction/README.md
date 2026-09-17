# Direct interaction

SiMBA prediction-based analyses for Fig5D (Social) and FigS9A/B (Attentive/Prosocial).

## Setup and run

Use Python 3.12. From the repository root:

```powershell
cd direct_interaction/analysis
python -m pip install -r requirements.txt
python -B -m run.reproduce
python -B -m run.verify
```

## Inputs and outputs

Supply these inputs separately:

- SiMBA predictions: `data/csv/Fig5D_FigS9AB/predictions/{modelA,modelB}` at the repository root.
- Reference results: `direct_interaction/analysis/output/final/`.

Outputs are saved to `direct_interaction/analysis/work/direct_interaction`.
Original videos and SiMBA models are available on request.

See [analysis options](analysis/README.md) and [data preparation](preprocessing/README.md).

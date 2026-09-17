# Direct interaction

> This repository contains code only. Raw data, Final results, and SiMBA models
> are supplied separately and must be placed at the documented paths.

Calculate direct-interaction metrics from SiMBA prediction probabilities for Fig5D
and FigS9A–B.

- `../data/csv/Fig5D_FigS9AB/tracking`: tracking CSV files.
- `../data/csv/Fig5D_FigS9AB/predictions/{modelA,modelB}`: SiMBA prediction CSV inputs.
- `../data/video/Fig5D_FigS9AB`: original videos, available on request.
- `modelA`, `modelB`: SiMBA projects/models, available on request.
- `analysis/core`, `analysis/plots`, `analysis/run`: calculations, plotting, and commands.
- `analysis/output/final`: separately supplied reference data, statistics, and PNG/SVG figures.
- `preprocessing`: tracking preparation and video correction tools.

See the [analysis guide](analysis/README.md) for the environment and execution commands,
and the [preprocessing guide](preprocessing/README.md) for data preparation. The SiMBA
classifier-training environment is separate from the analysis environment.

Starting from the repository root:

```powershell
cd direct_interaction/analysis
python -m pip install -r requirements.txt
python -B -m run.reproduce
python -B -m run.verify
```

Regenerated results are saved to `analysis/work/direct_interaction` within this project.
If outputs already exist, select another destination with `--output-dir work/new_run`.
Temporary outputs may be removed after verification.

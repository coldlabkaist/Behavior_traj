# Run from the project root: Rscript analysis/run/fit_models.R [output]
source('analysis/run/runtime.R')
out <- prepare_output('analysis/work/model_refit', work_only=TRUE)
source('analysis/core/statistics/fit_models.R')
refit_models(out)

# Run from the project root: Rscript analysis/run/inference.R [output]
source('analysis/run/runtime.R')
out <- prepare_output('analysis/work/inference', work_only=FALSE)
source('analysis/core/statistics/inference.R')
verify_frozen_inference(out)

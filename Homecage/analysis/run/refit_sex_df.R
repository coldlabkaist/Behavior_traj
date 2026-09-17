# Run from the project root: Rscript analysis/run/refit_sex_df.R [output]
source('analysis/run/runtime.R')
out <- prepare_output('analysis/work/sex_df_refit', work_only=TRUE)
source('analysis/core/statistics/refit_sex_df.R')
refit_sex_df(out)

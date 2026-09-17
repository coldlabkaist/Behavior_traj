# R statistical modules

These files define functions; sourcing them does not run an analysis.

| File | Function | Purpose |
|---|---|---|
| `inference.R` | `verify_frozen_inference(out)` | Inference from saved BOI and sex models, compared with Final tables |
| `fit_models.R` | `refit_models(out)` | Refit BOI models with heterogeneous residual variances |
| `refit_sex_df.R` | `refit_sex_df(out)` | Check approximate degrees of freedom for sex endpoints |

Use the R files with the same names under `analysis/run` as command-line entry points.
They validate arguments and output locations, set library paths, and call the core
functions. Run them from `Homecage/`. The local `.r-library/` is searched first,
followed by the standard R and user library paths.

## Install packages

```powershell
Rscript analysis/run/install_r_packages.R
```

The installer adds missing packages to `.r-library/` without updating packages already
installed. This directory is an OS-dependent local environment and is excluded from Git.

| Package | Validated version | Use |
|---|---|---|
| nlme | 3.1.164 | Mixed-effects models |
| clubSandwich | 0.7.0 | Cluster-robust covariance and tests |
| sandwich | 3.1.3 | Covariance support |
| emmeans | 1.11.1 | Marginal comparisons and approximate degrees of freedom |

The validated R version is 4.4.1. This table records the reference environment; the
installer does not pin these versions. Run numerical verification with
`analysis.run.reproduce` after setting up a new environment.

# R statistics

Run from `Homecage/` with R 4.4.1:

```powershell
Rscript analysis/run/install_r_packages.R
```

| Package | Validated version |
|---|---|
| nlme | 3.1.164 |
| clubSandwich | 0.7.0 |
| sandwich | 3.1.3 |
| emmeans | 1.11.1 |

The installer adds missing packages to `.r-library/`; it does not pin versions or update
existing packages. Run `python -B -m analysis.run.reproduce` to verify a new environment.
Use entry points under `analysis/run`, rather than executing these core modules directly.

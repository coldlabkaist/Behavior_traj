# Paper reproduction commands

Assay calculations are in `analysis/core/{three_chamber,mom_pup,oft}/paper.py`, and
plotting is in `analysis/plots/{three_chamber,mom_pup,oft}/paper.py`. This directory
handles execution, comparison, and output saving. Individual assay commands are
organized in assay-specific subfolders.

The runner uses separately supplied final inputs from `behavior_assay/output/Final`
and writes to `behavior_assay/output/reproduced`. It does not overwrite Final figures
or statistics, or read the historical `output/3chamber`, `output/mom_pup`, or `output/OFT`
directories.

Run from the `Behavior_traj` root in an environment with the analysis requirements
installed. The full run was validated in the local `moval` conda environment:

```powershell
conda activate moval
python behavior_assay/analysis/run/reproduce.py
```

To select panels or verify statistics only:

```powershell
python behavior_assay/analysis/run/reproduce.py --panels Fig5F Fig5H
python behavior_assay/analysis/run/reproduce.py --mode stats
python behavior_assay/analysis/run/reproduce.py --panels FigS10A FigS10B --output-dir behavior_assay/output/oft_check
```

- `--mode all` is the default: recalculate statistics, compare with Final, and generate figures.
- `--mode figures` also recalculates and checks statistics before annotating figures; it only skips duplicate CSV exports.
- A failed statistical comparison stops the run. Regenerated results never silently replace the reference.
- Change the input location with `--final-dir`. Default paths are resolved from source-file locations, independent of the current working directory or user name.
- `verification.json` records the tables compared and the scope of figure regeneration.

## Inputs

All paths below refer to the separately supplied Final package.

| Panel | Final input |
|---|---|
| Fig5B, FigS8A | Density, support, ROI, and session counts in `inputs/three_chamber/spatial_maps.npz` |
| Fig5C, FigS8B | Panel-specific `data/individual_preference.csv` |
| FigS8C, FigS8D | Panel-specific `data/individual_metrics.csv` |
| Fig5E | `data/occupancy_counts.npz`, `data/selected_sessions.csv` |
| Fig5F | `pct_time_sustained_body_scale_proximity` in `data/cage_proximity.csv` |
| Fig5H | `avg_velocity_mm_s` in `data/cage_velocity.csv` |
| FigS10A | `data/density_control.csv`, `data/density_vpa.csv` |
| FigS10B | `data/individual_metrics.csv` |

Fig5F checks and uses the time percentage for bouts lasting at least one second.
Fig5F/H comparisons use Holm correction; FigS10B uses the unadjusted Welch P values
reported in the caption. The Fig5E detection-example PNG is copied from the external
editing source, while occupancy maps are redrawn from count grids.

## Rebuild three-chamber density maps

The density grids were reconstructed from raw tracking and added to the local Final
package. Valid-coordinate counts for 82 sessions and summaries for four groups were
checked against existing records. If the NPZ is present, the default runner does not
read raw tracking or historical preprocessing files.

To rebuild the NPZ, place raw tracking and ROI pin files under the repository-root
`data/csv/Fig5BC_FigS8ABCD` directory and run:

```powershell
python behavior_assay/analysis/run/prepare_density.py
```

`analysis/paths.py` resolves historical paths in Final manifests to the current data
location. Temporary preprocessing files are removed when the task finishes.

The default reproduction scope is **final individual/cage-level values and numerical
maps → statistics and figures**. Recalculating all behavioral metrics from raw tracking
is a separate step. Assay-specific commands retain their original analysis options;
`run/reproduce.py` is the main entry point for paper reproduction.

## Validation

Validation recalculated 13 statistical tables and generated all 11 panels.
Fig5C and FigS8B/C/D PNGs matched the reference pixels. Three-chamber map color grids
also matched the raster content embedded in the reference SVGs. Other figures can
have font, margin, or raster differences due to their original rendering environment;
pixel identity is not claimed for those figures. Final reference figures are preserved.

The full runner also worked from a directory outside the project, with no reads from
historical output directories. Reproduction of all 13 tables and 11 panels was checked
with access to six old output folders blocked before those folders and duplicate outputs
were removed. `output/reproduced` is recreated on the next run.

Validated environment: Python 3.9.25, numpy 1.26.4, pandas 2.3.3, scipy 1.13.1,
matplotlib 3.9.4, statsmodels 0.14.6, OpenCV 4.11.0, and shapely 2.0.6 (`moval`).
Dependencies also include patsy and Pillow. Raw preprocessing uses this project's
`data_loader` module. Use an activated environment in which OpenCV DLLs load correctly.

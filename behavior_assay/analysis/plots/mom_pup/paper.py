from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

import shutil
import matplotlib
from matplotlib.colors import LogNorm
from analysis.core.common import read
from analysis.plots.common import rename_bundle

def render(final: Path, panel: str, output: Path, data: pd.DataFrame, tables: dict) -> str:
    directory = output / panel / 'figure'
    directory.mkdir(parents=True, exist_ok=True)
    if panel == 'Fig5F':
        from analysis.plots.mom_pup.maternal_proximity_line import plot_movement_matched_export
        plot_movement_matched_export(data, tables['holm_group_comparisons.csv'], directory, metric='pct_time_sustained_body_scale_proximity', stem='maternal_proximity', title='Maternal proximity', y_label='Proximity time (%)', star_size=32)
    elif panel == 'Fig5H':
        from analysis.plots.mom_pup.focused_results import plot_locomotion
        contrasts = tables['holm_group_comparisons.csv'].rename(columns={'p_holm_across_3_pnd': 'p_holm_within_metric'}).copy()
        contrasts['metric'] = 'avg_velocity_mm_s'
        plot_locomotion(data, contrasts, directory)
        rename_bundle(directory, 'maternal_locomotion_by_pnd', 'maternal_velocity')
    elif panel == 'Fig5E':
        from analysis.plots.mom_pup.spatial import plot_histogram_pair
        sessions = read(final, panel, 'data', 'selected_sessions.csv')
        with np.load(final / panel / 'data/occupancy_counts.npz', allow_pickle=False) as z:
            grids = {key: z[key] for key in z.files}
        cmap = matplotlib.colormaps['Reds'].copy()
        cmap.set_under('white')
        for (pnd, group) in sessions.groupby('pnd'):
            values = np.concatenate([h[h > 0] for (key, h) in grids.items() if key.startswith(f'PND{pnd}_')])
            norm = LogNorm(vmin=1, vmax=max(float(np.quantile(values, 0.995)), 2))
            for row in group.itertuples():
                stem = f'PND{pnd}_{row.condition}_{row.subject_id}'
                plot_histogram_pair(grids[stem + '_pup'], grids[stem + '_mother'], directory / f'{stem}.png', norm=norm, cmap=cmap, title=f'PND {pnd}')
        shutil.copy2(final / panel / 'figure/representative_detections.png', directory / 'representative_detections.png')
        return 'occupancy recomputed from grids; detection strip copied as external editorial artwork'
    return 'rendered from Final numerical inputs'

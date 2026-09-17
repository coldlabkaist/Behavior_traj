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
    if panel in ['Fig5C', 'FigS8B']:
        import analysis.plots.three_chamber.preference_bars as plot
        from analysis.core.three_chamber.preference_bars import _compute_shared_panel_ylims
        phase = 'nov' if panel == 'Fig5C' else 'soc'
        combined = pd.concat([read(final, p, 'data', 'individual_preference.csv') for p in ['Fig5C', 'FigS8B']])
        limits = _compute_shared_panel_ylims(combined, ['soc', 'nov'])
        (name, _) = plot._plot_phase_bars(data, phase=phase, output_dir=directory, keypoint='Nose', roi_mode='contact', radius_scale=1.2, file_suffix='', shared_panel_ylims=limits)
        rename_bundle(directory, name.stem, 'preference')
    elif panel in ['FigS8C', 'FigS8D']:
        import analysis.plots.three_chamber.locomotion as plot
        if panel == 'FigS8C':
            fn = lambda path: plot._plot_metric(data, phase_list=['hab_open', 'nov'], metric_col='total_distance_norm', ylabel='Total Distance', title='Movement Distance', output_path=path, minimal_axes=True)
            stem = 'locomotion_total_distance'
        else:
            fn = lambda path: plot._plot_open_spatial_preference(data, output_path=path)
            stem = 'open_spatial_preference'
        for ext in ['png', 'svg']:
            fn(directory / f'{stem}.{ext}')
    elif panel in ['Fig5B', 'FigS8A']:
        from analysis.plots.three_chamber.spatial_maps import _save_phase_figure
        path = final / 'inputs/three_chamber/spatial_maps.npz'
        if not path.exists():
            raise FileNotFoundError('Run analysis/run/prepare_density.py to build spatial_maps.npz')
        with np.load(path, allow_pickle=False) as archive:
            maps = {prefix: {tuple(key.split('__')[1:]): archive[key] for key in archive.files if key.startswith(prefix + '__')} for prefix in ['heat', 'support', 'count', 'object_polygon', 'analysis_polygon']}
        phase = 'nov' if panel == 'Fig5B' else 'soc'
        path = _save_phase_figure(phase=phase, condition_order=['Control', 'VPA'], group_heatmaps=maps['heat'], group_support_masks=maps['support'], group_counts={k: int(v) for (k, v) in maps['count'].items()}, group_object_circles={}, group_analysis_circles={}, group_object_polygons=maps['object_polygon'], group_analysis_polygons=maps['analysis_polygon'], output_dir=directory, keypoint='Nose', target_side='right', roi_radius_scale=1.2, file_suffix='', x_max=1.0, display_width_height_ratio=1.5, font_scale=1.0)
        rename_bundle(directory, path.stem, 'nose_density')
    return 'rendered from Final numerical inputs'

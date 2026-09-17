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
    if panel == 'FigS10B':
        from analysis.plots.oft.group_metrics import plot_core_oft_bars
        plot_core_oft_bars(data, tables['welch_tests.csv'], directory, p_column='p_value')
        rename_bundle(directory, 'core_oft_metrics', 'open_field_metrics')
    elif panel == 'FigS10A':
        from analysis.plots.oft.heatmaps import plot_group_densities
        grids = {c: np.loadtxt(final / panel / 'data' / f'density_{c.lower()}.csv', delimiter=',') for c in ['Control', 'VPA']}
        plot_group_densities(grids, directory)
        rename_bundle(directory, 'group_comparison', 'occupancy')
        rename_bundle(directory, 'shared_colorbar', 'colorbar')
    return 'rendered from Final numerical inputs'

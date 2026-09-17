from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from scipy import stats
from analysis.core.common import read

def compute(final: Path, panel: str) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    if panel in ['Fig5F', 'Fig5H']:
        from analysis.core.mom_pup.repeated_measures import METRIC, _validated_complete_data, split_plot_anova, tukey_emm_pairwise
        is_proximity = panel == 'Fig5F'
        data = read(final, panel, 'data', 'cage_proximity.csv' if is_proximity else 'cage_velocity.csv')
        metric = 'pct_time_sustained_body_scale_proximity' if is_proximity else 'avg_velocity_mm_s'
        if is_proximity:
            if not data.proximity_bout_min_duration_sec.eq(1).all():
                raise ValueError('Fig5F requires one-second bouts')
            np.testing.assert_allclose(data[metric], data.body_scale_proximity_bout_time_sec * data.fps / data.n_valid_frames * 100)
        model_data = _validated_complete_data(data.rename(columns={metric: METRIC}))
        anova = split_plot_anova(model_data)
        (emms, _, contrasts, info) = tukey_emm_pairwise(model_data, anova)
        if not is_proximity:
            for frame in [anova, emms, contrasts, info]:
                frame.insert(0, 'metric', metric)
        outputs = {'mixed_rm_anova.csv': anova, 'holm_group_comparisons.csv': contrasts, 'estimated_marginal_means.csv': emms, 'model_info.csv': info}
        return (data, {name: frame[read(final, panel, 'stat', name).columns] for (name, frame) in outputs.items()})
    return (pd.DataFrame(), {})

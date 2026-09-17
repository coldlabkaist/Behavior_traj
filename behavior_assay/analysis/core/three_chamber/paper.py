from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from scipy import stats
from analysis.core.common import read

def compute(final: Path, panel: str) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    if panel in ['Fig5C', 'FigS8B', 'FigS8C', 'FigS8D']:
        preference = panel in ['Fig5C', 'FigS8B']
        data = read(final, panel, 'data', 'individual_preference.csv' if preference else 'individual_metrics.csv')
        result_table = read(final, panel, 'stat', 'primary_tests.csv').copy()
        for (idx, row) in result_table.iterrows():
            sub = data.loc[data.phase.eq(row.phase)]
            if row.test == 'welch_ttest':
                metric = 'novel_preference_index' if panel == 'Fig5C' else 'total_distance_norm'
                (a, b) = [sub.loc[sub.condition.eq(c), metric] for c in ['Control', 'VPA']]
                result = stats.ttest_ind(a, b, equal_var=False)
                if 'n_a' in result_table:
                    result_table.loc[idx, ['n_a', 'n_b']] = [len(a), len(b)]
                else:
                    result_table.loc[idx, 'n'] = len(a) + len(b)
            else:
                condition = 'Control' if row.comparison.startswith('Control') else 'VPA'
                sub = sub.loc[sub.condition.eq(condition)]
                result_table.loc[idx, 'n'] = len(sub)
                if row.test == 'paired_ttest':
                    if preference:
                        (a, b) = ('familiar', 'novel') if panel == 'Fig5C' else ('empty', 'social')
                        suffix = 'time_s' if row.panel == 'investigation_time_s' else 'visit_count'
                        result = stats.ttest_rel(sub[f'{a}_{suffix}'], sub[f'{b}_{suffix}'])
                    else:
                        result = stats.ttest_rel(sub.left_zone_time_s, sub.right_zone_time_s)
                elif row.test == 'onesample_ttest':
                    metric = {'Fig5C': 'novel_preference_index', 'FigS8B': 'social_preference_index', 'FigS8D': 'spatial_preference_index'}[panel]
                    result = stats.ttest_1samp(sub[metric], 0)
                else:
                    raise ValueError(f'Unknown test: {row.test}')
            result_table.loc[idx, ['statistic', 'df', 'pvalue']] = [result.statistic, result.df, result.pvalue]
        return (data, {'primary_tests.csv': result_table})
    return (pd.DataFrame(), {})

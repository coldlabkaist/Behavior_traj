from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from scipy import stats
from analysis.core.common import read

def compute(final: Path, panel: str) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    if panel == 'FigS10B':
        data = read(final, panel, 'data', 'individual_metrics.csv')
        result_table = read(final, panel, 'stat', 'welch_tests.csv').copy()
        for (idx, row) in result_table.iterrows():
            (a, b) = [data.loc[data.condition.eq(c), row.metric] for c in ['VPA', 'Control']]
            result = stats.ttest_ind(a, b, equal_var=False)
            result_table.loc[idx, ['n_control', 'n_vpa', 'mean_control', 'mean_vpa', 'welch_t', 'df', 'p_value']] = [len(b), len(a), b.mean(), a.mean(), result.statistic, result.df, result.pvalue]
        return (data, {'welch_tests.csv': result_table})
    return (pd.DataFrame(), {})

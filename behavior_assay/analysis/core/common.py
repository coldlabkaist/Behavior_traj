from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

def read(final: Path, panel: str, category: str, name: str) -> pd.DataFrame:
    return pd.read_csv(final / panel / category / name)

def validate_table(actual: pd.DataFrame, expected: pd.DataFrame, label: str, report: list) -> None:
    """Compare only finite numerical values; labels/schema/NaN positions must also agree."""
    if list(actual.columns) != list(expected.columns) or len(actual) != len(expected):
        raise ValueError(f'{label}: schema or row count changed')
    for col in expected:
        a, b = actual[col], expected[col]
        if pd.api.types.is_numeric_dtype(b):
            np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-8, equal_nan=True, err_msg=f'{label}: {col}')
        elif a.fillna('').astype(str).tolist() != b.fillna('').astype(str).tolist():
            raise ValueError(f'{label}: {col} labels changed')
    report.append({'panel_table': label, 'rows': len(actual), 'matches_final': True})

"""core / rank_tests.

Published-analysis functions. Inputs and current sources are recorded in Final manifests.
"""

from __future__ import annotations
import itertools
from scipy import stats
import numpy as np
import pandas as pd


PERMUTATION_INDEX_CACHE: dict[int, np.ndarray] = {}


def permutation_indices(size: int) -> np.ndarray:
    if size not in PERMUTATION_INDEX_CACHE:
        PERMUTATION_INDEX_CACHE[size] = np.asarray(
            list(itertools.permutations(range(size))), dtype=np.int16
        )
    return PERMUTATION_INDEX_CACHE[size]


def centered_rank(values: np.ndarray, condition: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    result = np.zeros(len(values), dtype=float)
    for level in sorted(np.unique(condition)):
        indices = np.flatnonzero(condition == level)
        current = stats.rankdata(values[indices], method="average")
        result[indices] = current - current.mean()
    return result


def correlation_centered(x: np.ndarray, y: np.ndarray) -> float:
    denominator = float(np.sqrt((x @ x) * (y @ y)))
    return float((x @ y) / denominator) if denominator > 0 else np.nan


def exact_condition_adjusted_spearman(
    x_values: np.ndarray,
    y_values: np.ndarray,
    condition: np.ndarray,
) -> tuple[float, float]:
    x = centered_rank(x_values, condition)
    y = centered_rank(y_values, condition)
    observed = correlation_centered(x, y)
    threshold = abs(float(x @ y)) - 1e-12
    if threshold <= 0:
        return observed, 1.0

    dot_distributions: list[np.ndarray] = []
    for level in sorted(np.unique(condition)):
        indices = np.flatnonzero(condition == level)
        permutations = permutation_indices(len(indices))
        dot_distributions.append(x[indices][permutations] @ y[indices])
    if len(dot_distributions) != 2:
        raise ValueError("Two experimental conditions are required")

    first, second = dot_distributions
    second = np.sort(second)
    extreme = 0
    for value in first:
        lower = -threshold - value
        upper = threshold - value
        extreme += int(np.searchsorted(second, lower, side="right"))
        extreme += int(len(second) - np.searchsorted(second, upper, side="left"))
    return observed, float(extreme / (len(first) * len(second)))


def exact_spearman(x_values: np.ndarray, y_values: np.ndarray) -> tuple[float, float]:
    x = stats.rankdata(np.asarray(x_values, dtype=float), method="average")
    y = stats.rankdata(np.asarray(y_values, dtype=float), method="average")
    x -= x.mean()
    y -= y.mean()
    observed = correlation_centered(x, y)
    permutations = permutation_indices(len(x))
    null = (x[permutations] @ y) / np.sqrt((x @ x) * (y @ y))
    p_value = float(np.mean(np.abs(null) >= abs(observed) - 1e-12))
    return observed, p_value


def common_slope(x: np.ndarray, y: np.ndarray, condition: np.ndarray) -> float:
    code = (condition == "vpa").astype(float)
    centered = x.copy()
    for group in ("control", "vpa"):
        mask = condition == group
        centered[mask] -= centered[mask].mean()
    design = np.column_stack([np.ones(len(x)), code, centered])
    beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    return float(beta[-1])


def calculate_statistics(data: pd.DataFrame, predictor: str) -> pd.DataFrame:
    condition = data["condition"].to_numpy()
    x = data[predictor].to_numpy(float)
    y = data["score_3w"].to_numpy(float)
    overall_rho, overall_p = exact_condition_adjusted_spearman(x, y, condition)
    rows: list[dict[str, float | int | str]] = [
        {
            "analysis": "all_litters_condition_adjusted",
            "n": int(len(data)),
            "spearman_rho": overall_rho,
            "exact_permutation_p": overall_p,
        }
    ]
    for group in ("control", "vpa"):
        current = data.loc[data["condition"].eq(group)]
        rho, p_value = exact_spearman(
            current[predictor].to_numpy(float),
            current["score_3w"].to_numpy(float),
        )
        rows.append(
            {
                "analysis": group,
                "n": int(len(current)),
                "spearman_rho": rho,
                "exact_permutation_p": p_value,
            }
        )
    return pd.DataFrame(rows)

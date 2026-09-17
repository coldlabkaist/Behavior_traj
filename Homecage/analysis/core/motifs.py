"""Fit the K=4 motif codebook and assign window labels/posteriors."""
from __future__ import annotations


from sklearn.mixture import GaussianMixture

import numpy as np

import pandas as pd

from analysis.core.definitions import TOKENS as TOKEN_KEYS


WEEKS = (3, 4, 5, 6, 7, 8)

FACTOR_COLUMNS = ("spacing", "orientation", "dynamics")

K = 4

def _balanced_control_fit_indices(
    table: pd.DataFrame, *, per_cage_week: int, seed: int
) -> np.ndarray:
    control = table[table["condition"] == "control"]
    expected = {
        (str(cage), week)
        for cage in sorted(control["cage_id"].unique())
        for week in WEEKS
    }
    observed = set(control.groupby(["cage_id", "week"]).groups)
    missing = sorted(expected.difference(observed))
    if missing:
        raise RuntimeError(f"Incomplete Control cage-week cells: {missing}")
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for key, group in control.groupby(["cage_id", "week"], sort=True):
        indices = group.index.to_numpy(dtype=int)
        if len(indices) < per_cage_week:
            raise RuntimeError(
                f"Control cell {key} has {len(indices)} windows; "
                f"requested {per_cage_week}."
            )
        selected.append(rng.choice(indices, size=per_cage_week, replace=False))
    return np.concatenate(selected)

def _canonical_component_order(model: GaussianMixture) -> np.ndarray:
    """Map fitted components onto the established M0-M3 mechanical meanings."""
    dynamics_order = np.argsort(model.means_[:, 2])
    low_dynamics = dynamics_order[:2]
    high_dynamics = dynamics_order[2:]
    low_dynamics = low_dynamics[np.argsort(model.means_[low_dynamics, 0])]
    high_dynamics = high_dynamics[np.argsort(-model.means_[high_dynamics, 1])]
    mechanical_order = np.concatenate([low_dynamics, high_dynamics])
    return mechanical_order[np.asarray([0, 3, 1, 2], dtype=int)]

def _fit_and_assign(
    table: pd.DataFrame, *, per_cage_week: int, n_init: int, seed: int
) -> tuple[GaussianMixture, np.ndarray, np.ndarray, np.ndarray]:
    values = table[list(FACTOR_COLUMNS)].to_numpy(dtype=np.float64)
    fit_indices = _balanced_control_fit_indices(
        table, per_cage_week=per_cage_week, seed=seed
    )
    model = GaussianMixture(
        n_components=K,
        covariance_type="full",
        reg_covar=1e-4,
        n_init=n_init,
        max_iter=600,
        init_params="kmeans",
        random_state=seed,
    ).fit(values[fit_indices])
    if not model.converged_:
        raise RuntimeError("Control-only K=4 GMM did not converge.")
    order = _canonical_component_order(model)
    posterior = model.predict_proba(values)[:, order]
    if not np.allclose(posterior.sum(axis=1), 1.0, atol=1e-8):
        raise RuntimeError("GMM posterior rows do not sum to one.")
    return model, order, fit_indices, posterior

def _assignment_table(table: pd.DataFrame, posterior: np.ndarray) -> pd.DataFrame:
    result = table.copy()
    for token, key in enumerate(TOKEN_KEYS):
        result[f"p_{key}"] = posterior[:, token]
    result["hard_token"] = np.argmax(posterior, axis=1).astype(np.int16)
    result["max_posterior"] = posterior.max(axis=1)
    return result

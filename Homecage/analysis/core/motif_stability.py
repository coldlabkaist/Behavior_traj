"""Motif resolution: profile novelty, codebook refits and stability summaries."""
from __future__ import annotations
from pathlib import Path
from scipy.optimize import linear_sum_assignment, minimize
from scipy.stats import t
from sklearn.mixture import GaussianMixture
from typing import Any
import joblib
import numpy as np
import pandas as pd
import warnings
from analysis.core.definitions import FEATURES, PROFILE_KEYS
from analysis.core.latent_cache import load_latent_cache


# Novelty metrics

def _convex_novelty(parent: np.ndarray, child: np.ndarray) -> tuple[int, float]:
    distance = np.linalg.norm(parent[:, None, :] - child[None, :, :], axis=2)
    _, matched_children = linear_sum_assignment(distance)
    unmatched = sorted(set(range(len(child))) - set(matched_children))
    if len(unmatched) != 1:
        raise RuntimeError(f"Expected one added component, found {unmatched}")
    added = unmatched[0]
    target = child[added]
    result = minimize(
        lambda weights: np.sum((weights @ parent - target) ** 2),
        np.ones(len(parent), dtype=float) / len(parent),
        bounds=[(0.0, 1.0)] * len(parent),
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        method="SLSQP",
        options={"maxiter": 100, "ftol": 1e-10},
    )
    if not result.success:
        raise RuntimeError(result.message)
    return added, float(np.sqrt(result.fun / target.size))


def _cage_profiles(
    posterior: np.ndarray,
    features: np.ndarray,
    cages: np.ndarray,
    cage_order: list[str],
) -> np.ndarray:
    result = np.empty((len(cage_order), posterior.shape[1], features.shape[1]), dtype=float)
    for cage_index, cage in enumerate(cage_order):
        selected = cages == cage
        weights = posterior[selected]
        denominator = np.maximum(weights.sum(axis=0), 1e-12)
        result[cage_index] = weights.T @ features[selected] / denominator[:, None]
    return result


# Motif novelty

K_VALUES = tuple(range(2, 9))


def _cage_novelty(
    cache_path: Path,
    candidate_root: Path,
) -> pd.DataFrame:
    payload = load_latent_cache(cache_path)
    condition = np.asarray(payload["condition"], dtype=object).astype(str)
    cage_all = np.asarray(payload["cage_id"], dtype=object).astype(str)
    relation = payload["relation_mean"].detach().cpu().numpy().astype(float)
    profile_indices = [
        next(feature.cache_index for feature in FEATURES if feature.key == key)
        for key in PROFILE_KEYS
    ]
    raw = relation[:, profile_indices]
    control = np.char.lower(condition) == "control"
    median = np.median(raw[control], axis=0)
    scale = 1.4826 * np.median(np.abs(raw[control] - median), axis=0)
    fallback = np.std(raw[control], axis=0, ddof=1)
    scale = np.where(scale > 1e-8, scale, fallback)
    standardized = (raw - median) / np.maximum(scale, 1e-8)

    profiles_by_k: dict[int, np.ndarray] = {}
    evaluation_indices: np.ndarray | None = None
    evaluation_cages: np.ndarray | None = None
    cage_order: list[str] | None = None
    for k in K_VALUES:
        assignment = np.load(
            candidate_root / f"control_soft_gmm_k{k}_heldout_assignments.npz"
        )
        indices = assignment["cache_index"].astype(int)
        if evaluation_indices is None:
            evaluation_indices = indices
            evaluation_cages = cage_all[indices]
            cage_order = sorted(np.unique(evaluation_cages).tolist())
        elif not np.array_equal(evaluation_indices, indices):
            raise RuntimeError("K candidates do not use the same held-out rows.")
        profiles_by_k[k] = _cage_profiles(
            assignment["posterior"].astype(float),
            standardized[indices],
            evaluation_cages,
            cage_order,
        )

    assert evaluation_indices is not None
    assert evaluation_cages is not None
    assert cage_order is not None
    k1 = np.empty((len(cage_order), 1, len(profile_indices)), dtype=float)
    for cage_index, cage in enumerate(cage_order):
        selected = evaluation_cages == cage
        k1[cage_index, 0] = standardized[evaluation_indices[selected]].mean(axis=0)
    profiles_by_k[1] = k1

    rows: list[dict[str, str | int | float]] = []
    for cage_index, cage in enumerate(cage_order):
        for k in K_VALUES:
            _, novelty = _convex_novelty(
                profiles_by_k[k - 1][cage_index],
                profiles_by_k[k][cage_index],
            )
            rows.append({"cage": cage, "k": k, "novelty": novelty})
    return pd.DataFrame(rows)


# Codebook stability

WEEKS = (3, 4, 5, 6, 7, 8)


def _align_components(
    reference: GaussianMixture,
    bootstrap: GaussianMixture,
    axis_scale: np.ndarray,
) -> np.ndarray:
    difference = (
        reference.means_[:, None, :] - bootstrap.means_[None, :, :]
    ) / axis_scale[None, None, :]
    cost = np.sqrt(np.sum(difference**2, axis=2))
    reference_rows, bootstrap_columns = linear_sum_assignment(cost)
    aligned = np.empty(reference.n_components, dtype=int)
    aligned[reference_rows] = bootstrap_columns
    return aligned


def _load_control(
    coordinates_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[tuple[str, int], np.ndarray]]:
    with np.load(coordinates_path, allow_pickle=False) as payload:
        coordinates = payload["factor_coordinates"].astype(np.float64)
        condition = payload["condition"].astype(str)
        cage = payload["cage_id"].astype(str)
        week = payload["week"].astype(int)
    selected = (np.char.lower(condition) == "control") & np.isin(week, WEEKS)
    groups: dict[tuple[str, int], np.ndarray] = {}
    for cage_id in sorted(np.unique(cage[selected]).tolist()):
        for week_value in WEEKS:
            indices = np.flatnonzero(
                selected & (cage == cage_id) & (week == week_value)
            )
            if not len(indices):
                raise RuntimeError(f"Missing Control cell: {cage_id}, {week_value}W")
            groups[(cage_id, week_value)] = indices
    return coordinates, cage, week, groups


def _bootstrap_indices(
    groups: dict[tuple[str, int], np.ndarray],
    cages: list[str],
    *,
    per_cage_week: int,
    rng: np.random.Generator,
) -> np.ndarray:
    selected_cages = rng.choice(cages, size=len(cages), replace=True)
    pieces: list[np.ndarray] = []
    for cage_id in selected_cages:
        for week_value in WEEKS:
            pool = groups[(str(cage_id), week_value)]
            pieces.append(
                rng.choice(
                    pool,
                    size=per_cage_week,
                    replace=len(pool) < per_cage_week,
                )
            )
    return np.concatenate(pieces)


def _fit_gmm(values: np.ndarray, k: int, *, seed: int) -> GaussianMixture:
    model = GaussianMixture(
        n_components=k,
        covariance_type="full",
        reg_covar=1e-4,
        n_init=2,
        max_iter=350,
        random_state=seed,
        init_params="kmeans",
    )
    model.fit(values)
    if not model.converged_:
        raise RuntimeError(f"Bootstrap GMM did not converge for K={k}.")
    return model


def _stability_by_evaluation_cage(
    coordinates_path: Path,
    candidate_root: Path,
    output_path: Path,
    *,
    iterations: int,
    per_cage_week: int,
    seed: int,
) -> pd.DataFrame:
    if output_path.exists():
        cached = pd.read_csv(output_path)
        expected = {"k", "iteration", "evaluation_cage", "soft_agreement"}
        if expected.issubset(cached.columns) and cached["iteration"].nunique() == iterations:
            return cached

    coordinates, cage_all, _, groups = _load_control(coordinates_path)
    cages = sorted({key[0] for key in groups})
    references: dict[int, GaussianMixture] = {}
    evaluation_indices: np.ndarray | None = None
    train_indices: np.ndarray | None = None
    for k in K_VALUES:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            references[k] = joblib.load(candidate_root / f"control_soft_gmm_k{k}.joblib")
        parameters = np.load(
            candidate_root / f"control_soft_gmm_k{k}_parameters.npz",
            allow_pickle=False,
        )
        current_eval = parameters["heldout_cache_index"].astype(int)
        current_train = parameters["train_cache_index"].astype(int)
        if evaluation_indices is None:
            evaluation_indices = current_eval
            train_indices = current_train
        elif not np.array_equal(evaluation_indices, current_eval):
            raise RuntimeError("K candidates do not share held-out evaluation rows.")
        elif not np.array_equal(train_indices, current_train):
            raise RuntimeError("K candidates do not share balanced training rows.")
    assert evaluation_indices is not None and train_indices is not None
    evaluation = coordinates[evaluation_indices]
    evaluation_cages = cage_all[evaluation_indices]
    axis_scale = np.std(coordinates[train_indices], axis=0, ddof=1)

    rows: list[dict[str, Any]] = []
    for k in K_VALUES:
        reference = references[k]
        reference_posterior = reference.predict_proba(evaluation)
        rng = np.random.default_rng(seed + 1000 * k)
        for iteration in range(iterations):
            indices = _bootstrap_indices(
                groups,
                cages,
                per_cage_week=per_cage_week,
                rng=rng,
            )
            model = _fit_gmm(
                coordinates[indices],
                k,
                seed=seed + 10000 * k + iteration,
            )
            aligned = _align_components(reference, model, axis_scale)
            posterior = model.predict_proba(evaluation)[:, aligned]
            agreement = 1.0 - 0.5 * np.sum(
                np.abs(reference_posterior - posterior), axis=1
            )
            for evaluation_cage in cages:
                selected = evaluation_cages == evaluation_cage
                rows.append(
                    {
                        "k": k,
                        "iteration": iteration,
                        "evaluation_cage": evaluation_cage,
                        "soft_agreement": float(agreement[selected].mean()),
                    }
                )
        print(f"Completed cage-resolved stability refits for K={k}", flush=True)
    result = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def _mean_t_ci(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    sem = float(values.std(ddof=1) / np.sqrt(len(values)))
    margin = float(t.ppf(0.975, df=len(values) - 1) * sem)
    return mean - margin, mean + margin


def summarize_resolution(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in ("novelty", "stability"):
        for k, frame in metrics.groupby("k"):
            values = frame[metric].to_numpy(float)
            low, high = _mean_t_ci(values)
            rows.append(
                {
                    "metric": metric,
                    "k": int(k),
                    "mean": float(values.mean()),
                    "ci95_low": low,
                    "ci95_high": high,
                    "n_cages": int(len(values)),
                }
            )
    return pd.DataFrame(rows)

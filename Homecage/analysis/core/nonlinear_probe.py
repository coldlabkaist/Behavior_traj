"""Cage-balanced sampling, linear comparison and nonlinear probe validation."""
from __future__ import annotations
from pathlib import Path
from scipy import stats
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from typing import Any
import math
import numpy as np
import pandas as pd
import warnings
from analysis.core.latent_cache import load_latent_cache


# Probe sampling

BLOCKS = {
    "spacing": np.asarray([0, 6, 7]),
    "orientation": np.asarray([3]),
    "contact": np.asarray([1, 2]),
    "dynamics": np.asarray([4, 5, 8]),
}


BLOCK_METRIC_INDICES = BLOCKS


def _balanced_indices(
    condition: np.ndarray,
    cage: np.ndarray,
    week: np.ndarray,
    cap_per_cage_week: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    chosen: list[np.ndarray] = []
    frame = pd.DataFrame(
        {
            "row": np.arange(len(cage)),
            "condition": condition,
            "cage": cage,
            "week": week,
        }
    )
    for _, group in frame.groupby(["condition", "cage", "week"], sort=True):
        idx = group["row"].to_numpy(dtype=int)
        if len(idx) > cap_per_cage_week:
            idx = rng.choice(idx, size=cap_per_cage_week, replace=False)
        chosen.append(np.sort(idx))
    return np.sort(np.concatenate(chosen))


def _fit_scaled_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_mean = x_train.mean(axis=0)
    x_sd = x_train.std(axis=0, ddof=0)
    x_sd[x_sd < 1e-8] = 1.0
    y_mean = y_train.mean(axis=0)
    y_sd = y_train.std(axis=0, ddof=0)
    y_sd[y_sd < 1e-8] = 1.0
    model = Ridge(alpha=float(alpha))
    model.fit((x_train - x_mean) / x_sd, (y_train - y_mean) / y_sd)
    pred_scaled = model.predict((x_test - x_mean) / x_sd)
    return pred_scaled * y_sd + y_mean, y_mean, y_sd


def _choose_alpha(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    alphas: tuple[float, ...],
) -> float:
    unique_groups = np.unique(groups)
    splitter = GroupKFold(n_splits=min(4, len(unique_groups)))
    scores = {alpha: [] for alpha in alphas}
    for train_idx, val_idx in splitter.split(x, groups=groups):
        y_train = y[train_idx]
        y_mean = y_train.mean(axis=0)
        y_sd = y_train.std(axis=0, ddof=0)
        y_sd[y_sd < 1e-8] = 1.0
        for alpha in alphas:
            pred, _, _ = _fit_scaled_ridge(
                x[train_idx], y_train, x[val_idx], alpha
            )
            scaled_error = (y[val_idx] - pred) / y_sd
            scores[alpha].append(float(np.mean(scaled_error * scaled_error)))
    return min(alphas, key=lambda alpha: float(np.mean(scores[alpha])))


def _holm(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p, np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if len(valid) == 0:
        return adjusted
    order = valid[np.argsort(p[valid])]
    running = 0.0
    m = len(order)
    for rank, idx in enumerate(order):
        running = max(running, float(p[idx]) * (m - rank))
        adjusted[idx] = min(1.0, running)
    return adjusted


# Nonlinear probe

def _safe_r2(y: np.ndarray, pred: np.ndarray) -> float:
    if len(y) < 2 or float(np.std(y)) < 1e-12:
        return math.nan
    return float(r2_score(y, pred))


def _safe_corr(y: np.ndarray, pred: np.ndarray) -> float:
    if len(y) < 2 or float(np.std(y)) < 1e-12 or float(np.std(pred)) < 1e-12:
        return math.nan
    return float(np.corrcoef(y, pred)[0, 1])


def _block_for_target(index: int) -> str:
    return next(block for block, indices in BLOCKS.items() if index in indices)


def run_nonlinear_probe(
    cache_path: str | Path,
    out_dir: str | Path,
    *,
    cap_per_cage_week: int = 300,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare linear and shallow nonlinear probes with held-out cages."""
    payload = load_latent_cache(cache_path)
    z = payload["z"].numpy().astype(np.float64)
    relation = payload["relation_mean"].numpy().astype(np.float64)
    names = list(payload["relation_feature_names"])
    condition = np.asarray(payload["condition"], dtype=object)
    cage = np.asarray(payload["cage_id"], dtype=object)
    week = np.rint(np.asarray(payload["week"], dtype=float)).astype(int)

    selected = _balanced_indices(
        condition,
        cage,
        week,
        cap_per_cage_week=cap_per_cage_week,
        seed=seed,
    )
    z = z[selected]
    relation = relation[selected]
    condition = condition[selected]
    cage = cage[selected]

    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    pred_linear = np.full_like(relation, np.nan)
    pred_mlp = np.full_like(relation, np.nan)
    standardized_true = np.full_like(relation, np.nan)
    standardized_linear = np.full_like(relation, np.nan)
    standardized_mlp = np.full_like(relation, np.nan)
    fold_rows: list[dict[str, Any]] = []
    alphas = (0.1, 1.0, 10.0, 100.0)

    for fold, (train_idx, test_idx) in enumerate(
        outer.split(z, y=condition, groups=cage), start=1
    ):
        alpha = _choose_alpha(
            z[train_idx], relation[train_idx], cage[train_idx], alphas
        )
        pred_linear[test_idx], y_mean, y_sd = _fit_scaled_ridge(
            z[train_idx], relation[train_idx], z[test_idx], alpha
        )

        x_scaler = StandardScaler().fit(z[train_idx])
        y_scaler = StandardScaler().fit(relation[train_idx])
        mlp = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-3,
            batch_size=512,
            learning_rate_init=1e-3,
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=12,
            random_state=seed + fold,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            mlp.fit(
                x_scaler.transform(z[train_idx]),
                y_scaler.transform(relation[train_idx]),
            )
        pred_mlp[test_idx] = y_scaler.inverse_transform(
            mlp.predict(x_scaler.transform(z[test_idx]))
        )

        standardized_true[test_idx] = (relation[test_idx] - y_mean) / y_sd
        standardized_linear[test_idx] = (pred_linear[test_idx] - y_mean) / y_sd
        standardized_mlp[test_idx] = (pred_mlp[test_idx] - y_mean) / y_sd
        fold_rows.append(
            {
                "fold": fold,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "ridge_alpha": float(alpha),
                "mlp_iterations": int(mlp.n_iter_),
                "mlp_best_validation_score": float(mlp.best_validation_score_),
                "test_cages": "|".join(sorted(set(cage[test_idx]))),
            }
        )

    target_rows: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        target_rows.append(
            {
                "target": name,
                "block": _block_for_target(index),
                "linear_r2": _safe_r2(relation[:, index], pred_linear[:, index]),
                "nonlinear_r2": _safe_r2(relation[:, index], pred_mlp[:, index]),
                "nonlinear_minus_linear_r2": _safe_r2(
                    relation[:, index], pred_mlp[:, index]
                )
                - _safe_r2(relation[:, index], pred_linear[:, index]),
                "linear_r": _safe_corr(relation[:, index], pred_linear[:, index]),
                "nonlinear_r": _safe_corr(relation[:, index], pred_mlp[:, index]),
            }
        )
    target_metrics = pd.DataFrame(target_rows)
    block_rows = []
    for block, indices in BLOCK_METRIC_INDICES.items():
        selected_targets = target_metrics.iloc[indices]
        block_rows.append(
            {
                "block": block,
                "linear_r2": float(selected_targets["linear_r2"].mean()),
                "nonlinear_r2": float(selected_targets["nonlinear_r2"].mean()),
                "nonlinear_minus_linear_r2": float(
                    selected_targets["nonlinear_minus_linear_r2"].mean()
                ),
            }
        )
    block_metrics = pd.DataFrame(block_rows)

    cage_rows: list[dict[str, Any]] = []
    for cage_id in sorted(set(cage)):
        mask = cage == cage_id
        for block, indices in BLOCK_METRIC_INDICES.items():
            linear_error = np.mean(
                (standardized_true[mask][:, indices] - standardized_linear[mask][:, indices])
                ** 2
            )
            mlp_error = np.mean(
                (standardized_true[mask][:, indices] - standardized_mlp[mask][:, indices])
                ** 2
            )
            cage_rows.append(
                {
                    "condition": str(condition[mask][0]),
                    "cage_id": cage_id,
                    "block": block,
                    "linear_standardized_mse": float(linear_error),
                    "nonlinear_standardized_mse": float(mlp_error),
                    "mse_improvement": float(linear_error - mlp_error),
                }
            )
    cage_metrics = pd.DataFrame(cage_rows)

    test_rows: list[dict[str, Any]] = []
    for block in BLOCKS:
        delta = cage_metrics.loc[
            cage_metrics["block"] == block, "mse_improvement"
        ].to_numpy(dtype=float)
        result = stats.wilcoxon(delta, alternative="greater", zero_method="wilcox")
        test_rows.append(
            {
                "block": block,
                "n_cages": int(len(delta)),
                "positive_cages": int(np.sum(delta > 0)),
                "mean_mse_improvement": float(np.mean(delta)),
                "median_mse_improvement": float(np.median(delta)),
                "p_one_sided": float(result.pvalue),
            }
        )
    tests = pd.DataFrame(test_rows)
    tests["p_holm_4_blocks"] = _holm(tests["p_one_sided"].to_numpy())

    out_dir = Path(out_dir)
    target_metrics.to_csv(out_dir / "nonlinear_probe_targets.csv", index=False)
    block_metrics.to_csv(out_dir / "nonlinear_probe_blocks.csv", index=False)
    cage_metrics.to_csv(out_dir / "nonlinear_probe_cages.csv", index=False)
    tests.to_csv(out_dir / "nonlinear_probe_tests.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(out_dir / "nonlinear_probe_folds.csv", index=False)
    return target_metrics, block_metrics, tests

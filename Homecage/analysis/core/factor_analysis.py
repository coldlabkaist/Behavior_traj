"""Factor extraction, rotations and block parallel analysis."""
from __future__ import annotations
import itertools
import numpy as np
import pandas as pd


# Factor methods

def _fit_robust_scaler(reference: np.ndarray) -> dict[str, np.ndarray]:
    lower = np.quantile(reference, 0.01, axis=0)
    upper = np.quantile(reference, 0.99, axis=0)
    clipped = np.clip(reference, lower, upper)
    median = np.median(clipped, axis=0)
    iqr = np.quantile(clipped, 0.75, axis=0) - np.quantile(
        clipped, 0.25, axis=0
    )
    iqr = np.where(iqr > 1e-8, iqr, 1.0)
    return {"lower": lower, "upper": upper, "median": median, "iqr": iqr}


def _scale(values: np.ndarray, scaler: dict[str, np.ndarray]) -> np.ndarray:
    clipped = np.clip(values, scaler["lower"], scaler["upper"])
    return (clipped - scaler["median"]) / scaler["iqr"]


def _principal_axis_factor(
    correlation: np.ndarray,
    *,
    factors: int,
    max_iterations: int = 10000,
    tolerance: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, int, bool]:
    """Principal-axis factoring with squared multiple correlations as starts."""
    inverse = np.linalg.pinv(correlation)
    communalities = 1.0 - 1.0 / np.diag(inverse)
    communalities = np.clip(communalities, 0.01, 0.995)
    converged = False
    loadings = np.empty((len(correlation), factors), dtype=np.float64)
    for iteration in range(1, max_iterations + 1):
        reduced = correlation.copy()
        np.fill_diagonal(reduced, communalities)
        eigenvalues, eigenvectors = np.linalg.eigh(reduced)
        order = np.argsort(eigenvalues)[::-1][:factors]
        retained = np.maximum(eigenvalues[order], 0.0)
        loadings = eigenvectors[:, order] * np.sqrt(retained)[None, :]
        updated = np.clip(np.square(loadings).sum(axis=1), 0.01, 0.995)
        if float(np.max(np.abs(updated - communalities))) < tolerance:
            communalities = updated
            converged = True
            break
        communalities = updated
    return loadings, communalities, iteration, converged


def _varimax(
    loadings: np.ndarray,
    *,
    max_iterations: int = 1000,
    tolerance: float = 1e-8,
) -> np.ndarray:
    rows, factors = loadings.shape
    rotation = np.eye(factors, dtype=np.float64)
    previous = 0.0
    for _ in range(max_iterations):
        rotated = loadings @ rotation
        u, singular, vh = np.linalg.svd(
            loadings.T
            @ (
                rotated**3
                - (1.0 / rows)
                * rotated
                @ np.diag(np.diag(rotated.T @ rotated))
            )
        )
        rotation = u @ vh
        objective = float(singular.sum())
        if previous > 0 and objective / previous < 1.0 + tolerance:
            break
        previous = objective
    return loadings @ rotation


def _promax(
    varimax_loadings: np.ndarray,
    *,
    power: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Promax rotation starting from sklearn's Varimax solution."""
    communalities = np.square(varimax_loadings).sum(axis=1)
    normalization = np.sqrt(np.maximum(communalities, 1e-12))
    normalized = varimax_loadings / normalization[:, None]
    target = np.sign(normalized) * np.abs(normalized) ** power
    transform = np.linalg.pinv(normalized) @ target

    phi_unscaled = np.linalg.inv(transform.T @ transform)
    transform = transform @ np.diag(np.sqrt(np.diag(phi_unscaled)))
    pattern = normalized @ transform
    pattern = pattern * normalization[:, None]
    phi = np.linalg.inv(transform.T @ transform)
    return pattern, phi


# Factor analysis

FACTOR_NAMES = ("Spacing", "Orientation", "Dynamics")


FEATURE_NAMES = (
    "Body distance",
    "Nose-Nose distance",
    "Nose-Tailbase distance",
    "Mutual facing",
    "Approach rate",
    "Relative speed",
    "Mean pair distance",
    "Dyadic grouping",
    "Configuration speed",
)


DESIGNS = {
    "legacy_fa8": {
        "label": "Legacy FA8",
        "indices": (0, 1, 2, 3, 4, 5, 7, 8),
        "groups": ((0, 1, 2, 6), (3, 4), (5, 7)),
        "anchors": (0, 3, 5),
    },
    "reduced_fa6": {
        "label": "Reduced-indicator FA6",
        "indices": (0, 3, 4, 5, 7, 8),
        "groups": ((0, 4), (1, 2), (3, 5)),
        "anchors": (0, 1, 3),
    },
}


def _metadata(payload: dict) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "condition": np.asarray(payload["condition"], dtype=object).astype(str),
            "cage_id": np.asarray(payload["cage_id"], dtype=object).astype(str),
            "sex": np.asarray(payload["sex"], dtype=object).astype(str),
            "week": np.rint(np.asarray(payload["week"], dtype=float)).astype(int),
        }
    )


def _balanced_control_indices(
    metadata: pd.DataFrame, cap_per_cage_week: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    control = metadata.index[metadata["condition"].str.lower() == "control"].to_numpy()
    frame = metadata.loc[control]
    selected: list[np.ndarray] = []
    for _, group in frame.groupby(["cage_id", "week"], sort=True):
        indices = group.index.to_numpy(dtype=int)
        if len(indices) < cap_per_cage_week:
            raise RuntimeError(f"Control cage-week has only {len(indices)} windows.")
        selected.append(np.sort(rng.choice(indices, cap_per_cage_week, replace=False)))
    return np.concatenate(selected)


def _standardize(reference: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    scaler = _fit_robust_scaler(reference)
    robust = _scale(reference, scaler)
    center = robust.mean(axis=0)
    spread = robust.std(axis=0, ddof=1)
    spread = np.where(spread > 1e-8, spread, 1.0)
    return (robust - center) / spread, {**scaler, "center": center, "spread": spread}


def _block_parallel_analysis(
    values: np.ndarray,
    metadata: pd.DataFrame,
    iterations: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    observed = np.linalg.eigvalsh(np.corrcoef(values, rowvar=False))[::-1]
    blocks = [
        group.index.to_numpy(dtype=int)
        for _, group in metadata.reset_index(drop=True).groupby(["cage_id", "week"], sort=True)
    ]
    if len({len(block) for block in blocks}) != 1:
        raise RuntimeError("Balanced cage-week blocks are required.")
    rng = np.random.default_rng(seed)
    null = np.empty((iterations, values.shape[1]), dtype=float)
    permuted = np.empty_like(values)
    for iteration in range(iterations):
        for feature in range(values.shape[1]):
            order = rng.permutation(len(blocks))
            for target, source in enumerate(order):
                permuted[blocks[target], feature] = values[blocks[source], feature]
        null[iteration] = np.linalg.eigvalsh(np.corrcoef(permuted, rowvar=False))[::-1]
    return observed, np.quantile(null, 0.95, axis=0)


def _congruence(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 1e-12 else np.nan


def _order_by_concept(
    pattern: np.ndarray,
    phi: np.ndarray,
    groups: tuple[tuple[int, ...], ...],
    anchors: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    order = max(
        itertools.permutations(range(3)),
        key=lambda permutation: sum(
            float(np.mean(np.abs(pattern[list(groups[index]), factor])))
            for index, factor in enumerate(permutation)
        ),
    )
    pattern = pattern[:, order].copy()
    phi = phi[np.ix_(order, order)].copy()
    for factor, anchor in enumerate(anchors):
        if pattern[anchor, factor] < 0:
            pattern[:, factor] *= -1
            phi[factor, :] *= -1
            phi[:, factor] *= -1
    return pattern, phi


def _align_to_reference(
    pattern: np.ndarray, phi: np.ndarray, reference: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    order = max(
        itertools.permutations(range(3)),
        key=lambda permutation: sum(
            abs(_congruence(reference[:, factor], pattern[:, permutation[factor]]))
            for factor in range(3)
        ),
    )
    pattern = pattern[:, order].copy()
    phi = phi[np.ix_(order, order)].copy()
    for factor in range(3):
        if _congruence(reference[:, factor], pattern[:, factor]) < 0:
            pattern[:, factor] *= -1
            phi[factor, :] *= -1
            phi[:, factor] *= -1
    return pattern, phi


def _fit_three_factor(
    values: np.ndarray,
    groups: tuple[tuple[int, ...], ...],
    anchors: tuple[int, ...],
    reference: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, bool]:
    correlation = np.corrcoef(values, rowvar=False)
    unrotated, _, _, converged = _principal_axis_factor(correlation, factors=3)
    pattern, phi = _promax(_varimax(unrotated))
    if reference is None:
        pattern, phi = _order_by_concept(pattern, phi, groups, anchors)
    else:
        pattern, phi = _align_to_reference(pattern, phi, reference)
    return pattern, phi, bool(converged)

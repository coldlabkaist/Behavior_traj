"""Shared temporal-shift inputs, hard/soft validation and cage-level summaries."""
from __future__ import annotations
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from scipy.stats import t
from sklearn.metrics import adjusted_rand_score, cohen_kappa_score
from sklearn.mixture import GaussianMixture
import numpy as np
import pandas as pd


# Temporal shift

WEEKS = (3, 4, 5, 6, 7, 8)


SHIFT_FRAMES = (-15, -8, 8, 15)


def fit_gmm(values: np.ndarray, seed: int) -> GaussianMixture:
    model = GaussianMixture(
        n_components=4,
        covariance_type="full",
        reg_covar=1e-4,
        n_init=2,
        max_iter=300,
        init_params="kmeans",
        random_state=seed,
    )
    model.fit(values)
    if not model.converged_:
        raise RuntimeError(f"GMM failed to converge for seed {seed}.")
    return model


def align_by_centroids(
    first: GaussianMixture,
    second: GaussianMixture,
    scale: np.ndarray,
) -> np.ndarray:
    difference = (
        first.means_[:, None, :] - second.means_[None, :, :]
    ) / scale[None, None, :]
    cost = np.sqrt(np.sum(difference**2, axis=2))
    first_rows, second_columns = linear_sum_assignment(cost)
    second_to_first = np.empty(4, dtype=int)
    second_to_first[second_columns] = first_rows
    return second_to_first


def balanced_sample(
    cages: np.ndarray,
    pools: dict[tuple[str, int], np.ndarray],
    per_cage_week: int,
    rng: np.random.Generator,
) -> np.ndarray:
    pieces = []
    for cage in cages:
        for week in WEEKS:
            pool = pools[(str(cage), week)]
            if len(pool) < per_cage_week:
                raise RuntimeError(
                    f"Cell {cage}, {week}W has {len(pool)} rows; "
                    f"requested {per_cage_week}."
                )
            pieces.append(rng.choice(pool, size=per_cage_week, replace=False))
    return np.concatenate(pieces)


def circular_randomization(
    labels: np.ndarray,
    file_path: np.ndarray,
    start_frame: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    randomized = np.empty_like(labels)
    for recording in np.unique(file_path):
        positions = np.flatnonzero(file_path == recording)
        positions = positions[np.argsort(start_frame[positions])]
        minimum_lag = min(10, max(1, len(positions) // 4))
        if len(positions) > 2 * minimum_lag:
            roll = int(rng.integers(minimum_lag, len(positions) - minimum_lag + 1))
            randomized[positions] = np.roll(labels[positions], roll)
        else:
            randomized[positions] = rng.permutation(labels[positions])
    return randomized


def load_inputs(coordinates_path: Path, shifted_path: Path) -> dict[str, object]:
    with np.load(coordinates_path, allow_pickle=False) as payload:
        coordinates = payload["factor_coordinates"].astype(np.float64)
        condition = payload["condition"].astype(str)
        cage = payload["cage_id"].astype(str)
        week = payload["week"].astype(int)
        file_path = payload["file_path"].astype(str)
        start_frame = payload["clip_start_frame_id"].astype(int)
    with np.load(shifted_path, allow_pickle=False) as payload:
        shifted_frames = payload["shift_frames"].astype(int)
        shifted_coordinates = payload["factor_coordinates"].astype(np.float64)
        shifted_valid = payload["valid"].astype(bool)

    frame_positions = {
        int(frame): int(np.flatnonzero(shifted_frames == frame)[0])
        for frame in SHIFT_FRAMES
    }
    control = (np.char.lower(condition) == "control") & np.isin(week, WEEKS)
    cages = np.asarray(sorted(np.unique(cage[control]).tolist()), dtype=str)
    if len(cages) != 12:
        raise RuntimeError(f"Expected 12 Control cages, found {len(cages)}.")

    base_pools: dict[tuple[str, int], np.ndarray] = {}
    shift_pools: dict[int, dict[tuple[str, int], np.ndarray]] = {
        frame: {} for frame in SHIFT_FRAMES
    }
    for cage_id in cages:
        for week_value in WEEKS:
            cell = control & (cage == cage_id) & (week == week_value)
            base_pools[(str(cage_id), week_value)] = np.flatnonzero(cell)
            for frame in SHIFT_FRAMES:
                position = frame_positions[frame]
                shift_pools[frame][(str(cage_id), week_value)] = np.flatnonzero(
                    cell & shifted_valid[position]
                )

    return {
        "coordinates": coordinates,
        "shifted_coordinates": shifted_coordinates,
        "shifted_valid": shifted_valid,
        "frame_positions": frame_positions,
        "control": control,
        "cage": cage,
        "week": week,
        "file_path": file_path,
        "start_frame": start_frame,
        "cages": cages,
        "base_pools": base_pools,
        "shift_pools": shift_pools,
    }


def score_predictions(
    first: np.ndarray,
    second: np.ndarray,
    randomized_second: np.ndarray,
) -> dict[str, float]:
    return {
        "hard_agreement": float(np.mean(first == second)),
        "hard_agreement_randomized": float(np.mean(first == randomized_second)),
        "adjusted_rand": float(adjusted_rand_score(first, second)),
        "adjusted_rand_randomized": float(
            adjusted_rand_score(first, randomized_second)
        ),
        "cohen_kappa": float(cohen_kappa_score(first, second)),
        "cohen_kappa_randomized": float(
            cohen_kappa_score(first, randomized_second)
        ),
    }


def run_hard_validation(
    data: dict[str, object],
    *,
    iterations: int,
    per_cage_week: int,
    seed: int,
    cache_path: Path,
) -> pd.DataFrame:
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        if cached["iteration"].nunique() == iterations:
            return cached

    coordinates = data["coordinates"]
    shifted_coordinates = data["shifted_coordinates"]
    shifted_valid = data["shifted_valid"]
    frame_positions = data["frame_positions"]
    control = data["control"]
    cage = data["cage"]
    file_path = data["file_path"]
    start_frame = data["start_frame"]
    cages = data["cages"]
    base_pools = data["base_pools"]
    shift_pools = data["shift_pools"]
    axis_scale = np.std(coordinates[control], axis=0, ddof=1)
    rows: list[dict[str, float | int | str]] = []

    for held_out_number, held_out_cage in enumerate(cages, start=1):
        remaining = cages[cages != held_out_cage]
        for iteration in range(iterations):
            rng = np.random.default_rng(seed + held_out_number * 10000 + iteration)
            permuted = rng.permutation(remaining)
            first_training_cages = permuted[:5]
            second_training_cages = permuted[5:]

            first_indices = balanced_sample(
                first_training_cages, base_pools, per_cage_week, rng
            )
            second_indices = balanced_sample(
                second_training_cages, base_pools, per_cage_week, rng
            )
            first_model = fit_gmm(
                coordinates[first_indices],
                seed + held_out_number * 100000 + iteration * 10,
            )
            second_model = fit_gmm(
                coordinates[second_indices],
                seed + held_out_number * 100000 + iteration * 10 + 1,
            )

            evaluation = control & (cage == held_out_cage)
            evaluation_indices = np.flatnonzero(evaluation)
            first_labels = first_model.predict(coordinates[evaluation_indices])
            mapping = align_by_centroids(first_model, second_model, axis_scale)
            second_labels = mapping[
                second_model.predict(coordinates[evaluation_indices])
            ]
            randomized_labels = circular_randomization(
                second_labels,
                file_path[evaluation_indices],
                start_frame[evaluation_indices],
                rng,
            )
            scores = score_predictions(
                first_labels, second_labels, randomized_labels
            )
            rows.append(
                {
                    "held_out_cage": str(held_out_cage),
                    "iteration": iteration,
                    "scenario": "same_boundary",
                    "shift_frames": 0,
                    "shift_seconds": 0.0,
                    "shift_magnitude_seconds": 0.0,
                    "n_evaluation_windows": int(len(evaluation_indices)),
                    **scores,
                }
            )

            for frame in SHIFT_FRAMES:
                position = frame_positions[frame]
                shifted_second_indices = balanced_sample(
                    second_training_cages,
                    shift_pools[frame],
                    per_cage_week,
                    rng,
                )
                shifted_model = fit_gmm(
                    shifted_coordinates[position, shifted_second_indices],
                    seed
                    + held_out_number * 100000
                    + iteration * 10
                    + 2
                    + SHIFT_FRAMES.index(frame),
                )
                shifted_mapping = align_by_centroids(
                    first_model, shifted_model, axis_scale
                )
                valid_evaluation = evaluation & shifted_valid[position]
                valid_indices = np.flatnonzero(valid_evaluation)
                current_first_labels = first_model.predict(coordinates[valid_indices])
                current_second_labels = shifted_mapping[
                    shifted_model.predict(
                        shifted_coordinates[position, valid_indices]
                    )
                ]
                current_randomized = circular_randomization(
                    current_second_labels,
                    file_path[valid_indices],
                    start_frame[valid_indices],
                    rng,
                )
                scores = score_predictions(
                    current_first_labels,
                    current_second_labels,
                    current_randomized,
                )
                rows.append(
                    {
                        "held_out_cage": str(held_out_cage),
                        "iteration": iteration,
                        "scenario": "shifted_boundary",
                        "shift_frames": int(frame),
                        "shift_seconds": float(frame / 30.0),
                        "shift_magnitude_seconds": float(abs(frame) / 30.0),
                        "n_evaluation_windows": int(len(valid_indices)),
                        **scores,
                    }
                )
        print(
            f"Completed held-out cage {held_out_number:02d}/{len(cages)}: "
            f"{held_out_cage}",
            flush=True,
        )

    result = pd.DataFrame(rows)
    result.to_csv(cache_path, index=False)
    return result


def hard_cage_level(raw: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "hard_agreement",
        "hard_agreement_randomized",
        "adjusted_rand",
        "adjusted_rand_randomized",
        "cohen_kappa",
        "cohen_kappa_randomized",
    ]
    cage = (
        raw.groupby(
            ["held_out_cage", "scenario", "shift_magnitude_seconds"],
            as_index=False,
        )[metrics]
        .mean()
        .sort_values(["shift_magnitude_seconds", "held_out_cage"])
        .reset_index(drop=True)
    )
    return cage


MAGNITUDES = (0.0, 8.0 / 30.0, 0.5)


LABELS = ("No shift", "±0.27 s", "±0.50 s")


def _mean_ci(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    half = float(
        t.ppf(0.975, len(values) - 1)
        * values.std(ddof=1)
        / np.sqrt(len(values))
    )
    return mean, mean - half, mean + half


def summarize_temporal_shift(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for magnitude, label in zip(MAGNITUDES, LABELS):
        current = data[np.isclose(data["shift_magnitude_seconds"], magnitude)]
        for metric, metric_label in (
            ("hard_agreement", "Hard label"),
            ("js_similarity", "Soft posterior"),
        ):
            mean, low, high = _mean_ci(current[metric].to_numpy(dtype=float))
            rows.append(
                {
                    "shift_magnitude_seconds": float(magnitude),
                    "shift_label": label,
                    "metric": metric,
                    "metric_label": metric_label,
                    "mean": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    summary = pd.DataFrame(rows)
    baseline = summary[np.isclose(summary["shift_magnitude_seconds"], 0.0)].set_index("metric")["mean"]
    summary["decrease_from_no_shift"] = summary.apply(
        lambda row: float(baseline[row["metric"]] - row["mean"]), axis=1
    )
    return summary


# Temporal shift soft

def _aligned_posterior(raw_posterior: np.ndarray, second_to_first: np.ndarray) -> np.ndarray:
    aligned = np.empty_like(raw_posterior)
    aligned[:, second_to_first] = raw_posterior
    return aligned


def _posterior_overlap(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Probability-mass overlap, equal to 1 - total-variation distance."""
    return np.minimum(first, second).sum(axis=1)


def _js_similarity(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """One minus Jensen-Shannon divergence, using the same definition as the original shift validation."""
    midpoint = 0.5 * (first + second)
    first_term = np.where(
        first > 0,
        first * np.log2(first / np.maximum(midpoint, 1e-300)),
        0.0,
    )
    second_term = np.where(
        second > 0,
        second * np.log2(second / np.maximum(midpoint, 1e-300)),
        0.0,
    )
    divergence = 0.5 * (first_term.sum(axis=1) + second_term.sum(axis=1))
    return 1.0 - np.clip(divergence, 0.0, 1.0)


def _score(
    first_model,
    second_model,
    first_values: np.ndarray,
    second_values: np.ndarray,
    mapping: np.ndarray,
) -> dict[str, float]:
    first_posterior = first_model.predict_proba(first_values)
    second_raw = second_model.predict_proba(second_values)
    second_posterior = _aligned_posterior(second_raw, mapping)
    overlap = _posterior_overlap(first_posterior, second_posterior)
    js_similarity = _js_similarity(first_posterior, second_posterior)
    first_hard = np.argmax(first_posterior, axis=1)
    second_hard = np.argmax(second_posterior, axis=1)
    return {
        "posterior_overlap": float(np.mean(overlap)),
        "posterior_overlap_sd_windows": float(np.std(overlap, ddof=1)),
        "js_similarity": float(np.mean(js_similarity)),
        "js_similarity_sd_windows": float(np.std(js_similarity, ddof=1)),
        "hard_agreement_check": float(np.mean(first_hard == second_hard)),
    }


def run_soft_validation(
    data: dict[str, object],
    *,
    iterations: int,
    per_cage_week: int,
    seed: int,
    cache_path: Path,
) -> pd.DataFrame:
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        if cached["iteration"].nunique() == iterations and "js_similarity" in cached:
            return cached

    coordinates = data["coordinates"]
    shifted_coordinates = data["shifted_coordinates"]
    shifted_valid = data["shifted_valid"]
    frame_positions = data["frame_positions"]
    control = data["control"]
    cage = data["cage"]
    file_path = data["file_path"]
    start_frame = data["start_frame"]
    cages = data["cages"]
    base_pools = data["base_pools"]
    shift_pools = data["shift_pools"]
    axis_scale = np.std(coordinates[control], axis=0, ddof=1)
    rows: list[dict[str, float | int | str]] = []

    for held_out_number, held_out_cage in enumerate(cages, start=1):
        remaining = cages[cages != held_out_cage]
        for iteration in range(iterations):
            rng = np.random.default_rng(seed + held_out_number * 10000 + iteration)
            permuted = rng.permutation(remaining)
            first_training_cages = permuted[:5]
            second_training_cages = permuted[5:]

            first_indices = balanced_sample(first_training_cages, base_pools, per_cage_week, rng)
            second_indices = balanced_sample(second_training_cages, base_pools, per_cage_week, rng)
            first_model = fit_gmm(
                coordinates[first_indices],
                seed + held_out_number * 100000 + iteration * 10,
            )
            second_model = fit_gmm(
                coordinates[second_indices],
                seed + held_out_number * 100000 + iteration * 10 + 1,
            )

            evaluation = control & (cage == held_out_cage)
            evaluation_indices = np.flatnonzero(evaluation)
            mapping = align_by_centroids(first_model, second_model, axis_scale)
            scores = _score(
                first_model,
                second_model,
                coordinates[evaluation_indices],
                coordinates[evaluation_indices],
                mapping,
            )
            # Preserve the original analysis RNG sequence exactly so the fits and
            # training samples match the existing hard-label validation.
            second_labels = mapping[second_model.predict(coordinates[evaluation_indices])]
            circular_randomization(
                second_labels,
                file_path[evaluation_indices],
                start_frame[evaluation_indices],
                rng,
            )
            rows.append(
                {
                    "held_out_cage": str(held_out_cage),
                    "iteration": iteration,
                    "scenario": "same_boundary",
                    "shift_frames": 0,
                    "shift_seconds": 0.0,
                    "shift_magnitude_seconds": 0.0,
                    "n_evaluation_windows": int(len(evaluation_indices)),
                    **scores,
                }
            )

            for frame in SHIFT_FRAMES:
                position = frame_positions[frame]
                shifted_second_indices = balanced_sample(
                    second_training_cages,
                    shift_pools[frame],
                    per_cage_week,
                    rng,
                )
                shifted_model = fit_gmm(
                    shifted_coordinates[position, shifted_second_indices],
                    seed
                    + held_out_number * 100000
                    + iteration * 10
                    + 2
                    + SHIFT_FRAMES.index(frame),
                )
                shifted_mapping = align_by_centroids(first_model, shifted_model, axis_scale)
                valid_evaluation = evaluation & shifted_valid[position]
                valid_indices = np.flatnonzero(valid_evaluation)
                scores = _score(
                    first_model,
                    shifted_model,
                    coordinates[valid_indices],
                    shifted_coordinates[position, valid_indices],
                    shifted_mapping,
                )
                shifted_labels = shifted_mapping[
                    shifted_model.predict(shifted_coordinates[position, valid_indices])
                ]
                circular_randomization(
                    shifted_labels,
                    file_path[valid_indices],
                    start_frame[valid_indices],
                    rng,
                )
                rows.append(
                    {
                        "held_out_cage": str(held_out_cage),
                        "iteration": iteration,
                        "scenario": "shifted_boundary",
                        "shift_frames": int(frame),
                        "shift_seconds": float(frame / 30.0),
                        "shift_magnitude_seconds": float(abs(frame) / 30.0),
                        "n_evaluation_windows": int(len(valid_indices)),
                        **scores,
                    }
                )
        print(f"Completed held-out cage {held_out_number:02d}/{len(cages)}: {held_out_cage}", flush=True)

    result = pd.DataFrame(rows)
    result.to_csv(cache_path, index=False)
    return result


def soft_cage_level(raw: pd.DataFrame) -> pd.DataFrame:
    return (
        raw.groupby(
            ["held_out_cage", "scenario", "shift_magnitude_seconds"],
            as_index=False,
        )[["posterior_overlap", "js_similarity", "hard_agreement_check"]]
        .mean()
        .sort_values(["shift_magnitude_seconds", "held_out_cage"])
        .reset_index(drop=True)
    )

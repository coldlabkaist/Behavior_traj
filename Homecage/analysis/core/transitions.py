"""Hard/soft motif transitions, weekly changes and within-cage age effects."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import itertools
import numpy as np
import pandas as pd
from analysis.core.definitions import (
    TOKENS,
    TRANSITION_COLUMNS,
    TOKENS as TOKEN_KEYS,
    OCCUPANCY_COLUMNS,
)


# Soft joint

POSTERIOR_COLUMNS = ("p_M0", "p_M1", "p_M2", "p_M3")


def _load_assignments(path: Path) -> pd.DataFrame:
    columns = [
        "condition",
        "cage_id",
        "sex",
        "week",
        "file_path",
        "clip_start_frame_id",
        *POSTERIOR_COLUMNS,
    ]
    data = pd.read_csv(path, usecols=columns)
    data["condition"] = data["condition"].str.lower()
    data["sex"] = data["sex"].str.lower()
    data = data.sort_values(
        ["condition", "cage_id", "week", "file_path", "clip_start_frame_id"]
    ).reset_index(drop=True)
    posterior_sum = data[list(POSTERIOR_COLUMNS)].sum(axis=1).to_numpy(float)
    if not np.allclose(posterior_sum, 1.0, atol=1e-8):
        raise RuntimeError("GMM posterior probabilities do not sum to one.")
    return data


def _extract_soft_pair_contributions(
    assignments: pd.DataFrame,
) -> tuple[dict[tuple[str, str, int], np.ndarray], pd.DataFrame, pd.DataFrame]:
    """Return one 16-state soft joint contribution per valid adjacent window pair."""
    pair_contributions: dict[tuple[str, str, int], np.ndarray] = {}
    metadata_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    for (condition, cage_id, week), group in assignments.groupby(
        ["condition", "cage_id", "week"], sort=False
    ):
        group = group.sort_values(["file_path", "clip_start_frame_id"])
        posterior = group[list(POSTERIOR_COLUMNS)].to_numpy(float)
        starts = group["clip_start_frame_id"].to_numpy(float)
        files = group["file_path"].astype(str).to_numpy()
        within_file = files[:-1] == files[1:]
        exact_next_window = (starts[1:] - starts[:-1]) == 30
        valid = within_file & exact_next_window
        pair_joint = (
            posterior[:-1, :, None] * posterior[1:, None, :]
        ).reshape(-1, 16)[valid]
        if len(pair_joint) < 2:
            raise RuntimeError(
                f"Too few adjacent pairs for {condition}, {cage_id}, {week}W."
            )
        key = (str(condition), str(cage_id), int(week))
        pair_contributions[key] = pair_joint
        metadata_rows.append(
            {
                "condition": str(condition),
                "cage_id": str(cage_id),
                "week": int(week),
                "sex": str(group["sex"].iloc[0]),
                "n_windows": int(len(group)),
                "n_adjacent_pairs": int(valid.sum()),
            }
        )
        diagnostic_rows.append(
            {
                "condition": str(condition),
                "cage_id": str(cage_id),
                "week": int(week),
                "n_candidate_pairs": int(len(valid)),
                "n_exact_30_frame_pairs": int(valid.sum()),
                "n_pairs_excluded_for_gap": int((~valid).sum()),
                "retained_fraction": float(valid.mean()),
            }
        )
    metadata = pd.DataFrame(metadata_rows)
    if metadata.duplicated(["condition", "cage_id", "week"]).any():
        raise RuntimeError("Duplicate cage-week metadata rows were generated.")
    return pair_contributions, metadata, pd.DataFrame(diagnostic_rows)


# Soft probabilities

def _build_soft_probability_tables(
    assignments: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    keys = ["condition", "cage_id", "week"]

    occupancy = (
        assignments.groupby(keys, as_index=False)[list(POSTERIOR_COLUMNS)]
        .mean()
        .rename(columns={column: column for column in POSTERIOR_COLUMNS})
    )

    pair_contributions, pair_metadata, pair_diagnostics = (
        _extract_soft_pair_contributions(assignments)
    )
    joint_rows: list[dict[str, Any]] = []
    conditional_rows: list[dict[str, Any]] = []
    for (condition, cage_id, week), contributions in pair_contributions.items():
        joint = contributions.mean(axis=0).reshape(len(TOKENS), len(TOKENS))
        if not np.isclose(joint.sum(), 1.0, atol=1e-8):
            raise RuntimeError(
                f"Soft expected joint matrix does not sum to one for "
                f"{condition}, {cage_id}, {week}W."
            )
        source_mass = joint.sum(axis=1, keepdims=True)
        if np.any(source_mass <= 0):
            raise RuntimeError(
                f"A soft expected transition row has zero source mass for "
                f"{condition}, {cage_id}, {week}W."
            )
        conditional = joint / source_mass
        base = {
            "condition": condition,
            "cage_id": cage_id,
            "week": int(week),
        }
        joint_rows.append(
            {
                **base,
                **{
                    f"{source}_to_{target}": float(joint[i, j])
                    for i, source in enumerate(TOKENS)
                    for j, target in enumerate(TOKENS)
                },
            }
        )
        conditional_rows.append(
            {
                **base,
                **{
                    f"{source}_to_{target}": float(conditional[i, j])
                    for i, source in enumerate(TOKENS)
                    for j, target in enumerate(TOKENS)
                },
            }
        )

    joint_table = pd.DataFrame(joint_rows).sort_values(keys).reset_index(drop=True)
    conditional_table = (
        pd.DataFrame(conditional_rows).sort_values(keys).reset_index(drop=True)
    )
    occupancy = occupancy.sort_values(keys).reset_index(drop=True)
    pair_metadata = pair_metadata.sort_values(keys).reset_index(drop=True)
    pair_diagnostics = pair_diagnostics.sort_values(keys).reset_index(drop=True)

    if not occupancy[keys].equals(conditional_table[keys]):
        raise RuntimeError("Soft occupancy and transition cage-week keys are not aligned.")
    if not occupancy[keys].equals(pair_metadata[keys]):
        raise RuntimeError("Soft probability tables and pair metadata are not aligned.")

    conditional_values = conditional_table[list(TRANSITION_COLUMNS)].to_numpy(float)
    row_sums = conditional_values.reshape(-1, len(TOKENS), len(TOKENS)).sum(axis=2)
    if not np.allclose(row_sums, 1.0, atol=1e-8):
        raise RuntimeError("Soft expected conditional transition rows do not sum to one.")

    return occupancy, joint_table, conditional_table, pair_diagnostics


# Age effect

WEEKS = (3, 4, 5, 6, 7, 8)


CONDITIONS = ("control", "vpa")


JOINT_COLUMNS = tuple(f"M{i}_to_M{j}" for i in range(4) for j in range(4))


def holm(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    ranked = p_values[order] * (len(p_values) - np.arange(len(p_values)))
    ranked = np.maximum.accumulate(ranked)
    adjusted = np.empty_like(ranked)
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted


def load_joint_table(input_dir: Path) -> pd.DataFrame:
    table = pd.read_csv(input_dir / "primary_soft_joint_cage_week_distributions.csv")
    required = {"condition", "cage_id", "week", *JOINT_COLUMNS}
    missing = required.difference(table.columns)
    if missing:
        raise RuntimeError(f"Missing columns: {sorted(missing)}")
    probability = table[list(JOINT_COLUMNS)].to_numpy(float)
    if np.any(probability < -1e-12):
        raise RuntimeError("Joint probabilities contain negative values.")
    totals = probability.sum(axis=1)
    if not np.allclose(totals, 1.0, atol=1e-8):
        raise RuntimeError("Joint probability rows do not sum to one.")
    result = table[["condition", "cage_id", "week", *JOINT_COLUMNS]].copy()
    result[list(JOINT_COLUMNS)] = np.sqrt(np.clip(probability, 0.0, None))
    return result.sort_values(["condition", "cage_id", "week"]).reset_index(drop=True)


def global_age_permutation(
    frame: pd.DataFrame,
    *,
    n_permutations: int,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    frame = frame.reset_index(drop=True)
    values = frame[list(JOINT_COLUMNS)].to_numpy(float)
    cage_groups = [np.asarray(x, dtype=int) for x in frame.groupby("cage_id").indices.values()]
    centered = values.copy()
    for indices in cage_groups:
        centered[indices] -= centered[indices].mean(axis=0, keepdims=True)

    week = frame["week"].to_numpy(int)

    def statistic(current: np.ndarray) -> float:
        total = 0.0
        for current_week in WEEKS:
            mask = week == current_week
            if np.any(mask):
                mean = current[mask].mean(axis=0)
                total += float(mask.sum() * np.dot(mean, mean))
        return total

    observed = statistic(centered)
    exceedances = 0
    for _ in range(n_permutations):
        permuted = centered.copy()
        for indices in cage_groups:
            permuted[indices] = centered[rng.permutation(indices)]
        exceedances += statistic(permuted) >= observed - 1e-15

    return {
        "statistic": observed,
        "n_permutations": n_permutations,
        "p_value": (exceedances + 1) / (n_permutations + 1),
    }


# Motifs

K = 4


def _transition_tables(
    assignments: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    transition_rows: list[dict[str, Any]] = []
    occupancy_rows: list[dict[str, Any]] = []
    for (condition, cage_id, week), group in assignments.groupby(
        ["condition", "cage_id", "week"], sort=True
    ):
        counts = np.zeros((K, K), dtype=np.int64)
        adjacent_count = 0
        for _, recording in group.groupby("file_path", sort=False):
            recording = recording.sort_values("clip_start_frame_id")
            tokens = recording["hard_token"].to_numpy(dtype=int)
            starts = recording["clip_start_frame_id"].to_numpy(dtype=int)
            if len(tokens) < 2:
                continue
            adjacent = np.diff(starts) == 30
            source = tokens[:-1][adjacent]
            target = tokens[1:][adjacent]
            np.add.at(counts, (source, target), 1)
            adjacent_count += int(adjacent.sum())
        totals = counts.sum(axis=1)
        for source, target in itertools.product(range(K), repeat=2):
            probability = (
                float(counts[source, target] / totals[source])
                if totals[source] > 0
                else np.nan
            )
            transition_rows.append(
                {
                    "condition": condition,
                    "cage_id": str(cage_id),
                    "week": int(week),
                    "source": TOKEN_KEYS[source],
                    "target": TOKEN_KEYS[target],
                    "count": int(counts[source, target]),
                    "source_total": int(totals[source]),
                    "transition_probability": probability,
                    "n_adjacent_pairs": adjacent_count,
                }
            )
        for token, key in enumerate(TOKEN_KEYS):
            occupancy_rows.append(
                {
                    "condition": condition,
                    "cage_id": str(cage_id),
                    "week": int(week),
                    "token": key,
                    "soft_occupancy": float(group[f"p_{key}"].mean()),
                    "hard_occupancy": float(np.mean(group["hard_token"] == token)),
                    "n_windows": int(len(group)),
                }
            )
    cage_transition = pd.DataFrame(transition_rows)
    cage_occupancy = pd.DataFrame(occupancy_rows)
    transition = (
        cage_transition.groupby(
            ["condition", "week", "source", "target"], as_index=False
        )
        .agg(
            transition_probability=("transition_probability", "mean"),
            sem=("transition_probability", "sem"),
            n_cages=("transition_probability", "count"),
            total_count=("count", "sum"),
        )
    )
    occupancy = (
        cage_occupancy.groupby(["condition", "week", "token"], as_index=False)
        .agg(
            soft_occupancy=("soft_occupancy", "mean"),
            soft_sem=("soft_occupancy", "sem"),
            hard_occupancy=("hard_occupancy", "mean"),
            n_cages=("soft_occupancy", "count"),
        )
    )
    return cage_transition, cage_occupancy, transition, occupancy


INTERVALS = ((3, 4), (4, 5), (5, 6))


def build_change_summary(motif_dir: Path | None, transition_mode: str = "soft", *,
                         assignments: pd.DataFrame | None = None) -> pd.DataFrame:
    if transition_mode == "hard":

        supplied_assignments = assignments is not None
        assignments = assignments.copy() if supplied_assignments else pd.read_csv(motif_dir / "gmm_k4_window_assignments.csv.gz")
        posterior_columns = [f"p_{token}" for token in TOKENS]
        assignments["hard_token"] = assignments[posterior_columns].to_numpy().argmax(axis=1)
        transitions, occupancies, _, _ = _transition_tables(assignments)
        # Recompute from windows and verify against the main figure's source.
        if not supplied_assignments:
            reference = pd.read_csv(motif_dir / "gmm_k4_cage_week_transition_probability.csv")
            index_columns = ["condition", "cage_id", "week", "source", "target"]
            pd.testing.assert_frame_equal(
                transitions.set_index(index_columns).sort_index(),
                reference.set_index(index_columns).sort_index(),
                check_dtype=False, atol=1e-12, rtol=1e-12,
            )
        keys = ["condition", "cage_id", "week"]
        occupancy = occupancies.pivot(index=keys, columns="token", values="soft_occupancy")
        occupancy = occupancy.rename(columns={token: f"p_{token}" for token in TOKENS}).reset_index()
        transitions["feature"] = transitions["source"] + "_to_" + transitions["target"]
        conditional = transitions.pivot(index=keys, columns="feature", values="transition_probability").reset_index()
        if not np.isfinite(conditional[list(TRANSITION_COLUMNS)].to_numpy()).all():
            raise ValueError("Undefined hard transition rows require explicit missing-source handling.")
    else:
        assignments = _load_assignments(motif_dir / "gmm_k4_window_assignments.csv.gz")
        occupancy, _, conditional, _ = _build_soft_probability_tables(assignments)
    keys = ["condition", "cage_id", "week"]
    rows: list[dict[str, object]] = []
    for condition in CONDITIONS:
        condition_occ = occupancy[occupancy["condition"].eq(condition)]
        condition_trans = conditional[conditional["condition"].eq(condition)]
        for start, end in INTERVALS:
            occ_start = condition_occ[condition_occ["week"].eq(start)].set_index("cage_id")
            occ_end = condition_occ[condition_occ["week"].eq(end)].set_index("cage_id")
            trans_start = condition_trans[condition_trans["week"].eq(start)].set_index("cage_id")
            trans_end = condition_trans[condition_trans["week"].eq(end)].set_index("cage_id")
            cages = (
                occ_start.index.intersection(occ_end.index)
                .intersection(trans_start.index)
                .intersection(trans_end.index)
                .sort_values()
            )
            for column in OCCUPANCY_COLUMNS:
                token = column.replace("p_", "")
                delta = (
                    occ_end.loc[cages, column].to_numpy(float)
                    - occ_start.loc[cages, column].to_numpy(float)
                )
                rows.append(
                    {
                        "condition": condition,
                        "start_week": start,
                        "end_week": end,
                        "feature_type": "occupancy",
                        "feature": column,
                        "source": token,
                        "target": "",
                        "n_cages": len(cages),
                        "mean_change_pp": float(100.0 * delta.mean()),
                        "sd_change_pp": float(100.0 * delta.std(ddof=1)),
                    }
                )
            for column in TRANSITION_COLUMNS:
                source, target = column.split("_to_")
                delta = (
                    trans_end.loc[cages, column].to_numpy(float)
                    - trans_start.loc[cages, column].to_numpy(float)
                )
                rows.append(
                    {
                        "condition": condition,
                        "start_week": start,
                        "end_week": end,
                        "feature_type": "transition",
                        "feature": column,
                        "source": source,
                        "target": target,
                        "n_cages": len(cages),
                        "mean_change_pp": float(100.0 * delta.mean()),
                        "sd_change_pp": float(100.0 * delta.std(ddof=1)),
                    }
                )
    return pd.DataFrame(rows)

"""Behavioral-axis targets, held-out-cage latent projection and coordinate outputs."""
from __future__ import annotations
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from typing import Any
import gzip
import io
import json
import numpy as np
import pandas as pd
import warnings
import zipfile
from analysis.core.factor_analysis import FACTOR_NAMES
from analysis.core.latent_cache import load_latent_cache
from analysis.core.nonlinear_probe import _balanced_indices
from analysis.plots.validation import render_projection_validation as _plot_projection_validation


# Projection inputs

def _factor_targets(
    payload: dict[str, Any],
    factor_model: dict[str, np.ndarray],
) -> np.ndarray:
    relation = payload["relation_mean"].detach().cpu().numpy().astype(np.float64)
    indices = factor_model["input_feature_indices"].astype(int)
    values = relation[:, indices]
    values = np.clip(values, factor_model["lower"], factor_model["upper"])
    values = (values - factor_model["median"]) / factor_model["iqr"]
    values = (
        values - factor_model["correlation_center"]
    ) / factor_model["correlation_scale"]
    scores = values @ factor_model["score_coefficients"]
    if "score_intercept" in factor_model:
        scores = scores + factor_model["score_intercept"]
    if not np.isfinite(scores).all():
        raise RuntimeError("Factor-score targets contain NaN or Inf.")
    return scores.astype(np.float64)


def _metadata(payload: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row": np.arange(len(payload["condition"]), dtype=int),
            "condition": np.asarray(payload["condition"], dtype=object),
            "cage_id": np.asarray(payload["cage_id"], dtype=object),
            "sex": np.asarray(payload["sex"], dtype=object),
            "week": np.rint(np.asarray(payload["week"], dtype=float)).astype(int),
        }
    )


def _balanced_global_indices(
    metadata: pd.DataFrame,
    mask: np.ndarray,
    *,
    cap_per_cage_week: int,
    seed: int,
) -> np.ndarray:
    global_indices = np.flatnonzero(mask)
    subset = metadata.iloc[global_indices]
    local = _balanced_indices(
        subset["condition"].to_numpy(dtype=object),
        subset["cage_id"].to_numpy(dtype=object),
        subset["week"].to_numpy(dtype=int),
        cap_per_cage_week=cap_per_cage_week,
        seed=seed,
    )
    return global_indices[local]


# Coordinate io

COORDINATE_BASENAME = "primary_factor_coordinates"


def _save_deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for key in sorted(arrays):
            buffer = io.BytesIO()
            np.save(buffer, np.asarray(arrays[key]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)


def _coordinate_frame(
    payload: dict,
    metadata: pd.DataFrame,
    all_relation: np.ndarray,
    scores: np.ndarray,
    model_id: str,
) -> pd.DataFrame:
    starts = np.asarray(payload["clip_start_frame_id"], dtype=np.int64)
    frame = pd.DataFrame(
        {
            "window_id": np.arange(len(scores), dtype=np.int64),
            "coordinate_model_id": model_id,
            "condition": metadata["condition"].to_numpy(dtype=str),
            "cage_id": metadata["cage_id"].to_numpy(dtype=str),
            "sex": metadata["sex"].to_numpy(dtype=str),
            "week": metadata["week"].to_numpy(dtype=np.int64),
            "file_path": np.asarray(payload["file_path"], dtype=str),
            "clip_start_frame_id": starts,
            "clip_end_frame_id": starts + 29,
            "window_duration_s": np.ones(len(scores), dtype=float),
        }
    )
    relation_names = [str(value) for value in payload["relation_feature_names"]]
    for index, name in enumerate(relation_names):
        frame[f"observed__{name}"] = all_relation[:, index]
    frame["spacing"] = scores[:, 0]
    frame["orientation"] = scores[:, 1]
    frame["dynamics"] = scores[:, 2]
    return frame


def _save_coordinate_artifacts(
    out_dir: Path,
    frame: pd.DataFrame,
    scores: np.ndarray,
    relation: np.ndarray,
    model_id: str,
) -> tuple[Path, Path]:
    npz_path = out_dir / f"{COORDINATE_BASENAME}.npz"
    arrays = {
        "window_id": frame["window_id"].to_numpy(dtype=np.int64),
        "coordinate_model_id": np.asarray(model_id),
        "condition": frame["condition"].to_numpy(dtype=str),
        "cage_id": frame["cage_id"].to_numpy(dtype=str),
        "sex": frame["sex"].to_numpy(dtype=str),
        "week": frame["week"].to_numpy(dtype=np.int64),
        "file_path": frame["file_path"].to_numpy(dtype=str),
        "clip_start_frame_id": frame["clip_start_frame_id"].to_numpy(dtype=np.int64),
        "clip_end_frame_id": frame["clip_end_frame_id"].to_numpy(dtype=np.int64),
        "factor_names": np.asarray(FACTOR_NAMES),
        "factor_coordinates": scores,
        "relation_feature_names": np.asarray(
            [column.removeprefix("observed__") for column in frame.columns if column.startswith("observed__")]
        ),
        "relation_mean": relation,
    }
    _save_deterministic_npz(npz_path, arrays)

    csv_path = out_dir / f"{COORDINATE_BASENAME}.csv.gz"
    csv_bytes = frame.to_csv(index=False, float_format="%.10g").encode("utf-8")
    with csv_path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=6, mtime=0) as zipped:
            zipped.write(csv_bytes)
    frame.head(200).to_csv(out_dir / f"{COORDINATE_BASENAME}_preview.csv", index=False)
    return npz_path, csv_path


def _cage_week_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for keys, group in frame.groupby(["condition", "cage_id", "sex", "week"], sort=True):
        row = {
            "condition": keys[0],
            "cage_id": keys[1],
            "sex": keys[2],
            "week": int(keys[3]),
            "n_windows": int(len(group)),
            "n_source_files": int(group["file_path"].nunique()),
        }
        for factor in ("spacing", "orientation", "dynamics"):
            values = group[factor].to_numpy(dtype=float)
            for q, label in ((.10, "q10"), (.25, "q25"), (.50, "median"), (.75, "q75"), (.90, "q90")):
                row[f"{factor}_{label}"] = float(np.quantile(values, q))
            row[f"{factor}_iqr"] = row[f"{factor}_q75"] - row[f"{factor}_q25"]
        rows.append(row)
    return pd.DataFrame(rows)


# Projection

FACTORS = ("Spacing", "Orientation", "Dynamics")


FACTOR_COLUMNS = ("spacing", "orientation", "dynamics")


def _fit_projection(
    cache_path: Path,
    factor_model_path: Path,
    output_dir: Path,
    *,
    seed: int,
    cap_per_cage_week: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    payload = load_latent_cache(cache_path)
    factor_npz = np.load(factor_model_path, allow_pickle=False)
    factor_model = {key: factor_npz[key] for key in factor_npz.files}
    z = payload["z"].detach().cpu().numpy().astype(np.float64)
    target = _factor_targets(payload, factor_model)
    metadata = _metadata(payload)
    condition = metadata["condition"].astype(str).str.lower().to_numpy()
    cage = metadata["cage_id"].astype(str).to_numpy()
    control_mask = condition == "control"
    vpa_indices = np.flatnonzero(condition == "vpa")
    control_cages = sorted(np.unique(cage[control_mask]).tolist())
    if len(control_cages) != 12:
        raise RuntimeError(f"Expected 12 Control cages, found {len(control_cages)}")

    prediction = np.full_like(target, np.nan, dtype=np.float64)
    vpa_fold_predictions: list[np.ndarray] = []
    fold_rows: list[dict[str, float | int | str]] = []

    for fold, held_out in enumerate(control_cages):
        train_mask = control_mask & (cage != held_out)
        test_indices = np.flatnonzero(control_mask & (cage == held_out))
        train_indices = _balanced_global_indices(
            metadata,
            train_mask,
            cap_per_cage_week=cap_per_cage_week,
            seed=seed + fold,
        )
        x_scaler = StandardScaler().fit(z[train_indices])
        y_scaler = StandardScaler().fit(target[train_indices])
        model = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-3,
            batch_size=512,
            learning_rate_init=1e-3,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,
            random_state=seed + fold,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            model.fit(
                x_scaler.transform(z[train_indices]),
                y_scaler.transform(target[train_indices]),
            )
        held_prediction = y_scaler.inverse_transform(
            model.predict(x_scaler.transform(z[test_indices]))
        )
        prediction[test_indices] = held_prediction
        vpa_fold_predictions.append(
            y_scaler.inverse_transform(model.predict(x_scaler.transform(z[vpa_indices])))
        )
        fold_rows.append(
            {
                "fold": fold,
                "held_out_control_cage": held_out,
                "n_training_control_cages": int(
                    metadata.loc[train_mask, "cage_id"].nunique()
                ),
                "n_balanced_training_windows": int(len(train_indices)),
                "n_held_out_windows": int(len(test_indices)),
                "iterations": int(model.n_iter_),
                **{
                    f"{FACTOR_COLUMNS[index]}_r2": float(
                        r2_score(target[test_indices, index], held_prediction[:, index])
                    )
                    for index in range(3)
                },
            }
        )

    prediction[vpa_indices] = np.mean(np.stack(vpa_fold_predictions, axis=0), axis=0)
    if not np.isfinite(prediction).all():
        raise RuntimeError("Control-LOCO latent projection did not cover all windows")

    relation = payload["relation_mean"].detach().cpu().numpy().astype(np.float64)
    model_id = "control_loco_mlp_z64_to_fa3_20260822"
    frame = _coordinate_frame(payload, metadata, relation, prediction, model_id)
    projection_dir = output_dir / "01_latent_projection"
    projection_dir.mkdir(parents=True, exist_ok=True)
    coordinate_npz, coordinate_csv = _save_coordinate_artifacts(
        projection_dir, frame, prediction, relation, model_id
    )
    _cage_week_summary(frame).to_csv(
        projection_dir / "latent_projected_factor_cage_week_summary.csv", index=False
    )
    pd.DataFrame(fold_rows).to_csv(
        projection_dir / "control_loco_projection_folds.csv", index=False
    )

    metric_rows: list[dict[str, float | int | str]] = []
    metric_frame = metadata.copy()
    for index, factor in enumerate(FACTOR_COLUMNS):
        metric_frame[f"target_{factor}"] = target[:, index]
        metric_frame[f"projected_{factor}"] = prediction[:, index]
        for subset in ("all", "control", "vpa"):
            mask = np.ones(len(frame), dtype=bool) if subset == "all" else condition == subset
            true = target[mask, index]
            pred = prediction[mask, index]
            metric_rows.append(
                {
                    "factor": factor,
                    "level": "window",
                    "subset": subset,
                    "n": int(mask.sum()),
                    "r2": float(r2_score(true, pred)),
                    "pearson_r": float(pearsonr(true, pred).statistic),
                    "spearman_r": float(spearmanr(true, pred).statistic),
                    "mae": float(np.mean(np.abs(true - pred))),
                }
            )

    cage_rows: list[dict[str, float | int | str]] = []
    for keys, group in metric_frame.groupby(
        ["condition", "cage_id", "sex", "week"], sort=True
    ):
        row: dict[str, float | int | str] = {
            "condition": keys[0],
            "cage_id": keys[1],
            "sex": keys[2],
            "week": int(keys[3]),
        }
        for factor in FACTOR_COLUMNS:
            true = group[f"target_{factor}"].to_numpy(dtype=float)
            pred = group[f"projected_{factor}"].to_numpy(dtype=float)
            row[f"target_median_{factor}"] = float(np.median(true))
            row[f"projected_median_{factor}"] = float(np.median(pred))
            row[f"target_iqr_{factor}"] = float(
                np.quantile(true, 0.75) - np.quantile(true, 0.25)
            )
            row[f"projected_iqr_{factor}"] = float(
                np.quantile(pred, 0.75) - np.quantile(pred, 0.25)
            )
        cage_rows.append(row)
    cage_week = pd.DataFrame(cage_rows)
    for factor in FACTOR_COLUMNS:
        for statistic in ("median", "iqr"):
            true = cage_week[f"target_{statistic}_{factor}"].to_numpy(dtype=float)
            pred = cage_week[f"projected_{statistic}_{factor}"].to_numpy(dtype=float)
            metric_rows.append(
                {
                    "factor": factor,
                    "level": f"cage_week_{statistic}",
                    "subset": "all",
                    "n": int(len(cage_week)),
                    "r2": float(r2_score(true, pred)),
                    "pearson_r": float(pearsonr(true, pred).statistic),
                    "spearman_r": float(spearmanr(true, pred).statistic),
                    "mae": float(np.mean(np.abs(true - pred))),
                }
            )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(projection_dir / "latent_projection_metrics.csv", index=False)
    cage_week.to_csv(projection_dir / "latent_projection_cage_week_fidelity.csv", index=False)
    np.savez_compressed(
        projection_dir / "latent_projection_predictions.npz",
        target=target.astype(np.float32),
        prediction=prediction.astype(np.float32),
        factor_names=np.asarray(FACTORS, dtype=str),
    )

    _plot_projection_validation(target, prediction, cage_week, metrics, projection_dir)
    manifest = {
        "input_latent_cache": str(cache_path),
        "input_factor_model": str(factor_model_path),
        "n_windows": int(len(frame)),
        "projection_predictor": "frozen 64D DAE latent z",
        "projection_target": "Control-defined FA3 scores from pose-derived relation features",
        "projection_model": "StandardScaler + MLPRegressor(64,32; ReLU)",
        "control_prediction": "whole-Control-cage leave-one-out",
        "vpa_prediction": "mean across 12 Control-LOCO projection models",
        "vpa_used_for_projection_training": False,
        "coordinate_npz": str(coordinate_npz),
        "coordinate_csv": str(coordinate_csv),
    }
    (projection_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return frame, metrics, cage_week

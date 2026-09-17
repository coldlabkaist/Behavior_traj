"""core / latent_cache.

Published-analysis functions. Inputs and current sources are recorded in Final manifests.
"""
from __future__ import annotations

import re

from pathlib import Path

from typing import Any

import torch

from tqdm import tqdm

from analysis.common import build_analysis_loader, move_view

from model.stgcn import AnimalRelationBlock

def _as_list(value: Any, n: int) -> list[Any]:
    if torch.is_tensor(value):
        return value.detach().cpu().reshape(-1).tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value] * n


def _infer_sex(cage_id: str) -> str:
    matches = re.findall(r"(?:^|_)(f|m)(?:_|$)", str(cage_id), flags=re.IGNORECASE)
    return matches[-1].lower() if matches else "unknown"


def _relation_summary(
    pair_features: torch.Tensor,
    group_features: torch.Tensor,
) -> torch.Tensor:
    """Permutation-invariant window means for biological relation probes."""
    if pair_features.ndim != 4 or pair_features.shape[-1] != 8:
        raise ValueError(
            f"Expected pair features [B,T,3,8], got {tuple(pair_features.shape)}."
        )
    if group_features.ndim != 3 or group_features.shape[-1] != 3:
        raise ValueError(
            f"Expected group features [B,T,3], got {tuple(group_features.shape)}."
        )
    symmetric_pair = torch.stack(
        [
            pair_features[..., 0],
            pair_features[..., 1],
            0.5 * (pair_features[..., 2] + pair_features[..., 3]),
            0.5 * (pair_features[..., 4] + pair_features[..., 5]),
            pair_features[..., 6],
            pair_features[..., 7],
        ],
        dim=-1,
    ).mean(dim=(1, 2))
    return torch.cat([symmetric_pair, group_features.mean(dim=1)], dim=-1)


def _group_geometry_summary(
    node_position: torch.Tensor,
    *,
    body_center_idx: int,
) -> torch.Tensor:
    """Return current-loader group geometry for each 1-second window.

    The three columns are the temporal means of framewise nearest Body_C
    distance, farthest Body_C distance, and triad area. Coordinates have
    already been normalized by the cage-week representative body length in
    ``MousePoseDataset``.
    """
    if node_position.ndim != 5 or node_position.shape[2] != 3:
        raise ValueError(
            "Expected node_position [B,T,3,K,2], got "
            f"{tuple(node_position.shape)}."
        )
    centers = node_position[:, :, :, int(body_center_idx), :]
    pair_distance = torch.stack(
        [
            torch.linalg.vector_norm(centers[:, :, 0] - centers[:, :, 1], dim=-1),
            torch.linalg.vector_norm(centers[:, :, 0] - centers[:, :, 2], dim=-1),
            torch.linalg.vector_norm(centers[:, :, 1] - centers[:, :, 2], dim=-1),
        ],
        dim=-1,
    )
    nearest = pair_distance.amin(dim=-1)
    farthest = pair_distance.amax(dim=-1)
    edge_01 = centers[:, :, 1] - centers[:, :, 0]
    edge_02 = centers[:, :, 2] - centers[:, :, 0]
    area = 0.5 * torch.abs(
        edge_01[..., 0] * edge_02[..., 1]
        - edge_01[..., 1] * edge_02[..., 0]
    )
    return torch.stack(
        [nearest.mean(dim=1), farthest.mean(dim=1), area.mean(dim=1)],
        dim=-1,
    )


def extract_latent_cache(
    *,
    model,
    config: dict[str, Any],
    device: torch.device,
    cohorts: dict[str, str | Path],
    output_path: str | Path,
    batch_size: int | None = None,
) -> dict[str, Any]:
    z_chunks: list[torch.Tensor] = []
    relation_chunks: list[torch.Tensor] = []
    pose_summary_chunks: list[torch.Tensor] = []
    pair_summary_chunks: list[torch.Tensor] = []
    group_summary_chunks: list[torch.Tensor] = []
    geometry_chunks: list[torch.Tensor] = []
    group_speed_chunks: list[torch.Tensor] = []
    metadata: dict[str, list[Any]] = {
        "condition": [],
        "cage_id": [],
        "sex": [],
        "week": [],
        "file_path": [],
        "clip_start_frame_id": [],
    }

    model.eval()
    with torch.inference_mode():
        for expected_condition, data_dir in cohorts.items():
            loader = build_analysis_loader(
                config,
                data_dir=data_dir,
                batch_size=batch_size,
            )
            for _noisy, clean, batch_meta in tqdm(
                loader,
                desc=f"Extract z [{expected_condition}]",
            ):
                clean = move_view(clean, device)
                output = model(clean["node_position"], clean["group_speed"])
                z = output.get("z")
                if z is None or z.ndim != 2:
                    raise RuntimeError("Model must return one [B,D] latent tensor under key 'z'.")
                relation_mean = _relation_summary(
                    output["pair_features"], output["group_features"]
                )
                geometry_mean = _group_geometry_summary(
                    clean["node_position"],
                    body_center_idx=model.body_center_idx,
                )
                group_speed_mean = clean["group_speed"].mean(dim=(1, 2))
                if (
                    not torch.isfinite(z).all()
                    or not torch.isfinite(relation_mean).all()
                    or not torch.isfinite(geometry_mean).all()
                    or not torch.isfinite(group_speed_mean).all()
                ):
                    raise RuntimeError(
                        "Non-finite z, relation, geometry, or group-speed target encountered."
                    )

                z_chunks.append(z.detach().float().cpu())
                relation_chunks.append(relation_mean.detach().float().cpu())
                pose_summary_chunks.append(output["pose_summary"].detach().float().cpu())
                pair_summary_chunks.append(output["pair_summary"].detach().float().cpu())
                group_summary_chunks.append(output["group_summary"].detach().float().cpu())
                geometry_chunks.append(geometry_mean.detach().float().cpu())
                group_speed_chunks.append(group_speed_mean.detach().float().cpu())
                n = int(z.shape[0])
                conditions = [str(v).lower() for v in _as_list(batch_meta["condition"], n)]
                if set(conditions) != {expected_condition}:
                    raise RuntimeError(
                        f"Condition mismatch for {data_dir}: found={sorted(set(conditions))}, "
                        f"expected={expected_condition}."
                    )
                cages = [str(v) for v in _as_list(batch_meta["subject_id"], n)]
                metadata["condition"].extend(conditions)
                metadata["cage_id"].extend(cages)
                metadata["sex"].extend(_infer_sex(cage) for cage in cages)
                metadata["week"].extend(float(v) for v in _as_list(batch_meta["week_value"], n))
                metadata["file_path"].extend(str(v) for v in _as_list(batch_meta["file_path"], n))
                metadata["clip_start_frame_id"].extend(
                    int(v) for v in _as_list(batch_meta["clip_start_frame_id"], n)
                )

    payload = {
        "z": torch.cat(z_chunks, dim=0),
        "relation_mean": torch.cat(relation_chunks, dim=0),
        "relation_feature_names": [
            "body_distance_log1p",
            "nose_nose_distance_log1p",
            "bidirectional_nose_tailbase_distance_log1p",
            "mutual_facing_mean",
            "approach_rate_asinh",
            "relative_speed_log1p",
            *AnimalRelationBlock.GROUP_FEATURE_NAMES,
        ],
        "pose_summary": torch.cat(pose_summary_chunks, dim=0),
        "pair_summary": torch.cat(pair_summary_chunks, dim=0),
        "group_summary": torch.cat(group_summary_chunks, dim=0),
        "group_geometry_mean": torch.cat(geometry_chunks, dim=0),
        "group_geometry_feature_names": [
            "nearest_body_distance_bl",
            "farthest_body_distance_bl",
            "triad_area_bl2",
        ],
        "group_speed_mean": torch.cat(group_speed_chunks, dim=0),
        **metadata,
    }
    n_windows = int(payload["z"].shape[0])
    if any(len(payload[key]) != n_windows for key in metadata):
        raise RuntimeError("Latent cache metadata is not aligned with z rows.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(
        f"Saved {n_windows:,} windows, z_dim={payload['z'].shape[1]}, "
        f"relation_dim={payload['relation_mean'].shape[1]} to {output_path.resolve()}"
    )
    return payload


def load_latent_cache(path: str | Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"z", "relation_mean", "relation_feature_names", "condition", "cage_id", "week"}
    missing = sorted(required.difference(payload))
    if missing:
        raise RuntimeError(f"Latent cache is missing keys: {missing}.")
    return payload


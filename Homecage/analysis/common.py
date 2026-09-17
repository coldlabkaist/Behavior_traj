from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from datasets import MousePoseDataset
from train.checkpoint import validate_checkpoint_contract
from train.setup import DataSetup, ModelSetup


def load_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def choose_device(device: str = "") -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_frozen_model(
    config: dict[str, Any],
    checkpoint_path: str | os.PathLike[str],
    device: torch.device,
):
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    validate_checkpoint_contract(checkpoint, config)

    status = str(checkpoint.get("artifact_status", ""))
    if status != "frozen_for_continuous_latent_analysis":
        raise RuntimeError(
            "Analysis requires the frozen Stage-4 checkpoint; "
            f"found artifact_status={status!r}."
        )

    model = ModelSetup.create_model(config, device, rank=1)
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise RuntimeError(f"Missing model_state_dict in {checkpoint_path}.")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, checkpoint


def build_analysis_loader(
    config: dict[str, Any],
    *,
    data_dir: str | os.PathLike[str],
    batch_size: int | None = None,
) -> DataLoader:
    training = config.get("training", {})
    data = config.get("data", {})
    dataset = MousePoseDataset(
        data_dir=str(data_dir),
        clip_len=int(training.get("clip_len", 30)),
        clip_overlap=int(training.get("clip_overlap", 0)),
        keypoint_names=None,
        num_keypoints=None,
        config=config,
        use_augmentation=False,
        recursive=bool(data.get("recursive", False)),
    )

    animal = load_yaml("cfg/animal.yaml")
    expected_keypoints = int(animal["num_keypoints"])
    workers = int(training.get("num_workers", 0)) if os.name != "nt" else 0
    return DataLoader(
        dataset,
        batch_size=int(batch_size or training.get("batch_size", 128)),
        shuffle=False,
        num_workers=workers,
        pin_memory=False,
        collate_fn=DataSetup.make_collate_fn(expected_keypoints),
    )


def move_view(view: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "node_position": view["node_position"].to(device, non_blocking=True),
        "group_speed": view["group_speed"].to(device, non_blocking=True),
    }

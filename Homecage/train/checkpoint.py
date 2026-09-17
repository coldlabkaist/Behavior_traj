import torch
import os
import hashlib
import json
import yaml


INPUT_CONTRACT_VERSION = "relative_position_bl_plus_group_speed_pose_stability_v2"
MODEL_ARCHITECTURE_VERSION = "skeleton_pair_group_late_pool_v1"


def preprocessing_fingerprint(config):
    """Hash the resolved settings that define model-input meaning."""
    config = config or {}
    data = config.get("data", {}) or {}
    training = config.get("training", {}) or {}
    try:
        with open(os.path.join("cfg", "animal.yaml"), "r", encoding="utf-8") as f:
            animal = yaml.safe_load(f) or {}
    except OSError:
        animal = {}

    payload = {
        "input_contract": INPUT_CONTRACT_VERSION,
        "data": {
            "fps": float(data.get("fps", 30.0)),
            "score_thresh": float(data.get("score_thresh", 0.5)),
            "interpolation_max_gap": int(data.get("interpolation_max_gap", 4)),
            "roi_normalize": bool(data.get("roi_normalize", False)),
            "roi_id": str(data.get("roi_id", "floor")),
            "apply_aspect_correction": bool(data.get("apply_aspect_correction", False)),
            "frame_width": float(data.get("frame_width", 1920.0)),
            "frame_height": float(data.get("frame_height", 1080.0)),
            "pose_stability": {
                "long_missing_frames": int(data.get("interpolation_max_gap", 4)) + 1,
                "reacquisition_confirmation_frames": int(
                    data.get("interpolation_max_gap", 4)
                ) + 1,
                "trunk_ratio_range": [0.5, 2.0],
            },
        },
        "window": {
            "clip_len": int(training.get("clip_len", 30)),
            "clip_overlap": int(training.get("clip_overlap", 0)),
        },
        "animal": {
            "num_animals": int(animal.get("num_animals", 3)),
            "kpt_names": list(animal.get("kpt_names", [])),
            "body_center_keypoint": str(animal.get("body_center_keypoint", "Body_C")),
            "skeleton": list(animal.get("skeleton", [])),
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_checkpoint_contract(checkpoint, config):
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Checkpoint must be a dictionary with input-contract metadata.")
    found_contract = checkpoint.get("input_contract_version")
    if found_contract != INPUT_CONTRACT_VERSION:
        raise RuntimeError(
            "Checkpoint input contract is incompatible: "
            f"found={found_contract!r}, expected={INPUT_CONTRACT_VERSION!r}. "
            "Legacy 4-channel checkpoints cannot be loaded into this model."
        )
    found_architecture = checkpoint.get("model_architecture_version")
    if found_architecture != MODEL_ARCHITECTURE_VERSION:
        raise RuntimeError(
            "Checkpoint model architecture is incompatible: "
            f"found={found_architecture!r}, expected={MODEL_ARCHITECTURE_VERSION!r}. "
            "The current model uses separate pose, unordered-pair, and group-state "
            "temporal summaries and must be trained from scratch."
        )
    expected_fingerprint = preprocessing_fingerprint(config)
    found_fingerprint = checkpoint.get("preprocessing_fingerprint")
    if found_fingerprint != expected_fingerprint:
        raise RuntimeError(
            "Checkpoint preprocessing fingerprint does not match the current config: "
            f"found={found_fingerprint!r}, expected={expected_fingerprint!r}."
        )


class CheckpointManager:
    def __init__(
        self,
        model,
        is_multi_gpu,
        checkpoint_dir='checkpoints',
        config=None,
        split_manifest=None,
    ):
        self.model = model
        self.is_multi_gpu = is_multi_gpu
        self.checkpoint_dir = checkpoint_dir
        self.config = config or {}
        self.split_manifest = split_manifest
        if self.split_manifest is not None:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            manifest_path = os.path.abspath(
                os.path.join(self.checkpoint_dir, 'split_manifest.json')
            )
            if os.path.exists(manifest_path):
                with open(manifest_path, 'r', encoding='utf-8') as f:
                    existing_manifest = json.load(f)
                if existing_manifest != self.split_manifest:
                    raise RuntimeError(
                        "Existing split_manifest.json does not match the current "
                        "cage split. Use the matching split or a new checkpoint directory."
                    )
                print(f"Split manifest verified: {manifest_path}")
            else:
                with open(manifest_path, 'w', encoding='utf-8') as f:
                    json.dump(self.split_manifest, f, ensure_ascii=False, indent=2)
                    f.write('\n')
                print(f"Split manifest saved: {manifest_path}")

    def save_checkpoint(self, epoch, optimizer, scheduler, train_losses, 
                       val_losses, best_val_loss, config, is_best=False, best_tag=None):
        model_state_dict = (self.model.module.state_dict() if self.is_multi_gpu 
                           else self.model.state_dict())
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'config': config,
            'is_multi_gpu': self.is_multi_gpu,
            'input_contract_version': INPUT_CONTRACT_VERSION,
            'model_architecture_version': MODEL_ARCHITECTURE_VERSION,
            'preprocessing_fingerprint': preprocessing_fingerprint(config),
            'split_manifest': self.split_manifest,
        }
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.abspath(os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth'))
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        if is_best:
            best_path = os.path.abspath(os.path.join(self.checkpoint_dir, 'best_model.pth'))
            torch.save(checkpoint, best_path)
            print(f"Best checkpoint saved: {best_path}")
            if best_tag:
                safe_tag = "".join(ch if (ch.isalnum() or ch in ("_", "-")) else "_" for ch in str(best_tag))
                tagged_best_path = os.path.abspath(os.path.join(self.checkpoint_dir, f'best_{safe_tag}.pth'))
                torch.save(checkpoint, tagged_best_path)
                print(f"Best checkpoint saved: {tagged_best_path}")
            print(f"Best model saved at epoch {epoch}")

    def load_checkpoint(self, checkpoint_path, optimizer, scheduler, device):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        validate_checkpoint_contract(checkpoint, self.config)
        if self.split_manifest is not None:
            found_split_manifest = checkpoint.get('split_manifest')
            if found_split_manifest != self.split_manifest:
                raise RuntimeError(
                    "Checkpoint train/validation split does not match the current "
                    "cage split manifest."
                )
        if self.is_multi_gpu:
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        start_epoch = checkpoint['epoch']
        print(f"Loaded checkpoint from epoch {start_epoch}")
        return train_losses, val_losses, best_val_loss, start_epoch


class WeightAnomalyDetector:
    @staticmethod
    def detect_anomalies(model, step):
        eps = 1e-8
        model_to_check = model.module if hasattr(model, "module") else model
        for name, p in model_to_check.named_parameters():
            if torch.isnan(p).any() or torch.isinf(p).any():
                print(f"[Anomaly NAN/INF][step {step}] {name}")
            elif ".bias" not in name and p.data.abs().sum() < eps:
                print(f"[Anomaly ZERO][step {step}] {name}")

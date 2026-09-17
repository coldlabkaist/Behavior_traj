"""PyTorch dataset: accepted 30-frame model inputs, augmentation and metadata."""
from typing import Dict
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from datasets.pose_io import PoseReader


class MousePoseDataset(PoseReader, Dataset):
    """Use shared recording preprocessing and retain the original training contract."""

    def __init__(
        self,
        data_dir='data',
        clip_len=30,
        clip_overlap=0,
        score_thresh=0.5,
        keypoint_names=None,
        num_keypoints=None,
        config=None,
        use_augmentation=True,
        recursive=True,
    ):
        self.data_dir = data_dir
        self.clip_len = int(clip_len)
        self.clip_overlap = int(clip_overlap)
        if self.clip_len != 30 or self.clip_overlap != 0:
            raise ValueError(
                "MousePoseDataset now uses non-overlapping 30-frame windows only; "
                f"got clip_len={self.clip_len}, clip_overlap={self.clip_overlap}."
            )
        self.use_augmentation = use_augmentation
        self.recursive = bool(recursive)

        self._configure_pose_preprocessing(
            config=config,
            score_thresh=score_thresh,
            keypoint_names=keypoint_names,
            num_keypoints=num_keypoints,
        )

        self.data_files = self._get_all_filepaths(data_dir, recursive=self.recursive)
        rank = int(os.environ.get('RANK', 0))
        if rank == 0:
            print(f"Found {len(self.data_files)} data files")
            if self.roi_normalize:
                print(
                    f"ROI normalization enabled: roi_dir={self.roi_dir}, "
                    f"roi_id={self.roi_id}, roi_files={len(set(self._roi_index.values()))}"
                )
        self.processed_samples = []
        self.processed_observed_masks = []
        self.processed_interpolated_masks = []
        self.clip_to_file_mapping = []
        self.clip_to_condition_mapping = []
        self.clip_to_week_mapping = []
        self.clip_to_subject_mapping = []
        self.clip_to_start_frame_mapping = []
        self.clip_to_start_frame_id_mapping = []
        self.clip_to_body_scale_mapping = []
        self.clip_to_animal_body_scales_mapping = []
        self.clip_to_body_scale_qc_flag_mapping = []
        self.file_qc_summary = pd.DataFrame()
        self._load_and_process_data()

        if self.use_augmentation:
            from train.augmentation import PoseAugmentation
            self.augmentation1 = PoseAugmentation(config)
        else:
            self.augmentation1 = None

    def set_epoch(self, epoch):
        if self.augmentation1 is not None and hasattr(self.augmentation1, 'set_epoch'):
            self.augmentation1.set_epoch(epoch)

    def _center_node_positions(self, pos_tensor: torch.Tensor) -> torch.Tensor:
        """Re-enforce Body_C centroid zero after position augmentation."""
        bc = int(max(0, min(self.num_keypoints - 1, self.body_center_idx)))
        centers = pos_tensor[:, :, bc, :]
        centroid = centers.mean(dim=1, keepdim=True).unsqueeze(2)
        return pos_tensor - centroid

    def _load_and_process_data(self):
        skipped_short = 0
        skipped_nan = 0
        skipped_pose_stability = 0
        kept = 0
        file_qc_rows = []
        rank = int(os.environ.get('RANK', 0))
        for file_name in self.data_files:
            try:
                sequence = self.load_pose_sequence(file_name)
                file_qc = {
                    'file_path': str(file_name),
                    'condition': str(sequence['condition']),
                    'cage_id': str(sequence['cage_id']),
                    'week': float(sequence['week']),
                    'timeline_frames': int(len(sequence['frame_ids'])),
                    'candidate_windows': 0,
                    'kept_windows': 0,
                    'kept_windows_with_interpolation': 0,
                    'excluded_pose_stability_windows': 0,
                    'windows_with_long_missing': 0,
                    'windows_with_reacquisition_invalid': 0,
                    'excluded_remaining_nan_windows': 0,
                    'short_recording': False,
                }
                if rank == 0:
                    print(f"Loading {file_name}: {len(sequence['frame_ids'])} timeline frames")
                frame_ids = np.asarray(sequence['frame_ids'], dtype=np.int64)
                all_coords = np.asarray(sequence['coords'], dtype=np.float64)
                observed_keypoint_mask = np.asarray(sequence['observed_mask'], dtype=bool)
                interpolated_keypoint_mask = np.asarray(sequence['interpolated_mask'], dtype=bool)
                pose_long_missing = np.asarray(sequence['pose_long_missing'], dtype=bool)
                pose_reacquisition_invalid = np.asarray(
                    sequence['pose_reacquisition_invalid'], dtype=bool
                )
                pose_stability_invalid = np.asarray(
                    sequence['pose_stability_invalid'], dtype=bool
                )
                body_scale = float(sequence['body_scale'])
                animal_body_scales = np.asarray(
                    sequence['animal_body_scales'], dtype=np.float64
                )
                all_model_inputs = self._compute_model_inputs(all_coords, body_scale)
                total_frames = int(frame_ids.size)
                if total_frames < self.clip_len:
                    skipped_short += 1
                    file_qc['short_recording'] = True
                    file_qc_rows.append(file_qc)
                    continue
                step_size = max(1, self.clip_len - self.clip_overlap)
                for start_idx in range(0, total_frames - self.clip_len + 1, step_size):
                    file_qc['candidate_windows'] += 1
                    stop_idx = start_idx + self.clip_len
                    has_long_missing = bool(
                        pose_long_missing[start_idx:stop_idx].any()
                    )
                    has_reacquisition_invalid = bool(
                        pose_reacquisition_invalid[start_idx:stop_idx].any()
                    )
                    if has_long_missing:
                        file_qc['windows_with_long_missing'] += 1
                    if has_reacquisition_invalid:
                        file_qc['windows_with_reacquisition_invalid'] += 1
                    if bool(pose_stability_invalid[start_idx:stop_idx].any()):
                        skipped_pose_stability += 1
                        file_qc['excluded_pose_stability_windows'] += 1
                        continue
                    sample = {
                        key: values[start_idx:stop_idx]
                        for key, values in all_model_inputs.items()
                    }
                    if not all(np.isfinite(values).all() for values in sample.values()):
                        skipped_nan += 1
                        file_qc['excluded_remaining_nan_windows'] += 1
                        continue
                    observed_window = observed_keypoint_mask[
                        start_idx:start_idx + self.clip_len
                    ]
                    interpolated_window = interpolated_keypoint_mask[
                        start_idx:start_idx + self.clip_len
                    ]
                    self.processed_samples.append(sample)
                    self.processed_observed_masks.append(observed_window)
                    self.processed_interpolated_masks.append(interpolated_window)
                    self.clip_to_file_mapping.append(file_name)
                    self.clip_to_condition_mapping.append(str(sequence['condition']))
                    self.clip_to_week_mapping.append(float(sequence['week']))
                    self.clip_to_subject_mapping.append(str(sequence['cage_id']))
                    self.clip_to_start_frame_mapping.append(int(start_idx))
                    self.clip_to_start_frame_id_mapping.append(int(frame_ids[start_idx]))
                    self.clip_to_body_scale_mapping.append(body_scale)
                    self.clip_to_animal_body_scales_mapping.append(animal_body_scales.copy())
                    self.clip_to_body_scale_qc_flag_mapping.append(
                        bool(sequence['body_scale_qc_flag'])
                    )
                    kept += 1
                    file_qc['kept_windows'] += 1
                    if bool(interpolated_window.any()):
                        file_qc['kept_windows_with_interpolation'] += 1
                file_qc_rows.append(file_qc)
            except Exception as e:
                if rank == 0:
                    print(f"Error loading {file_name}: {e}")
                raise RuntimeError(f"Failed to load pose CSV: {file_name}") from e
        self.file_qc_summary = pd.DataFrame(file_qc_rows)
        if rank == 0:
            print(
                f"Total processed multi-animal clips: {len(self.processed_samples)} "
                f"(kept={kept}, skipped_short={skipped_short}, "
                f"skipped_pose_stability={skipped_pose_stability}, "
                f"skipped_nan={skipped_nan}"
                ")"
            )
            if not self.file_qc_summary.empty:
                count_columns = [
                    'candidate_windows',
                    'kept_windows',
                    'kept_windows_with_interpolation',
                    'excluded_pose_stability_windows',
                    'windows_with_long_missing',
                    'windows_with_reacquisition_invalid',
                    'excluded_remaining_nan_windows',
                ]
                grouped_qc = (
                    self.file_qc_summary
                    .groupby(['condition', 'week'], dropna=False)[count_columns]
                    .sum()
                    .reset_index()
                )
                print("Clip QC by condition/week (cage-level rows are in file_qc_summary):")
                print(grouped_qc.to_string(index=False))

    def __len__(self):
        return len(self.processed_samples)

    @staticmethod
    def _tensorize_sample(sample: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        return {
            "node_position": torch.from_numpy(sample["node_position"]).float(),
            "group_speed": torch.from_numpy(sample["group_speed"]).float(),
        }

    def get_item(self, idx, use_augmentation=None):
        clean_view = self._tensorize_sample(self.processed_samples[idx])
        aug_enabled = self.use_augmentation if use_augmentation is None else bool(use_augmentation)

        if aug_enabled:
            augmented_position = self.augmentation1(clean_view["node_position"])
            augmented_position = self._center_node_positions(augmented_position)
            view1 = {
                "node_position": augmented_position,
                "group_speed": clean_view["group_speed"].clone(),
            }
        else:
            view1 = clean_view

        batch_meta = {
            "file_path": self.clip_to_file_mapping[idx],
            "condition": self.clip_to_condition_mapping[idx],
            "week_value": float(self.clip_to_week_mapping[idx]),
            "subject_id": self.clip_to_subject_mapping[idx],
            "clip_start_frame": int(self.clip_to_start_frame_mapping[idx]),
            "clip_start_frame_id": int(self.clip_to_start_frame_id_mapping[idx]),
            "pose_observed_mask": torch.as_tensor(
                self.processed_observed_masks[idx], dtype=torch.bool
            ),
            "pose_interpolated_mask": torch.as_tensor(
                self.processed_interpolated_masks[idx], dtype=torch.bool
            ),
            "recording_body_scale": torch.as_tensor(
                self.clip_to_body_scale_mapping[idx], dtype=torch.float32
            ),
            "animal_body_scales": torch.as_tensor(
                self.clip_to_animal_body_scales_mapping[idx], dtype=torch.float32
            ),
            "body_scale_qc_flag": torch.as_tensor(
                self.clip_to_body_scale_qc_flag_mapping[idx], dtype=torch.bool
            ),
            "fps": torch.as_tensor(self.fps, dtype=torch.float32),
        }
        return view1, clean_view, batch_meta

    def __getitem__(self, idx):
        return self.get_item(idx)

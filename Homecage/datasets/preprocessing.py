"""Shared pose quality checks and model-input transformations.

PosePreprocessor methods use the recording settings configured by PoseReader.
They do not discover files or construct training samples.
"""
from typing import Dict
import numpy as np


class PosePreprocessor:
    """Numerical preprocessing shared by recording and model loaders."""

    def _compute_model_inputs(
        self,
        coords: np.ndarray,
        body_scale: float,
    ) -> Dict[str, np.ndarray]:
        """Build complete-recording model inputs before clip extraction.

        node_position contains body-length-normalized coordinates relative to
        the current three-animal Body_C centroid. group_speed preserves the
        translation removed by centering, in body lengths per second.
        """
        coords = np.asarray(coords, dtype=np.float64)
        expected_shape = (self.num_animals, self.num_keypoints, 2)
        if coords.ndim != 4 or tuple(coords.shape[1:]) != expected_shape:
            raise ValueError(
                "coords must have shape "
                f"[frame,{self.num_animals},{self.num_keypoints},2], got {coords.shape}"
            )
        if not np.isfinite(body_scale) or body_scale <= 1e-12:
            raise ValueError(f"body_scale must be finite and positive, got {body_scale}")

        normalized = coords / float(body_scale)
        bc = int(max(0, min(self.num_keypoints - 1, self.body_center_idx)))
        group_centroid = normalized[:, :, bc, :].mean(axis=1)
        node_position = normalized - group_centroid[:, None, None, :]
        group_speed = self._central_difference_speed(group_centroid)

        return {
            "node_position": node_position.astype(np.float32, copy=False),
            "group_speed": group_speed[:, None].astype(np.float32, copy=False),
        }

    def _central_difference_speed(self, group_centroid: np.ndarray) -> np.ndarray:
        """Return full-timeline centroid speed without spreading missingness.

        A centered difference is used when both neighboring frames are valid.
        At recording edges, or next to an unresolved pose gap, the available
        one-sided difference is used. Frames whose own three-animal centroid is
        unavailable remain NaN, so their 30-frame windows are rejected later.
        """
        group_centroid = np.asarray(group_centroid, dtype=np.float64)
        if group_centroid.ndim != 2 or group_centroid.shape[1] != 2:
            raise ValueError(
                f"group_centroid must have shape [frame,2], got {group_centroid.shape}"
            )

        frame_count = int(group_centroid.shape[0])
        speed = np.full(frame_count, np.nan, dtype=np.float64)
        if frame_count == 0:
            return speed
        valid = np.isfinite(group_centroid).all(axis=1)
        for frame_idx in np.flatnonzero(valid):
            has_previous = frame_idx > 0 and valid[frame_idx - 1]
            has_next = frame_idx + 1 < frame_count and valid[frame_idx + 1]
            if has_previous and has_next:
                displacement = group_centroid[frame_idx + 1] - group_centroid[frame_idx - 1]
                speed[frame_idx] = np.linalg.norm(displacement) * (self.fps / 2.0)
            elif has_next:
                displacement = group_centroid[frame_idx + 1] - group_centroid[frame_idx]
                speed[frame_idx] = np.linalg.norm(displacement) * self.fps
            elif has_previous:
                displacement = group_centroid[frame_idx] - group_centroid[frame_idx - 1]
                speed[frame_idx] = np.linalg.norm(displacement) * self.fps
        return speed

    @staticmethod
    def _interpolate_short_internal_gaps(
        coords: np.ndarray,
        max_gap: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Linearly fill only fully bounded missing runs no longer than ``max_gap``.

        Parameters
        ----------
        coords
            Coordinate array with shape [animal, frame, coordinate].
        max_gap
            Maximum consecutive missing frames that may be interpolated.

        Returns
        -------
        filled, observed_mask, interpolated_mask
            Arrays with the same shape as ``coords``. Leading/trailing gaps and
            runs longer than ``max_gap`` remain NaN.
        """
        filled = np.asarray(coords, dtype=np.float64).copy()
        observed = np.isfinite(filled)
        interpolated = np.zeros_like(observed, dtype=bool)
        max_gap = max(0, int(max_gap))
        if max_gap == 0 or filled.shape[1] == 0:
            return filled, observed, interpolated

        n_frames = int(filled.shape[1])
        for animal_idx in range(filled.shape[0]):
            for coord_idx in range(filled.shape[2]):
                values = filled[animal_idx, :, coord_idx]
                missing = ~np.isfinite(values)
                if not np.any(missing):
                    continue
                padded = np.r_[False, missing, False].astype(np.int8)
                edges = np.flatnonzero(np.diff(padded))
                for start, stop in zip(edges[::2], edges[1::2]):
                    gap_len = int(stop - start)
                    if (
                        gap_len > max_gap
                        or start == 0
                        or stop >= n_frames
                        or not np.isfinite(values[start - 1])
                        or not np.isfinite(values[stop])
                    ):
                        continue
                    left = float(values[start - 1])
                    right = float(values[stop])
                    weights = np.arange(1, gap_len + 1, dtype=np.float64) / float(gap_len + 1)
                    values[start:stop] = left + (right - left) * weights
                    interpolated[animal_idx, start:stop, coord_idx] = True

        return filled, observed, interpolated

    def _compute_pose_stability_masks(
        self,
        coords: np.ndarray,
        observed_keypoints: np.ndarray,
        animal_body_scales: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Flag long disappearance and unstable pose reacquisition.

        A disappearance starts when Body_C is absent or fewer than four
        keypoints are observed for at least ``interpolation_max_gap + 1``
        consecutive frames.  After such a run, the animal remains invalid
        until five consecutive frames have observed Neck/Body_C/Tail geometry
        between 0.5x and 2.0x its recording-level projected body scale.

        The broad geometry bounds are used only after a long disappearance;
        ordinary wall climbing and peripheral keypoints outside the ROI are not
        filtered.
        """
        coords = np.asarray(coords, dtype=np.float64)
        observed_keypoints = np.asarray(observed_keypoints, dtype=bool)
        animal_body_scales = np.asarray(animal_body_scales, dtype=np.float64)
        expected_coords = (coords.shape[0], self.num_animals, self.num_keypoints, 2)
        expected_mask = (coords.shape[0], self.num_animals, self.num_keypoints)
        if coords.shape != expected_coords:
            raise ValueError(f"coords must have shape {expected_coords}, got {coords.shape}")
        if observed_keypoints.shape != expected_mask:
            raise ValueError(
                f"observed_keypoints must have shape {expected_mask}, "
                f"got {observed_keypoints.shape}"
            )
        if animal_body_scales.shape != (self.num_animals,):
            raise ValueError(
                "animal_body_scales must have one value per animal, got "
                f"{animal_body_scales.shape}"
            )

        required = ('Neck', 'Body_C', 'Tail')
        missing = [name for name in required if name not in self.keypoint_names]
        if missing:
            raise ValueError(f"Pose stability requires keypoints: {missing}")
        neck_idx = int(self.keypoint_names.index('Neck'))
        center_idx = int(self.keypoint_names.index('Body_C'))
        tail_idx = int(self.keypoint_names.index('Tail'))

        valid_count = observed_keypoints.sum(axis=-1)
        visible = observed_keypoints[:, :, center_idx] & (valid_count >= 4)
        core_observed = (
            observed_keypoints[:, :, neck_idx]
            & observed_keypoints[:, :, center_idx]
            & observed_keypoints[:, :, tail_idx]
        )
        trunk_length = (
            np.linalg.norm(
                coords[:, :, neck_idx, :] - coords[:, :, center_idx, :], axis=-1
            )
            + np.linalg.norm(
                coords[:, :, center_idx, :] - coords[:, :, tail_idx, :], axis=-1
            )
        )
        ratio = np.divide(
            trunk_length,
            animal_body_scales[None, :],
            out=np.full_like(trunk_length, np.nan, dtype=np.float64),
            where=animal_body_scales[None, :] > 1e-12,
        )
        stable = (
            core_observed
            & (valid_count >= 4)
            & np.isfinite(ratio)
            & (ratio >= 0.5)
            & (ratio <= 2.0)
        )

        n_frames = int(coords.shape[0])
        confirmation_frames = max(1, int(self.interpolation_max_gap) + 1)
        long_missing = np.zeros((n_frames, self.num_animals), dtype=bool)
        reacquisition = np.zeros_like(long_missing)

        for animal_idx in range(self.num_animals):
            missing_frame = ~visible[:, animal_idx]
            edges = np.flatnonzero(
                np.diff(np.r_[False, missing_frame, False].astype(np.int8))
            )
            long_runs = [
                (int(start), int(stop))
                for start, stop in zip(edges[::2], edges[1::2])
                if int(stop - start) >= confirmation_frames
            ]
            for start, stop in long_runs:
                long_missing[start:stop, animal_idx] = True
                stable_end = None
                latest_start = n_frames - confirmation_frames
                for candidate in range(stop, latest_start + 1):
                    if stable[
                        candidate:candidate + confirmation_frames, animal_idx
                    ].all():
                        stable_end = candidate + confirmation_frames
                        break
                if stable_end is None:
                    reacquisition[stop:, animal_idx] = True
                else:
                    reacquisition[stop:stable_end, animal_idx] = True

        return {
            'long_missing': long_missing,
            'reacquisition': reacquisition,
            'invalid': long_missing | reacquisition,
        }

    def _compute_recording_body_scale(
        self,
        coords: np.ndarray,
    ) -> dict:
        """Compute one fixed projected-trunk scale for a complete recording.

        A temporal median is computed separately for each animal from
        ``Neck-Body_C + Body_C-Tail``.  The recording scale is the median of
        those animal medians, so every animal contributes equally regardless of
        how many valid frames it has. Bounded-interpolated values remain
        eligible, while pose-stability invalid frames are NaN before the final
        scale is calculated.
        """
        coords = np.asarray(coords, dtype=np.float64)
        if (
            coords.ndim != 4
            or coords.shape[1:] != (self.num_animals, self.num_keypoints, 2)
        ):
            raise ValueError(
                "coords must have shape "
                f"[frame,{self.num_animals},{self.num_keypoints},2], got {coords.shape}"
            )

        required = ('Neck', 'Body_C', 'Tail')
        missing = [name for name in required if name not in self.keypoint_names]
        if missing:
            raise ValueError(f"Body-length normalization requires keypoints: {missing}")
        neck_idx = int(self.keypoint_names.index('Neck'))
        center_idx = int(self.keypoint_names.index('Body_C'))
        tail_idx = int(self.keypoint_names.index('Tail'))

        neck = coords[:, :, neck_idx, :]
        center = coords[:, :, center_idx, :]
        tail = coords[:, :, tail_idx, :]
        lengths = (
            np.linalg.norm(neck - center, axis=-1)
            + np.linalg.norm(center - tail, axis=-1)
        )
        usable = np.isfinite(lengths) & (lengths > 1e-12)

        animal_medians = np.full(self.num_animals, np.nan, dtype=np.float64)
        valid_counts = usable.sum(axis=0).astype(np.int64)
        for animal_idx in range(self.num_animals):
            values = lengths[usable[:, animal_idx], animal_idx]
            if values.size:
                animal_medians[animal_idx] = float(np.median(values))

        invalid_animals = np.flatnonzero(
            ~np.isfinite(animal_medians) | (animal_medians <= 1e-12)
        )
        if invalid_animals.size:
            raise ValueError(
                "No valid Neck-Body_C-Tail scale for animal indices "
                f"{invalid_animals.tolist()}"
            )

        body_scale = float(np.median(animal_medians))
        pooled_values = lengths[usable]
        if not np.isfinite(body_scale) or body_scale <= 1e-12 or pooled_values.size == 0:
            raise ValueError("Recording body scale is not finite and positive")
        animal_ratio = float(np.max(animal_medians) / np.min(animal_medians))
        return {
            'body_scale': body_scale,
            'animal_body_scales': animal_medians,
            'body_scale_valid_counts': valid_counts,
            'body_scale_pooled_median': float(np.median(pooled_values)),
            'body_scale_q10': float(np.quantile(pooled_values, 0.10)),
            'body_scale_q90': float(np.quantile(pooled_values, 0.90)),
            'body_scale_animal_ratio': animal_ratio,
            'body_scale_qc_flag': bool(animal_ratio > 1.20),
        }

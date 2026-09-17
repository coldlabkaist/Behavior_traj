"""Read complete pose recordings with shared ROI, timeline and quality processing."""
from typing import Dict, List, Optional
import os
import re
import numpy as np
import pandas as pd
import yaml
from datasets.preprocessing import PosePreprocessor
from datasets.paths import resolve_data_path


class PoseReader(PosePreprocessor):
    """Return full recordings, preserving invalid frames as NaN and quality masks.

    No training-window exclusions, augmentation, body scaling or centering are
    applied to returned coordinates. ROI/aspect correction, short-gap filling
    and pose-stability masking follow the training preprocessing exactly.
    """

    def __init__(
        self,
        data_dir='data',
        *,
        config=None,
        score_thresh=0.5,
        keypoint_names=None,
        num_keypoints=None,
        recursive=True,
    ):
        self.data_dir = str(data_dir)
        self.recursive = bool(recursive)
        self._configure_pose_preprocessing(
            config=config,
            score_thresh=score_thresh,
            keypoint_names=keypoint_names,
            num_keypoints=num_keypoints,
        )
        self.data_files = self._get_all_filepaths(
            self.data_dir,
            recursive=self.recursive,
        )

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        return self.load_pose_sequence(self.data_files[int(idx)])

    def iter_sequences(self):
        for file_name in self.data_files:
            yield self.load_pose_sequence(file_name)

    @staticmethod
    def iter_windows(
        sequence: dict,
        *,
        window_len: int = 30,
        stride: Optional[int] = None,
    ):
        """Yield every complete time window without applying a QC exclusion."""
        window_len = int(window_len)
        if window_len <= 0:
            raise ValueError('window_len must be positive')
        stride = window_len if stride is None else int(stride)
        if stride <= 0:
            raise ValueError('stride must be positive')

        n_frames = int(np.asarray(sequence['frame_ids']).shape[0])
        keys_to_slice = (
            'frame_ids',
            'frame_present_mask',
            'coords',
            'confidence',
            'observed_mask',
            'interpolated_mask',
            'pose_long_missing',
            'pose_reacquisition_invalid',
            'pose_stability_invalid',
        )
        window_index = 0
        for start in range(0, n_frames - window_len + 1, stride):
            stop = start + window_len
            window = {
                'file_path': sequence['file_path'],
                'file_name': sequence['file_name'],
                'condition': sequence.get('condition'),
                'cage_id': sequence['cage_id'],
                'sex': sequence['sex'],
                'week': sequence['week'],
                'pnd': sequence['pnd'],
                'window_index': int(window_index),
                'window_start': int(start),
                'window_stop': int(stop),
                'window_start_frame_id': int(sequence['frame_ids'][start]),
            }
            for key in keys_to_slice:
                window[key] = np.asarray(sequence[key])[start:stop]
            window['frame_complete'] = bool(window['frame_present_mask'].all())
            window['pose_complete'] = bool(np.isfinite(window['coords']).all())
            window['interpolated_fraction'] = float(window['interpolated_mask'].mean())
            window['pose_stability_invalid_fraction'] = float(window['pose_stability_invalid'].mean())
            yield window
            window_index += 1

    def _configure_pose_preprocessing(
        self,
        *,
        config=None,
        score_thresh=0.5,
        keypoint_names=None,
        num_keypoints=None,
    ):
        """Configure preprocessing shared by model and EDA loaders."""

        self.animal_cfg = None
        try:
            with open(os.path.join('cfg', 'animal.yaml'), 'r', encoding='utf-8') as f:
                self.animal_cfg = yaml.safe_load(f)
        except Exception:
            self.animal_cfg = None

        if keypoint_names is None:
            if self.animal_cfg:
                self.keypoint_names = self.animal_cfg.get('kpt_names', None)
            else:
                raise ValueError('keypoint_names must be provided from config!')
        else:
            self.keypoint_names = keypoint_names

        if num_keypoints is None:
            if self.animal_cfg:
                self.num_keypoints = self.animal_cfg.get('num_keypoints', len(self.keypoint_names))
            else:
                self.num_keypoints = len(self.keypoint_names)
        else:
            self.num_keypoints = num_keypoints
        self.num_animals = self.animal_cfg.get('num_animals', 3) if self.animal_cfg else 3
        self.animal_track_ids = None
        if self.animal_cfg:
            self.animal_track_ids = self.animal_cfg.get('animal_track_ids', None)
        body_center_name = self.animal_cfg.get('body_center_keypoint', 'Body_C') if self.animal_cfg else 'Body_C'
        if self.keypoint_names and body_center_name in self.keypoint_names:
            self.body_center_idx = int(self.keypoint_names.index(body_center_name))
        else:
            self.body_center_idx = int(self.num_keypoints // 2)
        data_cfg = (config or {}).get('data', {})
        self.fps = float(data_cfg.get('fps', 30.0))
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError(f"data.fps must be finite and positive, got {self.fps}")
        self.score_thresh = float(data_cfg.get('score_thresh', score_thresh))
        self.interpolation_max_gap = max(
            0,
            int(data_cfg.get('interpolation_max_gap', 4)),
        )
        self.apply_aspect_correction = bool(data_cfg.get('apply_aspect_correction', False))
        self.frame_width = float(data_cfg.get('frame_width', 1920.0))
        self.frame_height = float(data_cfg.get('frame_height', 1080.0))
        if self.frame_height <= 0:
            self.frame_height = 1080.0
        self.aspect_scale_x = float(self.frame_width / self.frame_height)
        self.roi_normalize = bool(data_cfg.get('roi_normalize', data_cfg.get('use_roi_normalization', False)))
        self.roi_dir = str(data_cfg.get('roi_dir', 'data/roi'))
        self.roi_id = str(data_cfg.get('roi_id', 'floor')).lower().strip()
        self.roi_missing = str(data_cfg.get('roi_missing', 'warn')).lower().strip()
        if self.roi_missing not in {'warn', 'skip', 'error', 'ignore'}:
            self.roi_missing = 'warn'
        self._roi_index = self._build_roi_index(self.roi_dir) if self.roi_normalize else {}
        self._roi_transform_cache: Dict[str, Optional[np.ndarray]] = {}

    def _get_all_filepaths(self, data_dir: str, recursive: bool = True) -> List[str]:
        import glob
        physical = str(resolve_data_path(data_dir))
        pattern = os.path.join(physical, '**', '*.csv') if recursive else os.path.join(physical, '*.csv')
        # Keep the original file identifiers/order in checkpoints and latent metadata.
        return sorted(os.path.join(str(data_dir), os.path.relpath(p, physical))
                      for p in glob.glob(pattern, recursive=bool(recursive))
                      if not os.path.basename(p).endswith('_pins.csv'))

    def _canonical_roi_key(self, stem: str) -> str:
        stem = re.sub(r'_pins$', '', str(stem), flags=re.IGNORECASE)
        tokens = []
        for token in stem.lower().split('_'):
            token = token.strip()
            if not token or token in {'first', 'last'}:
                continue
            if re.match(r'^\d+w$', token):
                continue
            m = re.match(r'^pn(?:d)?(\d+)$', token)
            if m:
                token = f"pnd{m.group(1)}"
            tokens.append(token)
        return '|'.join(sorted(tokens))

    def _build_roi_index(self, roi_dir: str) -> Dict[str, str]:
        import glob
        paths = sorted(glob.glob(os.path.join(str(resolve_data_path(roi_dir)), '*_pins.csv')))
        raw: Dict[str, Optional[str]] = {}
        for path in paths:
            stem = os.path.splitext(os.path.basename(path))[0]
            base_stem = re.sub(r'_pins$', '', stem, flags=re.IGNORECASE)
            keys = [
                f"exact:{base_stem.lower()}",
                f"canon:{self._canonical_roi_key(base_stem)}",
            ]
            for key in keys:
                prev = raw.get(key)
                if prev is None and key in raw:
                    continue
                if prev is not None and prev != path:
                    raw[key] = None
                else:
                    raw[key] = path
        return {key: path for key, path in raw.items() if path is not None}

    def _roi_path_for_file(self, file_name: str) -> Optional[str]:
        stem = os.path.splitext(os.path.basename(str(file_name)))[0]
        keys = [
            f"exact:{stem.lower()}",
            f"canon:{self._canonical_roi_key(stem)}",
        ]
        for key in keys:
            path = self._roi_index.get(key)
            if path:
                return path
        return None

    def _load_roi_transform(self, file_name: str) -> Optional[np.ndarray]:
        if not self.roi_normalize:
            return None
        cache_key = os.path.abspath(str(file_name))
        if cache_key in self._roi_transform_cache:
            return self._roi_transform_cache[cache_key]

        roi_path = self._roi_path_for_file(file_name)
        rank = int(os.environ.get('RANK', 0))
        if roi_path is None:
            msg = f"ROI pins not found for {file_name}"
            if self.roi_missing == 'error':
                raise FileNotFoundError(msg)
            if self.roi_missing == 'warn' and rank == 0:
                print(f"Warning: {msg}; using full-frame coordinates.")
            self._roi_transform_cache[cache_key] = None
            return None

        roi_df = pd.read_csv(roi_path)
        if 'id' not in roi_df.columns:
            msg = f"ROI file missing id column: {roi_path}"
            if self.roi_missing == 'error':
                raise ValueError(msg)
            if self.roi_missing == 'warn' and rank == 0:
                print(f"Warning: {msg}; using full-frame coordinates.")
            self._roi_transform_cache[cache_key] = None
            return None

        sub = roi_df.loc[roi_df['id'].astype(str).str.lower() == self.roi_id].copy()
        if sub.shape[0] < 4:
            msg = f"ROI id `{self.roi_id}` has fewer than 4 points in {roi_path}"
            if self.roi_missing == 'error':
                raise ValueError(msg)
            if self.roi_missing == 'warn' and rank == 0:
                print(f"Warning: {msg}; using full-frame coordinates.")
            self._roi_transform_cache[cache_key] = None
            return None

        if {'x_norm', 'y_norm'}.issubset(sub.columns):
            pts = sub[['x_norm', 'y_norm']].to_numpy(dtype=np.float64)[:4]
        elif {'x', 'y'}.issubset(sub.columns):
            pts = sub[['x', 'y']].to_numpy(dtype=np.float64)[:4]
            pts[:, 0] = pts[:, 0] / max(float(self.frame_width), 1.0)
            pts[:, 1] = pts[:, 1] / max(float(self.frame_height), 1.0)
        else:
            msg = f"ROI file missing coordinate columns: {roi_path}"
            if self.roi_missing == 'error':
                raise ValueError(msg)
            if self.roi_missing == 'warn' and rank == 0:
                print(f"Warning: {msg}; using full-frame coordinates.")
            self._roi_transform_cache[cache_key] = None
            return None

        dst = np.asarray(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [1.0, 0.0],
            ],
            dtype=np.float64,
        )
        h = self._homography(pts, dst)
        self._roi_transform_cache[cache_key] = h
        return h

    @staticmethod
    def _homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        src = np.asarray(src, dtype=np.float64)
        dst = np.asarray(dst, dtype=np.float64)
        if src.shape != (4, 2) or dst.shape != (4, 2):
            raise ValueError("homography requires src and dst arrays with shape (4, 2)")
        rows = []
        for (x, y), (u, v) in zip(src, dst):
            rows.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
            rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])
        a = np.asarray(rows, dtype=np.float64)
        _, _, vt = np.linalg.svd(a)
        h = vt[-1].reshape(3, 3)
        if abs(h[2, 2]) > 1e-12:
            h = h / h[2, 2]
        return h

    def _apply_roi_transform(self, coords: np.ndarray, h: np.ndarray) -> np.ndarray:
        out = np.asarray(coords, dtype=np.float64).copy()
        x = out[:, 0::2]
        y = out[:, 1::2]
        xy = np.stack([x.reshape(-1), y.reshape(-1)], axis=1)
        valid = np.isfinite(xy).all(axis=1)
        if not np.any(valid):
            return out

        homo = np.ones((xy.shape[0], 3), dtype=np.float64)
        homo[:, :2] = xy
        proj = homo @ h.T
        denom = proj[:, 2]
        valid = valid & (np.abs(denom) > 1e-12)
        xy_out = xy.copy()
        xy_out[valid, 0] = proj[valid, 0] / denom[valid]
        xy_out[valid, 1] = proj[valid, 1] / denom[valid]
        x[:, :] = xy_out[:, 0].reshape(x.shape)
        y[:, :] = xy_out[:, 1].reshape(y.shape)
        out[:, 0::2] = x
        out[:, 1::2] = y
        return out

    def _extract_week_value(self, file_name: str) -> float:
        stem = os.path.basename(file_name)
        m = re.search(r'_(\d+)w(?:_|\.|$)', stem, flags=re.IGNORECASE)
        if m:
            return float(m.group(1))
        m = re.search(r'_PND?(\d+)(?:_|\.|$)', stem, flags=re.IGNORECASE)
        if not m:
            return float('nan')
        pnd_to_week = {
            23: 3,
            29: 4,
            30: 4,
            36: 5,
            37: 5,
            44: 6,
            51: 7,
            52: 7,
            53: 7,
            58: 8,
        }
        return float(pnd_to_week.get(int(m.group(1)), float('nan')))

    def _extract_subject_id(self, file_name: str) -> str:
        stem = os.path.splitext(os.path.basename(file_name))[0]
        m = re.match(
            r"^(B6_VPA(?:_\d+)+_[fm]_\d+)_(cont|exp)_(?:PND?\d+)(?:_\d+w)?(?:_.+)?$",
            stem,
            flags=re.IGNORECASE,
        )
        if m:
            return f"{m.group(1)}_{m.group(2).lower()}"
        m = re.match(
            r"^(B6_VPA(?:_\d+)+_[fm]_\d+)_(?:PND?\d+)_(cont|exp)(?:_\d+w)?(?:_.+)?$",
            stem,
            flags=re.IGNORECASE,
        )
        if m:
            return f"{m.group(1)}_{m.group(2).lower()}"
        m = re.match(r"^\d+_\d+w_(B6_HC_[fm]_\d+)$", stem, flags=re.IGNORECASE)
        if m:
            return str(m.group(1))
        m = re.match(r"^\d+_\d+w_(\d+)_(f|m)(?:_.+)?$", stem, flags=re.IGNORECASE)
        if m:
            return f"{m.group(1)}_{m.group(2).lower()}"
        return stem

    @staticmethod
    def _extract_condition_value(file_name: str) -> str:
        """Infer the biological cohort without treating `B6_VPA` as condition.

        Experimental filenames encode `_cont_` or `_exp_`.  Directory names
        are used only as a fallback for external/reference recordings whose
        filenames do not carry a condition token.
        """
        path = os.path.normpath(str(file_name))
        stem = os.path.splitext(os.path.basename(path))[0].lower()
        if re.search(r'(?:^|_)cont(?:_|$)', stem):
            return 'control'
        if re.search(r'(?:^|_)exp(?:_|$)', stem):
            return 'vpa'

        path_parts = {part.lower() for part in path.split(os.sep)}
        if path_parts.intersection({'cont', 'control'}):
            return 'control'
        if path_parts.intersection({'exp', 'experiment', 'experiments', 'vpa'}):
            return 'vpa'
        if 'reference' in path_parts:
            return 'reference'
        return 'unknown'

    @staticmethod
    def _extract_pnd_value(file_name: str) -> float:
        stem = os.path.basename(str(file_name))
        match = re.search(r'_PND?(\d+)(?:_|\.|$)', stem, flags=re.IGNORECASE)
        return float(match.group(1)) if match else float('nan')

    @staticmethod
    def _extract_sex_value(file_name: str) -> str:
        stem = os.path.splitext(os.path.basename(str(file_name)))[0]
        matches = re.findall(r'(?:^|_)(f|m)(?:_|$)', stem, flags=re.IGNORECASE)
        return str(matches[-1]).lower() if matches else ''

    def load_pose_sequence(self, file_name: str) -> dict:
        """Load one complete recording for EDA without dropping any time windows.

        Coordinates are ROI-normalized and aspect-corrected using the same
        preprocessing as the model dataset. Short, fully bounded gaps are
        interpolated on the full recording timeline. Longer gaps remain NaN.
        Raw coordinates are not body-scaled, standardized, augmented, or
        centroid-subtracted here. Long disappearance and unstable reacquisition
        frames remain NaN. A fixed complete-recording body scale and its QC
        metadata are returned alongside those coordinates.
        """
        file_name = str(file_name)
        df = pd.read_csv(resolve_data_path(file_name))
        required_columns = ['frame_idx', 'track'] + [
            f'{kp}.{suffix}'
            for kp in self.keypoint_names
            for suffix in ('x', 'y', 'score')
        ]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in {file_name}: {missing_columns}")

        frame_values = pd.to_numeric(df['frame_idx'], errors='coerce').to_numpy(dtype=np.float64)
        if not np.isfinite(frame_values).all():
            raise ValueError(f'frame_idx contains non-numeric or missing values: {file_name}')
        rounded_frames = np.rint(frame_values)
        if not np.allclose(frame_values, rounded_frames):
            raise ValueError(f'frame_idx contains non-integer values: {file_name}')
        df['frame_idx'] = rounded_frames.astype(np.int64)
        if df.duplicated(['frame_idx', 'track']).any():
            duplicate_count = int(df.duplicated(['frame_idx', 'track']).sum())
            raise ValueError(f'duplicate frame_idx/track rows in {file_name}: {duplicate_count}')

        df.set_index(['frame_idx', 'track'], inplace=True)
        present_tracks = set(df.index.get_level_values('track').astype(str).unique())
        if self.animal_track_ids is None:
            all_tracks = sorted(present_tracks)
            if len(all_tracks) != self.num_animals:
                raise ValueError(
                    f"File {file_name} has {len(all_tracks)} tracks, "
                    f"but num_animals is {self.num_animals}."
                )
            self.animal_track_ids = all_tracks
        expected_tracks = set(str(track) for track in self.animal_track_ids)
        if present_tracks != expected_tracks:
            raise ValueError(
                f"Track mismatch in {file_name}: expected {sorted(expected_tracks)}, "
                f"found {sorted(present_tracks)}"
            )

        observed_frame_ids = np.sort(df.index.get_level_values('frame_idx').unique())
        if observed_frame_ids.size == 0:
            raise ValueError(f'No frame_idx values found: {file_name}')
        frame_ids = np.arange(
            int(observed_frame_ids[0]),
            int(observed_frame_ids[-1]) + 1,
            dtype=np.int64,
        )
        frame_present_mask = np.isin(frame_ids, observed_frame_ids)
        aligned_index = pd.MultiIndex.from_product(
            [frame_ids, self.animal_track_ids],
            names=['frame_idx', 'track'],
        )
        df_aligned = df.reindex(aligned_index)

        roi_transform = self._load_roi_transform(file_name)
        if self.roi_normalize and roi_transform is None and self.roi_missing == 'skip':
            raise ValueError(f'ROI normalization configured to skip file: {file_name}')
        track_coords = []
        track_scores = []
        for track_id in self.animal_track_ids:
            track = df_aligned.loc[(slice(None), track_id), :]
            track_coords.append(
                self._extract_coordinates(
                    track,
                    self.keypoint_names,
                    roi_transform=roi_transform,
                )
            )
            score_columns = []
            for keypoint in self.keypoint_names:
                score_columns.append(
                    pd.to_numeric(
                        track.get(
                            f'{keypoint}.score',
                            pd.Series(np.nan, index=track.index),
                        ),
                        errors='coerce',
                    ).to_numpy(dtype=np.float64)
                )
            track_scores.append(np.stack(score_columns, axis=1))

        raw_coords = np.stack(track_coords, axis=0)
        filled_coords, observed_coord_mask, interpolated_coord_mask = (
            self._interpolate_short_internal_gaps(
                raw_coords,
                max_gap=self.interpolation_max_gap,
            )
        )
        total_frames = int(frame_ids.size)
        coords = filled_coords.reshape(
            self.num_animals,
            total_frames,
            self.num_keypoints,
            2,
        ).transpose(1, 0, 2, 3)
        observed_mask = observed_coord_mask.reshape(
            self.num_animals,
            total_frames,
            self.num_keypoints,
            2,
        ).all(axis=-1).transpose(1, 0, 2)
        interpolated_mask = interpolated_coord_mask.reshape(
            self.num_animals,
            total_frames,
            self.num_keypoints,
            2,
        ).any(axis=-1).transpose(1, 0, 2)
        confidence = np.stack(track_scores, axis=1)

        provisional_scale = self._compute_recording_body_scale(coords)
        pose_stability = self._compute_pose_stability_masks(
            coords,
            observed_mask,
            provisional_scale['animal_body_scales'],
        )
        coords = coords.copy()
        coords[pose_stability['invalid']] = np.nan
        body_scale_info = self._compute_recording_body_scale(coords)

        return {
            'file_path': file_name,
            'file_name': os.path.basename(file_name),
            'condition': self._extract_condition_value(file_name),
            'cage_id': self._extract_subject_id(file_name),
            'sex': self._extract_sex_value(file_name),
            'week': self._extract_week_value(file_name),
            'pnd': self._extract_pnd_value(file_name),
            'frame_ids': frame_ids,
            'frame_present_mask': frame_present_mask.astype(bool),
            'coords': coords,
            'confidence': confidence,
            'observed_mask': observed_mask.astype(bool),
            'interpolated_mask': interpolated_mask.astype(bool),
            'pose_long_missing': pose_stability['long_missing'].astype(bool),
            'pose_reacquisition_invalid': pose_stability['reacquisition'].astype(bool),
            'pose_stability_invalid': pose_stability['invalid'].astype(bool),
            'roi_available': bool((not self.roi_normalize) or (roi_transform is not None)),
            **body_scale_info,
        }

    def _extract_coordinates(
        self,
        df_track,
        keypoint_names,
        roi_transform: Optional[np.ndarray] = None,
        clip: bool = True,
        apply_aspect: bool = True,
    ):
        coords = []
        for kp in keypoint_names:
            x = df_track.get(f'{kp}.x', pd.Series(np.nan, index=df_track.index)).values
            y = df_track.get(f'{kp}.y', pd.Series(np.nan, index=df_track.index)).values
            score = df_track.get(f'{kp}.score', pd.Series(0, index=df_track.index)).values
            mask = (score >= self.score_thresh) & np.isfinite(x) & np.isfinite(y)
            x = np.where(mask, x, np.nan)
            y = np.where(mask, y, np.nan)
            coords.extend([x, y])
        coords = np.stack(coords, axis=1)
        # Most CSVs are already full-frame normalized. If pixel coordinates appear,
        # convert to full-frame normalized coordinates before ROI mapping.
        if np.isfinite(coords).any() and np.nanmax(coords) > 2.0:
            coords[:, 0::2] = coords[:, 0::2] / max(float(self.frame_width), 1.0)
            coords[:, 1::2] = coords[:, 1::2] / max(float(self.frame_height), 1.0)
        if roi_transform is not None:
            coords = self._apply_roi_transform(coords, roi_transform)
        # Existing coordinate normalization after optional ROI normalization.
        if clip:
            coords = np.clip(coords, 0.0, 1.0, out=coords, where=~np.isnan(coords))
        # Optionally scale x-axis so Euclidean geometry is isotropic in height units.
        if apply_aspect and self.apply_aspect_correction:
            coords[:, 0::2] *= float(self.aspect_scale_x)
        return coords

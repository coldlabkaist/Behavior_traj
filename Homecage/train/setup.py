import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DataParallel
import os
import random
import json
from collections import Counter, defaultdict
import numpy as np
import yaml
from datasets import MousePoseDataset
from model import PoseAutoEncoder


class MousePoseSubset(Dataset):
    def __init__(self, base_dataset, indices, use_augmentation, clean_input_prob=0.0):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.use_augmentation = bool(use_augmentation)
        self.clean_input_prob = float(min(max(clean_input_prob, 0.0), 1.0))

    def set_epoch(self, epoch):
        if hasattr(self.base_dataset, "set_epoch"):
            self.base_dataset.set_epoch(epoch)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        base_idx = self.indices[idx]
        use_augmentation = self.use_augmentation
        if use_augmentation and self.clean_input_prob > 0.0:
            use_augmentation = bool(torch.rand(()) >= self.clean_input_prob)
        return self.base_dataset.get_item(base_idx, use_augmentation=use_augmentation)


class ModelSetup:
    @staticmethod
    def print_dimension_report(model, config, rank):
        if rank != 0:
            return
        m = model.module if hasattr(model, "module") else model
        n_animals = int(m.num_animals)
        n_keypoints = int(m.num_keypoints)
        in_dim = int(getattr(m, "input_dim", 2))
        rec_dim = int(getattr(m, "reconstruction_dim", 2))
        hidden_dim = int(getattr(m, "hidden_dim", config.get("model", {}).get("hidden_dim", 128)))
        latent_dim = int(
            getattr(m, "latent_dim", config.get("model", {}).get("latent_dim", 64))
        )

        dec = getattr(m, "decoder", None)
        enc = getattr(m, "encoder", None)
        relation = getattr(m, "relation_block", None)
        keypoint_names = list(m.keypoint_names)

        print("[Model Dim Report]")
        print(
            f"Input: [B,T,N,K,D]=[B,T,{n_animals},{n_keypoints},{in_dim}] | "
            f"Reconstruction channels={rec_dim}"
        )
        print(
            f"ST-GCN output: [B,T,N,{getattr(enc, 'out_channels', hidden_dim * 2)}] | "
            f"shared animal projection={getattr(m, 'animal_projection_dim', None)} | "
            f"shared animal BiGRU hidden={getattr(m, 'animal_temporal_hidden_dim', None)}"
        )
        if relation is not None:
            print(
                "Interaction: unordered pair tokens kept through a shared pair BiGRU | "
                f"pair_features={len(getattr(relation, 'FEATURE_NAMES', ()))}, "
                f"group_features={len(getattr(relation, 'GROUP_FEATURE_NAMES', ()))}"
            )
        print(
            "Pose summary + pair summary + group-state summary -> "
            f"one shared window z: [B,{latent_dim}]"
        )
        print(
            "Decoder: shared slot-conditioned GRU | "
            f"slots={getattr(dec, 'num_animals', n_animals)}, "
            f"slot_dim={getattr(dec, 'slot_embedding_dim', None)}, "
            f"frames={getattr(dec, 'clip_len', None)}"
        )
        print(f"Body-center keypoint idx: {getattr(m, 'body_center_idx', None)}")
        if keypoint_names:
            print(f"Keypoints: {keypoint_names}")

    @staticmethod
    def setup_reproducibility(config):
        training_config = config.get('training', {})
        seed = int(training_config.get('seed', 42))
        deterministic = bool(training_config.get('deterministic', True))
        deterministic_warn_only = bool(training_config.get('deterministic_warn_only', False))

        os.environ.setdefault("PYTHONHASHSEED", str(seed))
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = deterministic
        torch.use_deterministic_algorithms(deterministic, warn_only=deterministic_warn_only)

        return seed

    @staticmethod
    def setup_gpu(config):
        gpu_ids = config.get('gpu_ids', [0])
        is_multi_gpu = len(gpu_ids) > 1 and torch.cuda.device_count() > 1
        if torch.cuda.is_available():
            if is_multi_gpu:
                 print(f"Using DataParallel on GPUs: {gpu_ids}")
            else:
                print(f"Using single GPU: cuda:{gpu_ids[0]}")
            device = torch.device(f'cuda:{gpu_ids[0]}')
        else:
            print("CUDA not available, using CPU")
            device = torch.device('cpu')
            gpu_ids = []
            is_multi_gpu = False
        return device, gpu_ids, is_multi_gpu, 0

    @staticmethod
    def setup_amp(config, rank):
        use_amp = (config.get('training', {}).get('use_amp', True) and
                   torch.cuda.is_available())
        scaler = GradScaler() if use_amp else None
        if rank == 0:
            if use_amp:
                print("AMP (Automatic Mixed Precision) enabled!")
            else:
                print("AMP disabled.")
        return use_amp, scaler

    @staticmethod
    def create_model(config, device, rank):
        if rank == 0:
            print("=== model creation started ===")
        model_config = config.get('model', {})
        with open(os.path.join('cfg', 'animal.yaml'), 'r', encoding='utf-8') as f:
            animal_cfg = yaml.safe_load(f) or {}
        model = PoseAutoEncoder(
            hidden_dim=model_config.get('hidden_dim', 512),
            num_animals=animal_cfg.get('num_animals', 3),
            num_keypoints=animal_cfg.get('num_keypoints', 6),
            keypoint_names=animal_cfg.get('kpt_names', None),
            body_center_keypoint=animal_cfg.get('body_center_keypoint', None),
            skeleton_edges=animal_cfg.get('skeleton', None),
            config=config
        )
        if rank == 0:
            print("=== model creation completed ===")
        
        model = model.to(device)
        ModelSetup.print_dimension_report(model, config, rank)
        return model
    
    @staticmethod
    def setup_multi_gpu(model, gpu_ids, is_multi_gpu):
        if is_multi_gpu:
            model = DataParallel(model, device_ids=gpu_ids)
            print(f"Model wrapped with DataParallel on GPUs: {gpu_ids}")
            return model
        else:
            return model
    
    @staticmethod
    def print_model_info(model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() 
                              if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        print("\n=================== Model Architecture ===================")
        print(model)
        print("==========================================================\n")


class DataSetup:
    @staticmethod
    def _worker_init_fn(worker_id):
        worker_seed = torch.initial_seed() % (2 ** 32)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    @staticmethod
    def _extract_subject_key(file_path):
        stem = os.path.splitext(os.path.basename(str(file_path)))[0]
        parts = stem.split('_')
        if len(parts) >= 4 and parts[2]:
            return f"{parts[2]}_{parts[3].lower()}"
        return stem

    @staticmethod
    def _summarize_split(indices, dataset_full):
        week_counts = Counter()
        subject_ids = set()
        file_paths = set()
        subject_sexes = {}
        has_subject_mapping = hasattr(dataset_full, "clip_to_subject_mapping")
        for idx in indices:
            file_path = str(dataset_full.clip_to_file_mapping[idx])
            # Persist a platform-independent relative path in the split
            # manifest.  The same cage split must compare exactly after a
            # checkpoint moves between Linux training and Windows analysis.
            file_paths.add(os.path.normpath(file_path).replace("\\", "/"))
            week_value = float(dataset_full.clip_to_week_mapping[idx])
            if np.isfinite(week_value):
                week_counts[int(round(week_value))] += 1
            if has_subject_mapping and idx < len(dataset_full.clip_to_subject_mapping):
                subject_id = str(dataset_full.clip_to_subject_mapping[idx])
            else:
                subject_id = DataSetup._extract_subject_key(file_path)
            subject_ids.add(subject_id)
            if hasattr(dataset_full, "_extract_sex_value"):
                sex = str(dataset_full._extract_sex_value(file_path)).lower()
                if sex:
                    previous = subject_sexes.get(subject_id)
                    if previous is not None and previous != sex:
                        raise ValueError(
                            f"Cage {subject_id!r} has inconsistent sex labels: "
                            f"{previous!r} and {sex!r}."
                        )
                    subject_sexes[subject_id] = sex
        return {
            "n_clips": len(indices),
            "n_subjects": len(subject_ids),
            "week_counts": {
                str(week): int(count)
                for week, count in sorted(week_counts.items())
            },
            "subject_ids": sorted(subject_ids),
            "n_files": len(file_paths),
            "file_paths": sorted(file_paths),
            "sex_cage_counts": dict(sorted(Counter(subject_sexes.values()).items())),
        }

    @staticmethod
    def _validate_grouped_split(dataset_full, train_indices, val_indices):
        train_indices = [int(idx) for idx in train_indices]
        val_indices = [int(idx) for idx in val_indices]
        train_index_set = set(train_indices)
        val_index_set = set(val_indices)
        expected_indices = set(range(len(dataset_full)))

        if not train_indices or not val_indices:
            raise ValueError("Both train and validation splits must contain clips.")
        if len(train_index_set) != len(train_indices):
            raise ValueError("Duplicate clip indices were found in the train split.")
        if len(val_index_set) != len(val_indices):
            raise ValueError("Duplicate clip indices were found in the validation split.")
        index_overlap = train_index_set.intersection(val_index_set)
        if index_overlap:
            raise ValueError(
                "Train/validation clip overlap detected: "
                f"{sorted(index_overlap)[:10]}"
            )
        assigned_indices = train_index_set.union(val_index_set)
        if assigned_indices != expected_indices:
            missing = sorted(expected_indices - assigned_indices)
            unexpected = sorted(assigned_indices - expected_indices)
            raise ValueError(
                "Train/validation split does not partition the dataset exactly: "
                f"missing={missing[:10]}, unexpected={unexpected[:10]}."
            )

        train_summary = DataSetup._summarize_split(train_indices, dataset_full)
        val_summary = DataSetup._summarize_split(val_indices, dataset_full)
        cage_overlap = set(train_summary["subject_ids"]).intersection(
            val_summary["subject_ids"]
        )
        if cage_overlap:
            raise ValueError(
                "Cage leakage detected between train and validation: "
                f"{sorted(cage_overlap)}"
            )
        return train_summary, val_summary

    @staticmethod
    def _build_split_manifest(*, seed, val_ratio, train_summary, val_summary):
        return {
            "manifest_version": 1,
            "split_unit": "cage",
            "seed": int(seed),
            "requested_val_ratio": float(val_ratio),
            "cage_overlap": [],
            "normalization": {
                "body_scale_scope": "each_cage_week_recording",
                "body_scale_source": "recording_neck_bodyc_tail_temporal_median",
                "validation_uses_own_body_scale": True,
                "population_mean_std_used": False,
                "population_velocity_scale_used": False,
            },
            "train": train_summary,
            "validation": val_summary,
        }

    @staticmethod
    def _build_cage_week_sampling_weights(dataset_full, train_indices):
        """Give every cage, then every available week within cage, equal mass."""
        train_indices = [int(idx) for idx in train_indices]
        if not train_indices:
            raise ValueError("Cannot build balanced sampling weights for an empty split.")

        subject_mapping = getattr(dataset_full, "clip_to_subject_mapping", None)
        cage_week_counts = Counter()
        cage_weeks = defaultdict(set)
        clip_groups = []
        for idx in train_indices:
            file_path = str(dataset_full.clip_to_file_mapping[idx])
            if subject_mapping is not None and idx < len(subject_mapping):
                cage_id = str(subject_mapping[idx])
            else:
                cage_id = DataSetup._extract_subject_key(file_path)
            week_value = float(dataset_full.clip_to_week_mapping[idx])
            if not np.isfinite(week_value):
                raise ValueError(
                    "Cage-week balanced sampling requires a finite week for every "
                    f"training clip; missing week at index {idx} ({file_path})."
                )
            week = int(round(week_value))
            group = (cage_id, week)
            clip_groups.append(group)
            cage_week_counts[group] += 1
            cage_weeks[cage_id].add(week)

        weights = np.asarray(
            [
                1.0
                / (
                    float(len(cage_weeks[cage_id]))
                    * float(cage_week_counts[(cage_id, week)])
                )
                for cage_id, week in clip_groups
            ],
            dtype=np.float64,
        )
        if not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError("Balanced sampling produced invalid clip weights.")

        cage_weight_totals = defaultdict(float)
        cage_week_weight_totals = defaultdict(float)
        for weight, (cage_id, week) in zip(weights, clip_groups):
            cage_weight_totals[cage_id] += float(weight)
            cage_week_weight_totals[(cage_id, week)] += float(weight)
        for cage_id, total in cage_weight_totals.items():
            if not np.isclose(total, 1.0, rtol=1e-10, atol=1e-12):
                raise RuntimeError(
                    f"Balanced sampler cage mass is not 1 for {cage_id}: {total}"
                )
        for cage_id, weeks in cage_weeks.items():
            target = 1.0 / float(len(weeks))
            for week in weeks:
                total = cage_week_weight_totals[(cage_id, week)]
                if not np.isclose(total, target, rtol=1e-10, atol=1e-12):
                    raise RuntimeError(
                        "Balanced sampler cage-week mass is incorrect for "
                        f"{cage_id} {week}W: {total} vs {target}"
                    )

        retained_by_cage_week = defaultdict(dict)
        for (cage_id, week), count in sorted(cage_week_counts.items()):
            retained_by_cage_week[cage_id][str(week)] = int(count)
        n_cages = len(cage_weeks)
        summary = {
            "policy": "equal_cage_then_equal_available_week_then_window",
            "replacement": True,
            "num_samples_per_epoch": len(train_indices),
            "n_train_cages": n_cages,
            "expected_probability_per_cage": 1.0 / float(n_cages),
            "clip_weight_formula": "1 / (available_weeks_in_cage * retained_windows_in_cage_week)",
            "retained_windows_by_cage_week": dict(retained_by_cage_week),
        }
        return torch.as_tensor(weights, dtype=torch.double), summary

    @staticmethod
    def _build_grouped_subject_split(dataset_full, val_ratio, seed, search_trials):
        if len(dataset_full) < 2:
            raise ValueError("Need at least 2 clips to create a train/validation split.")

        subject_to_indices = defaultdict(list)
        subject_to_week_counts = defaultdict(Counter)
        total_week_counts = Counter()
        subject_mapping = getattr(dataset_full, "clip_to_subject_mapping", None)

        for idx, file_path in enumerate(dataset_full.clip_to_file_mapping):
            if subject_mapping is not None and idx < len(subject_mapping):
                subject_id = str(subject_mapping[idx])
            else:
                subject_id = DataSetup._extract_subject_key(file_path)
            subject_to_indices[subject_id].append(idx)

            week_value = float(dataset_full.clip_to_week_mapping[idx])
            if np.isfinite(week_value):
                week_num = int(round(week_value))
                subject_to_week_counts[subject_id][week_num] += 1
                total_week_counts[week_num] += 1

        subject_ids = sorted(subject_to_indices.keys())
        if len(subject_ids) < 2:
            raise ValueError("Grouped validation requires at least 2 distinct subjects.")

        target_val_clips = int(round(len(dataset_full) * float(val_ratio)))
        target_val_clips = min(max(1, target_val_clips), len(dataset_full) - 1)
        target_val_subjects = min(
            max(1, int(round(len(subject_ids) * float(val_ratio)))),
            len(subject_ids) - 1,
        )
        week_keys = sorted(total_week_counts.keys())
        target_week_counts = {
            week_key: float(total_week_counts[week_key]) * float(val_ratio)
            for week_key in week_keys
        }

        def _evaluate_candidate(val_subject_ids):
            val_subject_set = set(val_subject_ids)
            if not val_subject_set or len(val_subject_set) == len(subject_ids):
                return None

            val_clip_count = sum(len(subject_to_indices[sid]) for sid in val_subject_set)
            val_week_counts = Counter()
            for sid in val_subject_set:
                val_week_counts.update(subject_to_week_counts[sid])

            size_error = abs(val_clip_count - target_val_clips) / max(1.0, float(target_val_clips))
            subject_error = abs(len(val_subject_set) - target_val_subjects) / max(1.0, float(target_val_subjects))
            if week_keys:
                week_error = sum(
                    abs(float(val_week_counts.get(week_key, 0)) - target_week_counts[week_key]) /
                    max(1.0, target_week_counts[week_key])
                    for week_key in week_keys
                ) / float(len(week_keys))
            else:
                week_error = 0.0
            score = size_error + (0.25 * subject_error) + week_error
            return score, val_clip_count, tuple(sorted(val_subject_set))

        rng = random.Random(int(seed))
        best = None
        n_trials = max(32, int(search_trials))

        for _ in range(n_trials):
            shuffled_subjects = list(subject_ids)
            rng.shuffle(shuffled_subjects)

            cumulative = 0
            chosen = []
            for subject_id in shuffled_subjects:
                chosen.append(subject_id)
                cumulative += len(subject_to_indices[subject_id])
                if cumulative >= target_val_clips:
                    break

            candidates = []
            if chosen:
                candidates.append(chosen)
            if len(chosen) > 1:
                candidates.append(chosen[:-1])

            for candidate in candidates:
                evaluated = _evaluate_candidate(candidate)
                if evaluated is None:
                    continue
                if best is None or evaluated[0] < best[0]:
                    best = evaluated

        if best is None:
            raise RuntimeError("Failed to build a grouped validation split.")

        val_subject_ids = set(best[2])
        val_indices = []
        train_indices = []
        for subject_id, subject_indices in subject_to_indices.items():
            if subject_id in val_subject_ids:
                val_indices.extend(subject_indices)
            else:
                train_indices.extend(subject_indices)

        train_indices.sort()
        val_indices.sort()
        return train_indices, val_indices, sorted(val_subject_ids)

    @staticmethod
    def make_collate_fn(expected_k):
        def _collate_metadata(meta_items):
            if len(meta_items) == 0:
                return {}
            if not isinstance(meta_items[0], dict):
                return meta_items
            meta = {}
            keys = meta_items[0].keys()
            for key in keys:
                vals = [item.get(key) for item in meta_items]
                first = vals[0]
                if torch.is_tensor(first):
                    meta[key] = torch.stack(vals, dim=0)
                elif isinstance(first, (float, int, np.floating, np.integer)):
                    meta[key] = torch.tensor(vals, dtype=torch.float32)
                else:
                    meta[key] = vals
            return meta

        def _collate_fn(batch):
            if len(batch) == 0:
                raise ValueError("Received empty batch in collate_fn.")
            if any(not isinstance(item, tuple) or len(item) != 3 for item in batch):
                raise ValueError("Each dataset sample must be (noisy, clean, metadata).")

            def _stack_view(view_items, label):
                required = {"node_position", "group_speed"}
                for view in view_items:
                    if not isinstance(view, dict) or set(view) != required:
                        raise ValueError(
                            f"{label} view must contain exactly {sorted(required)}, "
                            f"got {sorted(view) if isinstance(view, dict) else type(view).__name__}"
                        )
                result = {
                    key: torch.stack([view[key] for view in view_items], dim=0)
                    for key in sorted(required)
                }
                pos = result["node_position"]
                speed = result["group_speed"]
                if pos.ndim != 5 or tuple(pos.shape[2:]) != (3, expected_k, 2):
                    raise ValueError(
                        f"{label}.node_position must be [B,T,3,{expected_k},2], "
                        f"got {tuple(pos.shape)}"
                    )
                if speed.ndim != 3 or speed.shape[-1] != 1 or speed.shape[:2] != pos.shape[:2]:
                    raise ValueError(
                        f"{label}.group_speed must be [B,T,1] aligned to positions, "
                        f"got {tuple(speed.shape)} for positions {tuple(pos.shape)}"
                    )
                return result

            noisy = _stack_view([item[0] for item in batch], "noisy")
            clean = _stack_view([item[1] for item in batch], "clean")
            batch_meta = _collate_metadata([item[2] for item in batch])
            return noisy, clean, batch_meta
        return _collate_fn

    @staticmethod
    def setup_data_loaders(config, rank):
        training_config = config.get('training', {})
        data_config = config.get('data', {})
        seed = int(training_config.get('seed', 42))
        val_ratio = float(training_config.get('val_ratio', 0.2))
        grouped_split_trials = int(training_config.get('grouped_val_search_trials', 256))
        data_dir = data_config.get('data_dir', 'data')
        recursive = bool(data_config.get('recursive', False))
        dataset_full = MousePoseDataset(
            data_dir=data_dir,
            clip_len=training_config.get('clip_len', 30),
            clip_overlap=training_config.get('clip_overlap', 0),
            keypoint_names=None,
            num_keypoints=None,
            config=config,
            use_augmentation=True,
            recursive=recursive,
        )
        train_indices, val_indices, val_subject_ids = DataSetup._build_grouped_subject_split(
            dataset_full,
            val_ratio=val_ratio,
            seed=seed,
            search_trials=grouped_split_trials,
        )
        train_summary, val_summary = DataSetup._validate_grouped_split(
            dataset_full,
            train_indices,
            val_indices,
        )
        train_sampling_weights, train_sampling_summary = (
            DataSetup._build_cage_week_sampling_weights(
                dataset_full,
                train_indices,
            )
        )
        split_manifest = DataSetup._build_split_manifest(
            seed=seed,
            val_ratio=val_ratio,
            train_summary=train_summary,
            val_summary=val_summary,
        )
        split_manifest["training_sampling"] = train_sampling_summary
        train_clean_input_prob = float(training_config.get('train_clean_input_prob', 0.25))
        train_dataset = MousePoseSubset(
            dataset_full,
            train_indices,
            use_augmentation=True,
            clean_input_prob=train_clean_input_prob,
        )
        dataset_full.split_manifest = split_manifest
        train_dataset.split_manifest = split_manifest
        val_noisy_use_augmentation = bool(training_config.get('val_noisy_use_augmentation', True))
        val_clean_use_augmentation = bool(training_config.get('val_clean_use_augmentation', False))
        val_datasets = {
            "noisy": MousePoseSubset(dataset_full, val_indices, use_augmentation=val_noisy_use_augmentation),
            "clean": MousePoseSubset(dataset_full, val_indices, use_augmentation=val_clean_use_augmentation),
        }
        num_workers = training_config.get('num_workers', 4) if os.name != 'nt' else 0
        train_sampler_generator = torch.Generator()
        train_sampler_generator.manual_seed(seed)
        train_sampler = WeightedRandomSampler(
            train_sampling_weights,
            num_samples=len(train_indices),
            replacement=True,
            generator=train_sampler_generator,
        )
        train_loader_generator = torch.Generator()
        train_loader_generator.manual_seed(seed + 1)
        with open(os.path.join('cfg', 'animal.yaml'), 'r', encoding='utf-8') as f:
            animal_cfg = yaml.safe_load(f) or {}
        expected_k = int(animal_cfg['num_keypoints'])

        train_loader = DataLoader(
            train_dataset,
            batch_size=training_config.get('batch_size', 64),
            shuffle=False,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True if num_workers > 0 else False,
            collate_fn=DataSetup.make_collate_fn(expected_k),
            worker_init_fn=DataSetup._worker_init_fn if num_workers > 0 else None,
            generator=train_loader_generator
        )
        val_loaders = {}
        for loader_idx, (loader_name, val_dataset) in enumerate(val_datasets.items()):
            val_loader_generator = torch.Generator()
            val_loader_generator.manual_seed(seed + 2 + loader_idx)
            val_loaders[loader_name] = DataLoader(
                val_dataset,
                batch_size=training_config.get('batch_size', 64),
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True if num_workers > 0 else False,
                collate_fn=DataSetup.make_collate_fn(expected_k),
                worker_init_fn=DataSetup._worker_init_fn if num_workers > 0 else None,
                generator=val_loader_generator
            )
        if rank == 0:
            print(f"Train samples: {len(train_dataset)}")
            print(f"Val samples: {len(val_indices)}")
            print(
                "Grouped split summary: "
                f"train_subjects={train_summary['n_subjects']}, "
                f"val_subjects={val_summary['n_subjects']}, "
                f"val_subject_ids={val_subject_ids}"
            )
            print(f"Train week counts: {train_summary['week_counts']}")
            print(f"Val week counts: {val_summary['week_counts']}")
            print(f"Train sex cage counts: {train_summary['sex_cage_counts']}")
            print(f"Val sex cage counts: {val_summary['sex_cage_counts']}")
            print("Train/validation cage overlap: 0 (validated)")
            print(
                "Training sampler: equal cage -> equal available week -> window; "
                f"replacement=True, samples/epoch={len(train_indices)}"
            )
            print(
                "Validation modes: "
                f"noisy={'noisy->clean' if val_noisy_use_augmentation else 'clean->clean'}, "
                f"clean={'noisy->clean' if val_clean_use_augmentation else 'clean->clean'}"
            )
            print(f"Train clean-input probability: {train_clean_input_prob:.3f}")
            print(f"DataLoader workers: {num_workers}")
            print(f"Data split seed: {seed}")
            print(f"Data dir: {data_dir} (recursive={recursive})")
        return train_loader, val_loaders, train_dataset


class OptimizerSetup:
    @staticmethod
    def setup_optimizer(model, config, rank):
        training_config = config.get('training', {})
        learning_rate = float(training_config.get('learning_rate', 1e-4))
        weight_decay = float(training_config.get('weight_decay', 1e-5))
        eta_min = float(training_config.get('eta_min', 1e-5))
        warmup_epochs = int(training_config.get('warmup_epochs', 0))
        total_epochs = int(training_config.get('epochs', 200))
        cosine_tmax = int(training_config.get('cosine_tmax', max(1, total_epochs - warmup_epochs)))

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError("No trainable model parameters found.")
        optimizer = optim.AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, cosine_tmax),
            eta_min=eta_min
        )
        if warmup_epochs > 0:
            warmup_scheduler = optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0 / max(1, warmup_epochs),
                end_factor=1.0,
                total_iters=warmup_epochs
            )
            scheduler = optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs]
            )
        else:
            scheduler = cosine_scheduler

        if rank == 0:
            print("Optimizer initialized successfully.")
            if warmup_epochs > 0:
                print(f"LR schedule: Linear warmup ({warmup_epochs} epochs) + CosineAnnealing")
            else:
                print("LR schedule: CosineAnnealing")
        return optimizer, scheduler

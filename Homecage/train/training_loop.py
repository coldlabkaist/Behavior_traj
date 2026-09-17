import torch
from tqdm import tqdm


class TrainingLoop:
    """Strict Stage 1-2 denoising-autoencoder training loop.

    A batch contains a noisy and a clean view. Each view has exactly two inputs:
    body-length-normalized, group-centered node positions and group-centroid
    speed in body lengths per second. Coordinate and group-speed reconstruction
    are complemented by two low-weight relation-consistency terms recomputed
    from the aligned reconstruction.
    """

    TERM_KEYS = (
        "recon_raw",
        "group_speed_recon_raw",
        "pair_consistency_raw",
        "group_consistency_raw",
        "total",
    )
    PROGRESS_TERMS = (
        ("total", "loss"),
        ("recon_raw", "pos"),
        ("group_speed_recon_raw", "speed"),
        ("pair_consistency_raw", "pair"),
        ("group_consistency_raw", "group"),
    )

    def __init__(self, model, loss_functions, device, use_amp, config, scaler=None):
        self.model = model
        self.loss_functions = loss_functions
        self.device = device
        self.use_amp = bool(use_amp)
        self.config = config
        self.scaler = scaler
        self.ema_alpha = 0.9
        if self.use_amp and self.scaler is None:
            raise ValueError("AMP is enabled but scaler is not provided.")

    @staticmethod
    def _validate_view(view, label: str):
        required = {"node_position", "group_speed"}
        if not isinstance(view, dict) or set(view) != required:
            actual = sorted(view) if isinstance(view, dict) else type(view).__name__
            raise ValueError(
                f"{label} must contain exactly {sorted(required)}, got {actual}"
            )

        position = view["node_position"]
        group_speed = view["group_speed"]
        if not torch.is_tensor(position) or not torch.is_tensor(group_speed):
            raise TypeError(f"{label} values must both be torch tensors.")
        if position.ndim != 5 or position.shape[-1] != 2:
            raise ValueError(
                f"{label}.node_position must be [B,T,N,K,2], "
                f"got {tuple(position.shape)}"
            )
        if (
            group_speed.ndim != 3
            or group_speed.shape[-1] != 1
            or group_speed.shape[:2] != position.shape[:2]
        ):
            raise ValueError(
                f"{label}.group_speed must be [B,T,1] aligned with node_position, "
                f"got {tuple(group_speed.shape)}"
            )
        if not torch.is_floating_point(position) or not torch.is_floating_point(group_speed):
            raise TypeError(f"{label} tensors must have floating-point dtype.")
        if not torch.isfinite(position).all():
            raise ValueError(f"{label}.node_position contains NaN or Inf.")
        if not torch.isfinite(group_speed).all():
            raise ValueError(f"{label}.group_speed contains NaN or Inf.")
        if torch.any(group_speed < 0):
            raise ValueError(f"{label}.group_speed must be non-negative.")
        return position, group_speed

    def _move_view_to_device(self, view, label: str):
        position, group_speed = self._validate_view(view, label)
        return {
            "node_position": position.to(self.device, non_blocking=True),
            "group_speed": group_speed.to(self.device, non_blocking=True),
        }

    @staticmethod
    def _unpack_batch(batch):
        if not isinstance(batch, (tuple, list)) or len(batch) != 3:
            raise ValueError(
                "Each batch must be (noisy_view, clean_view, metadata)."
            )
        noisy_view, clean_view, _ = batch
        return noisy_view, clean_view

    def _zero_term_totals(self):
        return {key: 0.0 for key in self.TERM_KEYS}

    def _empty_ema_terms(self):
        return {key: None for key in self.TERM_KEYS}

    @staticmethod
    def _terms_to_float(terms):
        return {
            key: float(value.detach().item() if torch.is_tensor(value) else value)
            for key, value in terms.items()
        }

    def _update_ema_terms(self, ema_terms, current_terms):
        for key, value in current_terms.items():
            previous = ema_terms[key]
            ema_terms[key] = (
                value
                if previous is None
                else self.ema_alpha * previous + (1.0 - self.ema_alpha) * value
            )

    def _progress_postfix(self, terms, lr=None):
        postfix = {
            label: f"{terms[key]:.3e}"
            for key, label in self.PROGRESS_TERMS
        }
        if lr is not None:
            postfix["lr"] = f"{lr:.3e}"
        return postfix

    @staticmethod
    def _summary_text(terms):
        return (
            f"Total: {terms['total']:.6f}, "
            f"Pos: {terms['recon_raw']:.6f}, "
            f"GroupSpeed: {terms['group_speed_recon_raw']:.6f}, "
            f"Pair: {terms['pair_consistency_raw']:.6f}, "
            f"Group: {terms['group_consistency_raw']:.6f}"
        )

    def _average_terms(self, local_totals, local_count):
        if local_count == 0:
            return {key: float("inf") for key in self.TERM_KEYS}
        return {
            key: float(local_totals[key] / local_count)
            for key in self.TERM_KEYS
        }

    def _compute_losses(self, noisy_view, clean_view):
        noisy_position, noisy_group_speed = self._validate_view(noisy_view, "noisy")
        clean_position, clean_group_speed = self._validate_view(clean_view, "clean")

        output = self.model(noisy_position, noisy_group_speed)
        if not isinstance(output, dict):
            raise TypeError(f"Model output must be a dict, got {type(output).__name__}.")
        if "reconstruction" not in output or "group_speed_reconstruction" not in output:
            raise RuntimeError(
                "Model output must contain reconstruction and "
                "group_speed_reconstruction."
            )

        position_reconstruction = output["reconstruction"]
        speed_reconstruction = output["group_speed_reconstruction"]
        if position_reconstruction.shape != clean_position.shape:
            raise RuntimeError(
                "Position reconstruction must exactly match node_position: "
                f"pred={tuple(position_reconstruction.shape)}, "
                f"target={tuple(clean_position.shape)}"
            )
        if speed_reconstruction.shape != clean_group_speed.shape:
            raise RuntimeError(
                "Group-speed reconstruction must exactly match group_speed: "
                f"pred={tuple(speed_reconstruction.shape)}, "
                f"target={tuple(clean_group_speed.shape)}"
            )

        reconstruction_loss, aligned_reconstruction, _ = self.loss_functions.position_set_loss(
            position_reconstruction, clean_position
        )
        group_speed_loss = self.loss_functions.group_speed_loss(
            speed_reconstruction, clean_group_speed
        )
        base_model = self.model.module if hasattr(self.model, "module") else self.model
        reconstructed_interaction = base_model.compute_interaction_features(
            aligned_reconstruction
        )
        target_interaction = base_model.compute_interaction_features(clean_position)
        pair_consistency_loss = self.loss_functions.consistency_loss(
            reconstructed_interaction["pair"], target_interaction["pair"]
        )
        group_consistency_loss = self.loss_functions.consistency_loss(
            reconstructed_interaction["group"], target_interaction["group"]
        )
        total_loss = self.loss_functions.total_loss(
            reconstruction_loss,
            group_speed_loss,
            pair_consistency_loss,
            group_consistency_loss,
        )
        terms = {
            "recon_raw": reconstruction_loss,
            "group_speed_recon_raw": group_speed_loss,
            "pair_consistency_raw": pair_consistency_loss,
            "group_consistency_raw": group_consistency_loss,
            "total": total_loss,
        }
        return total_loss, reconstruction_loss, aligned_reconstruction, terms

    def _validation_step(self, batch):
        noisy_view, clean_view = self._unpack_batch(batch)
        noisy_view = self._move_view_to_device(noisy_view, "noisy")
        clean_view = self._move_view_to_device(clean_view, "clean")
        with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
            return self._compute_losses(noisy_view, clean_view)

    def train_epoch(self, train_loader, optimizer, current_epoch=None):
        self.model.train()
        totals = self._zero_term_totals()
        num_batches = 0
        ema_terms = self._empty_ema_terms()
        epoch_number = "?" if current_epoch is None else str(current_epoch + 1)
        progress_bar = tqdm(
            train_loader,
            desc=f"Training Epoch {epoch_number}",
        )

        for batch_idx, batch in enumerate(progress_bar):
            noisy_view, clean_view = self._unpack_batch(batch)
            noisy_view = self._move_view_to_device(noisy_view, "noisy")
            clean_view = self._move_view_to_device(clean_view, "clean")
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                loss, reconstruction_loss, _, terms = self._compute_losses(
                    noisy_view, clean_view
                )
            if not torch.isfinite(loss) or not torch.isfinite(reconstruction_loss):
                raise FloatingPointError(
                    f"Non-finite loss at training batch {batch_idx}."
                )

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
            else:
                loss.backward()

            grad_clip = float(self.config.get("training", {}).get("grad_clip", 1.0))
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(
                    f"Non-finite gradient norm at training batch {batch_idx}."
                )

            if self.use_amp:
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                optimizer.step()

            current_terms = self._terms_to_float(terms)
            self._update_ema_terms(ema_terms, current_terms)
            for key in self.TERM_KEYS:
                totals[key] += current_terms[key]
            num_batches += 1
            progress_bar.set_postfix(
                self._progress_postfix(
                    ema_terms,
                    lr=float(optimizer.param_groups[0]["lr"]),
                )
            )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        average_terms = self._average_terms(totals, num_batches)
        return average_terms["total"], average_terms

    def validate_epoch(self, val_loader, epoch=None, split_name="val"):
        self.model.eval()
        totals = self._zero_term_totals()
        num_batches = 0

        with torch.no_grad():
            progress_bar = tqdm(
                val_loader,
                desc=f"Validation[{split_name}] Epoch {epoch}",
                leave=False,
            )
            for batch_idx, batch in enumerate(progress_bar):
                loss, reconstruction_loss, _, terms = self._validation_step(batch)
                if not torch.isfinite(loss) or not torch.isfinite(reconstruction_loss):
                    raise FloatingPointError(
                        f"Non-finite loss at validation batch {batch_idx}."
                    )
                current_terms = self._terms_to_float(terms)
                for key in self.TERM_KEYS:
                    totals[key] += current_terms[key]
                num_batches += 1
                progress_bar.set_postfix(self._progress_postfix(current_terms))

        average_terms = self._average_terms(totals, num_batches)
        print(
            f"Validation[{split_name}] Summary - "
            f"{self._summary_text(average_terms)}"
        )
        return average_terms["total"], average_terms

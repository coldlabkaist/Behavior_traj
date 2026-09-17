import itertools

import torch
import torch.nn.functional as F


class LossFunctions:
    """True-bottleneck DAE losses for unordered three-animal trajectories."""

    def __init__(self, config, device):
        del device
        loss_weights = config.get("loss_weights", {})
        self.lambda_recon = float(loss_weights.get("lambda_recon", 1.0))
        self.lambda_group_speed = float(
            loss_weights.get("lambda_group_speed", 1.0)
        )
        self.lambda_pair_consistency = float(
            loss_weights.get("lambda_pair_consistency", 0.02)
        )
        self.lambda_group_consistency = float(
            loss_weights.get("lambda_group_consistency", 0.02)
        )
        self.group_speed_huber_delta = float(
            loss_weights.get("group_speed_huber_delta", 1.0)
        )
        self.consistency_huber_delta = float(
            loss_weights.get("consistency_huber_delta", 1.0)
        )

    @staticmethod
    def position_set_loss(
        reconstructed_position: torch.Tensor,
        target_position: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Minimum whole-window MSE over all animal assignments.

        One permutation is selected for the complete 30-frame trajectory of
        each sample.  Per-frame reassignment is intentionally impossible.

        Returns:
            loss: Mean of the best per-sample permutation losses.
            aligned_prediction: Prediction reordered to target animal slots.
            best_permutation_index: Selected permutation index per sample.
        """
        if reconstructed_position.shape != target_position.shape:
            raise ValueError(
                "Position set loss requires identical shapes, "
                f"got pred={tuple(reconstructed_position.shape)}, "
                f"target={tuple(target_position.shape)}."
            )
        if reconstructed_position.ndim != 5 or reconstructed_position.shape[-1] != 2:
            raise ValueError(
                "Position tensors must be [B,T,N,K,2], got "
                f"{tuple(reconstructed_position.shape)}."
            )
        n_animals = int(reconstructed_position.shape[2])
        if n_animals < 1:
            raise ValueError("Position set loss requires at least one animal.")

        permutations = list(itertools.permutations(range(n_animals)))
        candidates = torch.stack(
            [reconstructed_position[:, :, list(perm), :, :] for perm in permutations],
            dim=1,
        )
        target = target_position.unsqueeze(1)
        per_sample_permutation_mse = (candidates - target).square().mean(
            dim=(2, 3, 4, 5)
        )
        best_permutation_index = per_sample_permutation_mse.argmin(dim=1)
        batch_index = torch.arange(
            reconstructed_position.shape[0],
            device=reconstructed_position.device,
        )
        aligned_prediction = candidates[batch_index, best_permutation_index]
        best_loss = per_sample_permutation_mse[
            batch_index, best_permutation_index
        ].mean()
        return best_loss, aligned_prediction, best_permutation_index

    def group_speed_loss(
        self,
        reconstructed_group_speed: torch.Tensor,
        target_group_speed: torch.Tensor,
    ) -> torch.Tensor:
        return F.huber_loss(
            reconstructed_group_speed,
            target_group_speed,
            reduction="mean",
            delta=self.group_speed_huber_delta,
        )

    @staticmethod
    def _deterministic_channel_scale(target: torch.Tensor) -> torch.Tensor:
        """Detached CUDA-deterministic per-channel scale.

        CUDA median-with-indices is incompatible with PyTorch deterministic
        mode.  The relation channels are already log/asinh transformed, so a
        mean absolute deviation around the channel mean is a stable scale here.
        The normal-consistency factor puts it on an approximate SD scale.
        """
        flat = target.detach().reshape(-1, target.shape[-1]).float()
        center = flat.mean(dim=0)
        mean_absolute_deviation = (flat - center).abs().mean(dim=0)
        # The floor prevents nearly constant channels from receiving an
        # arbitrarily large weight.  Features are already log/asinh transformed.
        scale = (1.253314 * mean_absolute_deviation).clamp_min(0.1)
        view_shape = [1] * (target.ndim - 1) + [target.shape[-1]]
        return scale.to(device=target.device, dtype=target.dtype).view(*view_shape)

    def consistency_loss(
        self,
        reconstructed_features: torch.Tensor,
        target_features: torch.Tensor,
    ) -> torch.Tensor:
        if reconstructed_features.shape != target_features.shape:
            raise ValueError(
                "Consistency feature shapes must match, got "
                f"pred={tuple(reconstructed_features.shape)}, "
                f"target={tuple(target_features.shape)}."
            )
        if not torch.isfinite(reconstructed_features).all():
            raise ValueError("Reconstructed consistency features contain NaN or Inf.")
        if not torch.isfinite(target_features).all():
            raise ValueError("Target consistency features contain NaN or Inf.")
        scale = self._deterministic_channel_scale(target_features)
        return F.huber_loss(
            reconstructed_features / scale,
            target_features / scale,
            reduction="mean",
            delta=self.consistency_huber_delta,
        )

    def total_loss(
        self,
        position_set_loss: torch.Tensor,
        group_speed_loss: torch.Tensor,
        pair_consistency_loss: torch.Tensor,
        group_consistency_loss: torch.Tensor,
    ) -> torch.Tensor:
        return (
            self.lambda_recon * position_set_loss
            + self.lambda_group_speed * group_speed_loss
            + self.lambda_pair_consistency * pair_consistency_loss
            + self.lambda_group_consistency * group_consistency_loss
        )

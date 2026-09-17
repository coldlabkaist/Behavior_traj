import torch
import torch.nn as nn
import torch.utils.checkpoint

from .decoder import GroupSpeedDecoder, SharedSlotTrajectoryDecoder
from .stgcn import AnimalRelationBlock, MultiAnimalSkeletonSTGCNEncoder


class PoseAutoEncoder(nn.Module):
    """Group-level DAE with one permutation-invariant latent per window."""

    def __init__(
        self,
        hidden_dim=128,
        num_animals=3,
        num_keypoints=5,
        config=None,
        keypoint_names=None,
        body_center_keypoint=None,
        skeleton_edges=None,
    ):
        super().__init__()
        self.num_animals = int(num_animals)
        self.num_keypoints = int(num_keypoints)
        self.input_dim = 2
        self.reconstruction_dim = 2
        self.hidden_dim = int(hidden_dim)
        self.config = config or {}
        self.keypoint_names = keypoint_names
        self.body_center_keypoint = body_center_keypoint or "Body_C"
        if not isinstance(self.keypoint_names, list) or len(self.keypoint_names) != self.num_keypoints:
            raise ValueError(
                "keypoint_names must list every model keypoint in input order."
            )
        if self.body_center_keypoint not in self.keypoint_names:
            raise ValueError(
                f"body_center_keypoint={self.body_center_keypoint!r} is not in keypoint_names."
            )
        self.body_center_idx = int(self.keypoint_names.index(self.body_center_keypoint))
        required_relation_keypoints = ("Nose", "Neck", "Tail")
        missing_relation_keypoints = [
            name for name in required_relation_keypoints if name not in self.keypoint_names
        ]
        if missing_relation_keypoints:
            raise ValueError(
                "R2/R3 relation features require keypoints "
                f"{required_relation_keypoints}; missing={missing_relation_keypoints}."
            )
        self.nose_idx = int(self.keypoint_names.index("Nose"))
        self.neck_idx = int(self.keypoint_names.index("Neck"))
        self.tailbase_idx = int(self.keypoint_names.index("Tail"))

        model_cfg = self.config.get("model", {})
        skeleton_cfg = model_cfg.get("skeleton", {})
        temporal_cfg = model_cfg.get("temporal", {})
        interaction_cfg = model_cfg.get("interaction", {})
        training_cfg = self.config.get("training", {})
        decoder_cfg = self.config.get("decoder", {})
        data_cfg = self.config.get("data", {})

        self.clip_len = int(training_cfg.get("clip_len", 30))
        self.latent_dim = int(model_cfg.get("latent_dim", 64))
        self.animal_projection_dim = int(
            model_cfg.get("animal_projection_dim", 64)
        )
        self.animal_temporal_hidden_dim = int(
            model_cfg.get("animal_temporal_hidden_dim", 64)
        )
        self.pair_temporal_hidden_dim = int(
            model_cfg.get("pair_temporal_hidden_dim", 64)
        )
        self.group_temporal_hidden_dim = int(
            model_cfg.get("group_temporal_hidden_dim", 64)
        )

        self.encoder = MultiAnimalSkeletonSTGCNEncoder(
            hidden_dim=self.hidden_dim,
            num_animals=self.num_animals,
            num_keypoints=self.num_keypoints,
            skeleton_edges=[tuple(edge) for edge in (skeleton_edges or [])],
            temporal_kernel=int(temporal_cfg.get("kernel_size", 3)),
            dropout=float(temporal_cfg.get("dropout", 0.1)),
            num_blocks=int(temporal_cfg.get("num_layers", 3)),
            keypoint_pool_type=str(
                skeleton_cfg.get("keypoint_pool_type", "attention")
            ),
        )

        # The same projection and temporal encoder are applied to every animal.
        self.animal_feature_projection = nn.Sequential(
            nn.Linear(self.encoder.out_channels, self.animal_projection_dim),
            nn.LayerNorm(self.animal_projection_dim),
            nn.GELU(),
        )
        self.relation_block = AnimalRelationBlock(
            feature_channels=self.animal_projection_dim,
            relation_hidden_dim=int(
                interaction_cfg.get(
                    "relation_hidden_dim", self.animal_projection_dim
                )
            ),
            num_animals=self.num_animals,
            body_center_idx=self.body_center_idx,
            nose_idx=self.nose_idx,
            neck_idx=self.neck_idx,
            tailbase_idx=self.tailbase_idx,
            fps=float(data_cfg.get("fps", 30.0)),
        )
        self.animal_temporal_encoder = nn.GRU(
            input_size=self.animal_projection_dim,
            hidden_size=self.animal_temporal_hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.pair_temporal_encoder = nn.GRU(
            input_size=self.animal_projection_dim,
            hidden_size=self.pair_temporal_hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.group_speed_projection = nn.Sequential(
            nn.Linear(1, self.animal_projection_dim),
            nn.LayerNorm(self.animal_projection_dim),
            nn.GELU(),
        )
        self.group_token_projection = nn.Sequential(
            nn.Linear(6 * self.animal_projection_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.animal_projection_dim),
            nn.LayerNorm(self.animal_projection_dim),
        )
        self.group_temporal_encoder = nn.GRU(
            input_size=self.animal_projection_dim,
            hidden_size=self.group_temporal_hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        animal_summary_dim = 2 * self.animal_temporal_hidden_dim
        pair_summary_dim = 2 * self.pair_temporal_hidden_dim
        self.pose_summary_dim = 2 * animal_summary_dim
        self.pair_summary_dim = 2 * pair_summary_dim
        self.group_summary_dim = 2 * self.group_temporal_hidden_dim
        self.z_input_dim = (
            self.pose_summary_dim
            + self.pair_summary_dim
            + self.group_summary_dim
        )
        self.z_proj = nn.Sequential(
            nn.Linear(self.z_input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.latent_dim),
        )

        decoder_hidden_dim = int(decoder_cfg.get("hidden_dim", self.hidden_dim))
        decoder_layers = int(decoder_cfg.get("num_layers", 1))
        decoder_dropout = float(decoder_cfg.get("dropout", 0.0))
        time_embedding_dim = int(decoder_cfg.get("time_embedding_dim", 16))
        self.decoder = SharedSlotTrajectoryDecoder(
            latent_dim=self.latent_dim,
            hidden_dim=decoder_hidden_dim,
            num_animals=self.num_animals,
            num_keypoints=self.num_keypoints,
            body_center_idx=self.body_center_idx,
            clip_len=self.clip_len,
            output_dim=self.reconstruction_dim,
            slot_embedding_dim=int(decoder_cfg.get("slot_embedding_dim", 16)),
            time_embedding_dim=time_embedding_dim,
            num_layers=decoder_layers,
            dropout=decoder_dropout,
        )
        self.group_speed_decoder = GroupSpeedDecoder(
            latent_dim=self.latent_dim,
            hidden_dim=decoder_hidden_dim,
            clip_len=self.clip_len,
            time_embedding_dim=time_embedding_dim,
            num_layers=decoder_layers,
            dropout=decoder_dropout,
        )

        for module in self.z_proj:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def _validate_inputs(
        self,
        node_position: torch.Tensor,
        group_speed: torch.Tensor,
    ) -> tuple[int, int]:
        if node_position.ndim != 5:
            raise ValueError(
                f"node_position must be [B,T,N,K,2], got shape={tuple(node_position.shape)}"
            )
        b, t, n, k, d = node_position.shape
        expected = (self.num_animals, self.num_keypoints, 2)
        if (n, k, d) != expected:
            raise ValueError(
                f"node_position must be [B,T,{expected[0]},{expected[1]},2], "
                f"got shape={tuple(node_position.shape)}"
            )
        if t != self.clip_len:
            raise ValueError(
                f"Expected exactly {self.clip_len} frames per window, got {t}."
            )
        if group_speed.ndim != 3 or tuple(group_speed.shape) != (b, t, 1):
            raise ValueError(
                f"group_speed must be [B,T,1] matching node_position, got {tuple(group_speed.shape)}"
            )
        if not torch.isfinite(node_position).all():
            raise ValueError(
                "node_position contains NaN or Inf; missing-data handling must occur in the loader."
            )
        if not torch.isfinite(group_speed).all():
            raise ValueError(
                "group_speed contains NaN or Inf; missing-data handling must occur in the loader."
            )
        if torch.any(group_speed < 0):
            raise ValueError("group_speed must be non-negative.")
        return b, t

    @staticmethod
    def _bidirectional_summary(hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3 or hidden.shape[0] != 2:
            raise RuntimeError(
                "Expected one-layer bidirectional GRU hidden state [2,B,H], "
                f"got {tuple(hidden.shape)}."
            )
        return torch.cat([hidden[0], hidden[1]], dim=-1)

    def encode_components(
        self,
        node_position: torch.Tensor,
        group_speed: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        b, t = self._validate_inputs(node_position, group_speed)
        if self.training and self.config.get("training", {}).get("use_checkpointing", False):
            animal_features = torch.utils.checkpoint.checkpoint(
                self.encoder, node_position, use_reentrant=False
            )
        else:
            animal_features = self.encoder(node_position)

        expected_shape = (
            b,
            t,
            self.num_animals,
            self.encoder.out_channels,
        )
        if tuple(animal_features.shape) != expected_shape:
            raise RuntimeError(
                "ST-GCN encoder must return animal-preserving [B,T,N,C] features: "
                f"expected={expected_shape}, got={tuple(animal_features.shape)}."
            )

        projected = self.animal_feature_projection(animal_features)
        interaction = self.relation_block(projected, node_position)
        pair_tokens = interaction["pair_tokens"]
        animal_sequences = projected.permute(0, 2, 1, 3).reshape(
            b * self.num_animals,
            t,
            self.animal_projection_dim,
        )
        _, animal_hidden = self.animal_temporal_encoder(animal_sequences)
        animal_summary = self._bidirectional_summary(animal_hidden)
        animal_summary = animal_summary.view(b, self.num_animals, -1)

        # DeepSets-style symmetric aggregation: permutation of the three input
        # animals cannot alter the resulting group representation.
        animal_mean = animal_summary.mean(dim=1)
        animal_max = animal_summary.max(dim=1).values
        pose_summary = torch.cat([animal_mean, animal_max], dim=-1)

        group_speed = group_speed.to(
            device=animal_features.device,
            dtype=animal_features.dtype,
        )
        pair_sequences = pair_tokens.permute(0, 2, 1, 3).reshape(
            b * self.relation_block.num_pairs,
            t,
            self.animal_projection_dim,
        )
        _, pair_hidden = self.pair_temporal_encoder(pair_sequences)
        pair_summary_per_pair = self._bidirectional_summary(pair_hidden).view(
            b, self.relation_block.num_pairs, -1
        )
        pair_summary = torch.cat(
            [
                pair_summary_per_pair.mean(dim=1),
                pair_summary_per_pair.max(dim=1).values,
            ],
            dim=-1,
        )

        # A group-state token sees configuration and collective translation at
        # every frame.  Animal/pair pooling is symmetric, but occurs only after
        # the corresponding pose/pair tokens have been constructed.
        animal_frame_pool = torch.cat(
            [projected.mean(dim=2), projected.max(dim=2).values], dim=-1
        )
        pair_frame_pool = torch.cat(
            [pair_tokens.mean(dim=2), pair_tokens.max(dim=2).values], dim=-1
        )
        speed_token = self.group_speed_projection(group_speed)
        group_tokens = self.group_token_projection(
            torch.cat(
                [
                    animal_frame_pool,
                    pair_frame_pool,
                    interaction["group_feature_tokens"],
                    speed_token,
                ],
                dim=-1,
            )
        )
        _, group_hidden = self.group_temporal_encoder(group_tokens)
        group_summary = self._bidirectional_summary(group_hidden)

        z = self.z_proj(
            torch.cat([pose_summary, pair_summary, group_summary], dim=-1)
        )
        if tuple(z.shape) != (b, self.latent_dim):
            raise RuntimeError(
                f"Window encoder must return [B,{self.latent_dim}], got {tuple(z.shape)}."
            )
        if not torch.isfinite(z).all():
            raise RuntimeError("Window encoder produced NaN or Inf latent values.")
        return {
            "z": z,
            "pose_summary": pose_summary,
            "pair_summary": pair_summary,
            "group_summary": group_summary,
            # Inference-facing equivariant components. These tensors are
            # already part of the frozen forward graph; exposing them does not
            # add parameters or alter the group-level permutation-invariant z.
            "animal_tokens": projected,
            "pair_tokens": pair_tokens,
            "pair_features": interaction["pair_features"],
            "group_features": interaction["group_features"],
        }

    def encode_window(
        self,
        node_position: torch.Tensor,
        group_speed: torch.Tensor,
    ) -> torch.Tensor:
        return self.encode_components(node_position, group_speed)["z"]

    def compute_interaction_features(
        self, node_position: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Differentiable pair/group targets used by training and analysis."""
        return self.relation_block.compute_features(node_position)

    def decode_window(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        reconstruction = self.decoder(z)
        group_speed_reconstruction = self.group_speed_decoder(z)
        return {
            "reconstruction": reconstruction,
            "group_speed_reconstruction": group_speed_reconstruction,
        }

    def forward(
        self,
        node_position: torch.Tensor,
        group_speed: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        components = self.encode_components(node_position, group_speed)
        output = self.decode_window(components["z"])
        output.update(components)
        return output

import math
from typing import Optional

import torch
import torch.nn as nn


def _normalize_adj(a: torch.Tensor) -> torch.Tensor:
    deg = a.sum(dim=-1).clamp_min(1e-6)
    inv_sqrt = torch.pow(deg, -0.5)
    return inv_sqrt.unsqueeze(-1) * a * inv_sqrt.unsqueeze(-2)


def build_skeleton_adjacency(
    num_animals: int,
    num_keypoints: int,
    skeleton_edges: Optional[list[tuple[int, int]]] = None,
) -> torch.Tensor:
    """Return self and within-animal skeleton partitions only.

    Inter-animal communication deliberately does not occur at a particular
    keypoint.  It is handled after keypoint pooling by ``AnimalRelationBlock``.
    """
    v = int(num_animals) * int(num_keypoints)
    a_self = torch.eye(v, dtype=torch.float32)
    a_skeleton = torch.zeros(v, v, dtype=torch.float32)

    if not skeleton_edges:
        skeleton_edges = [(i, i + 1) for i in range(num_keypoints - 1)]

    def idx(animal_id: int, keypoint_id: int) -> int:
        return animal_id * num_keypoints + keypoint_id

    for animal_id in range(num_animals):
        for i, j in skeleton_edges:
            i = int(i)
            j = int(j)
            if not (0 <= i < num_keypoints and 0 <= j < num_keypoints):
                raise ValueError(
                    f"Skeleton edge {(i, j)} is outside {num_keypoints} keypoints."
                )
            ii = idx(animal_id, i)
            jj = idx(animal_id, j)
            a_skeleton[ii, jj] = 1.0
            a_skeleton[jj, ii] = 1.0

    return torch.stack(
        [_normalize_adj(a_self), _normalize_adj(a_skeleton)],
        dim=0,
    )


class SpatialGraphConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_parts: int = 2):
        super().__init__()
        self.num_parts = int(num_parts)
        self.proj = nn.Conv2d(
            in_channels,
            out_channels * self.num_parts,
            kernel_size=(1, 1),
            bias=False,
        )

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        # x: [B,C,T,V], adjacency: [P,V,V]
        b, _, t, v = x.shape
        if tuple(adjacency.shape) != (self.num_parts, v, v):
            raise ValueError(
                "Adjacency shape mismatch: "
                f"expected {(self.num_parts, v, v)}, got {tuple(adjacency.shape)}."
            )
        projected = self.proj(x).view(b, self.num_parts, -1, t, v)
        return sum(
            torch.einsum("bctv,vw->bctw", projected[:, part], adjacency[part])
            for part in range(self.num_parts)
        )


class STGCNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temporal_kernel: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        pad_t = temporal_kernel // 2
        self.gcn = SpatialGraphConv(in_channels, out_channels, num_parts=2)
        self.norm1 = nn.GroupNorm(1, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.tcn = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(temporal_kernel, 1),
                padding=(pad_t, 0),
                bias=False,
            ),
            nn.GroupNorm(1, out_channels),
            nn.Dropout(dropout),
        )
        self.res = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), bias=False),
                nn.GroupNorm(1, out_channels),
            )
        )

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        y = self.gcn(x, adjacency)
        y = self.norm1(y)
        y = self.relu(y)
        y = self.tcn(y)
        y = y + self.res(x)
        return self.relu(y)


class KeypointAttentionPool(nn.Module):
    def __init__(self, channels: int, num_keypoints: int):
        super().__init__()
        self.score = nn.Linear(int(channels), 1)
        self.keypoint_bias = nn.Parameter(torch.zeros(int(num_keypoints)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,T,N,K] -> [B,T,N,C]
        x_btnkc = x.permute(0, 2, 3, 4, 1).contiguous()
        logits = self.score(x_btnkc).squeeze(-1)
        logits = logits + self.keypoint_bias.view(1, 1, 1, -1)
        weights = torch.softmax(logits, dim=-1).unsqueeze(-1)
        return (x_btnkc * weights).sum(dim=3)


class MultiAnimalSkeletonSTGCNEncoder(nn.Module):
    """Shared skeleton-temporal encoder without cross-animal edges."""

    def __init__(
        self,
        hidden_dim: int,
        num_animals: int,
        num_keypoints: int,
        skeleton_edges: Optional[list[tuple[int, int]]] = None,
        temporal_kernel: int = 3,
        dropout: float = 0.1,
        num_blocks: int = 3,
        keypoint_pool_type: str = "attention",
    ):
        super().__init__()
        self.num_animals = int(num_animals)
        self.num_keypoints = int(num_keypoints)
        self.out_channels = int(hidden_dim) * 2
        self.keypoint_pool_type = str(keypoint_pool_type).lower()

        self.register_buffer(
            "adj",
            build_skeleton_adjacency(
                num_animals=self.num_animals,
                num_keypoints=self.num_keypoints,
                skeleton_edges=skeleton_edges,
            ),
        )

        channels = [2, hidden_dim, self.out_channels]
        while len(channels) < num_blocks + 1:
            channels.append(self.out_channels)
        channels = channels[: num_blocks + 1]
        self.blocks = nn.ModuleList(
            [
                STGCNBlock(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    temporal_kernel=temporal_kernel,
                    dropout=dropout,
                )
                for i in range(num_blocks)
            ]
        )
        if self.keypoint_pool_type in {"attention", "attn"}:
            self.keypoint_pool = KeypointAttentionPool(
                self.out_channels, self.num_keypoints
            )
        elif self.keypoint_pool_type == "mean":
            self.keypoint_pool = None
        else:
            raise ValueError(
                f"Unsupported keypoint_pool_type: {self.keypoint_pool_type}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(
                f"Expected node positions [B,T,N,K,2], got {tuple(x.shape)}."
            )
        b, t, n, k, d = x.shape
        if (n, k, d) != (self.num_animals, self.num_keypoints, 2):
            raise ValueError(
                "Expected node positions "
                f"[B,T,{self.num_animals},{self.num_keypoints},2], got {tuple(x.shape)}."
            )

        x = x.view(b, t, n * k, d).permute(0, 3, 1, 2).contiguous()
        for block in self.blocks:
            x = block(x, self.adj)

        channels = x.shape[1]
        x = x.view(b, channels, t, n, k)
        if self.keypoint_pool is None:
            return x.mean(dim=-1).permute(0, 2, 3, 1).contiguous()
        return self.keypoint_pool(x).contiguous()


class AnimalRelationBlock(nn.Module):
    """Build unordered pair tokens and minimal group-state features.

    Pair identity and the 30-frame time axis are preserved here.  Pooling over
    the three animals/pairs is deliberately deferred to the window encoder.
    Endpoint order cannot change a pair token: the two directed descriptions
    are encoded with shared weights and combined symmetrically.
    """

    FEATURE_NAMES = (
        "body_distance_log1p",
        "nose_nose_distance_log1p",
        "nose_i_tailbase_j_distance_log1p",
        "nose_j_tailbase_i_distance_log1p",
        "facing_i_to_j",
        "facing_j_to_i",
        "approach_rate_asinh",
        "relative_speed_log1p",
    )
    GROUP_FEATURE_NAMES = (
        "group_cohesion_log1p",
        "dyadic_grouping",
        "configuration_speed_log1p",
    )

    def __init__(
        self,
        feature_channels: int,
        relation_hidden_dim: int,
        num_animals: int,
        body_center_idx: int,
        nose_idx: int,
        neck_idx: int,
        tailbase_idx: int,
        fps: float = 30.0,
    ):
        super().__init__()
        self.feature_channels = int(feature_channels)
        self.relation_hidden_dim = int(relation_hidden_dim)
        self.num_animals = int(num_animals)
        self.body_center_idx = int(body_center_idx)
        self.nose_idx = int(nose_idx)
        self.neck_idx = int(neck_idx)
        self.tailbase_idx = int(tailbase_idx)
        self.fps = float(fps)
        if self.num_animals != 3:
            raise ValueError(
                "The group-state definition requires exactly three animals."
            )
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError(f"fps must be finite and positive, got {self.fps}.")

        pair_indices = torch.tensor(
            [(0, 1), (0, 2), (1, 2)], dtype=torch.long
        )
        self.register_buffer("pair_indices", pair_indices, persistent=False)

        # Each endpoint sees the same symmetric quantities plus its own
        # directed Nose->TailBase distance and facing score.
        directed_relation_dim = 6
        self.directed_relation_encoder = nn.Sequential(
            nn.Linear(directed_relation_dim, self.relation_hidden_dim),
            nn.LayerNorm(self.relation_hidden_dim),
            nn.GELU(),
            nn.Linear(self.relation_hidden_dim, self.feature_channels),
        )
        self.pair_token_projection = nn.Sequential(
            nn.Linear(4 * self.feature_channels, self.relation_hidden_dim),
            nn.LayerNorm(self.relation_hidden_dim),
            nn.GELU(),
            nn.Linear(self.relation_hidden_dim, self.feature_channels),
            nn.LayerNorm(self.feature_channels),
        )
        self.group_feature_encoder = nn.Sequential(
            nn.Linear(len(self.GROUP_FEATURE_NAMES), self.relation_hidden_dim),
            nn.LayerNorm(self.relation_hidden_dim),
            nn.GELU(),
            nn.Linear(self.relation_hidden_dim, self.feature_channels),
        )

    @property
    def num_pairs(self) -> int:
        return int(self.pair_indices.shape[0])

    @staticmethod
    def _safe_unit(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        return x / torch.linalg.vector_norm(
            x, dim=-1, keepdim=True
        ).clamp_min(eps)

    def _temporal_derivative(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[1] < 2:
            raise ValueError("Interaction dynamics require at least two frames.")
        derivative = torch.empty_like(value)
        derivative[:, 0] = (value[:, 1] - value[:, 0]) * self.fps
        derivative[:, -1] = (value[:, -1] - value[:, -2]) * self.fps
        if value.shape[1] > 2:
            derivative[:, 1:-1] = (
                value[:, 2:] - value[:, :-2]
            ) * (self.fps / 2.0)
        return derivative

    def _validate_positions(self, node_position: torch.Tensor) -> None:
        if node_position.ndim != 5 or node_position.shape[-1] != 2:
            raise ValueError(
                "node_position must be [B,T,N,K,2], got "
                f"{tuple(node_position.shape)}."
            )
        _, _, n, k, _ = node_position.shape
        if n != self.num_animals:
            raise ValueError(f"Expected {self.num_animals} animals, got {n}.")
        for name, idx in {
            "body_center": self.body_center_idx,
            "nose": self.nose_idx,
            "neck": self.neck_idx,
            "tailbase": self.tailbase_idx,
        }.items():
            if not 0 <= idx < k:
                raise ValueError(f"{name} index {idx} is outside {k} keypoints.")
        if not torch.isfinite(node_position).all():
            raise ValueError("node_position contains NaN or Inf.")

    def compute_pair_features(self, node_position: torch.Tensor) -> torch.Tensor:
        """Return transformed relation sequences [B,T,3 unordered pairs,8]."""
        self._validate_positions(node_position)
        body = node_position[:, :, :, self.body_center_idx, :]
        nose = node_position[:, :, :, self.nose_idx, :]
        neck = node_position[:, :, :, self.neck_idx, :]
        tailbase = node_position[:, :, :, self.tailbase_idx, :]
        velocity = self._temporal_derivative(body)
        heading = self._safe_unit(nose - neck)

        i = self.pair_indices[:, 0]
        j = self.pair_indices[:, 1]
        body_i, body_j = body[:, :, i], body[:, :, j]
        nose_i, nose_j = nose[:, :, i], nose[:, :, j]
        tail_i, tail_j = tailbase[:, :, i], tailbase[:, :, j]
        heading_i, heading_j = heading[:, :, i], heading[:, :, j]
        velocity_i, velocity_j = velocity[:, :, i], velocity[:, :, j]

        pair_vector = body_j - body_i
        body_distance = torch.linalg.vector_norm(pair_vector, dim=-1)
        pair_direction = self._safe_unit(pair_vector)
        nose_nose_distance = torch.linalg.vector_norm(nose_j - nose_i, dim=-1)
        nose_i_tail_j = torch.linalg.vector_norm(tail_j - nose_i, dim=-1)
        nose_j_tail_i = torch.linalg.vector_norm(tail_i - nose_j, dim=-1)
        facing_i_to_j = (heading_i * pair_direction).sum(dim=-1)
        facing_j_to_i = (heading_j * (-pair_direction)).sum(dim=-1)
        relative_velocity = velocity_j - velocity_i
        approach_rate = -(
            relative_velocity * pair_direction
        ).sum(dim=-1)
        relative_speed = torch.linalg.vector_norm(relative_velocity, dim=-1)

        pair_features = torch.stack(
            [
                torch.log1p(body_distance),
                torch.log1p(nose_nose_distance),
                torch.log1p(nose_i_tail_j),
                torch.log1p(nose_j_tail_i),
                facing_i_to_j,
                facing_j_to_i,
                torch.asinh(approach_rate),
                torch.log1p(relative_speed),
            ],
            dim=-1,
        )
        if not torch.isfinite(pair_features).all():
            raise RuntimeError("Pair feature computation produced NaN or Inf.")
        return pair_features

    def _group_features_from_pair(
        self, pair_features: torch.Tensor
    ) -> torch.Tensor:
        body_distances = torch.expm1(pair_features[..., 0]).clamp_min(0.0)
        sorted_distances = body_distances.sort(dim=2).values
        d_min = sorted_distances[:, :, 0]
        d_mid = sorted_distances[:, :, 1]
        d_max = sorted_distances[:, :, 2]

        cohesion = torch.log1p(body_distances.mean(dim=2))
        dyadic_grouping = 1.0 - d_min / (
            0.5 * (d_mid + d_max) + 1e-6
        )
        dyadic_grouping = dyadic_grouping.clamp(0.0, 1.0)
        distance_rate = self._temporal_derivative(body_distances)
        configuration_speed = torch.sqrt(
            distance_rate.square().mean(dim=2) + 1e-8
        )
        group_features = torch.stack(
            [
                cohesion,
                dyadic_grouping,
                torch.log1p(configuration_speed),
            ],
            dim=-1,
        )
        if not torch.isfinite(group_features).all():
            raise RuntimeError("Group feature computation produced NaN or Inf.")
        return group_features

    def compute_group_features(self, node_position: torch.Tensor) -> torch.Tensor:
        """Return minimal group-state sequences [B,T,3]."""
        pair_features = self.compute_pair_features(node_position)
        return self._group_features_from_pair(pair_features)

    def compute_features(
        self, node_position: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        pair = self.compute_pair_features(node_position)
        group = self._group_features_from_pair(pair)
        return {"pair": pair, "group": group}

    def forward(
        self,
        animal_features: torch.Tensor,
        node_position: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if animal_features.ndim != 4:
            raise ValueError(
                "animal_features must be [B,T,N,C], got "
                f"{tuple(animal_features.shape)}."
            )
        b, t, n, channels = animal_features.shape
        if n != self.num_animals or channels != self.feature_channels:
            raise ValueError(
                "Animal feature shape mismatch: expected "
                f"[B,T,{self.num_animals},{self.feature_channels}], got "
                f"{tuple(animal_features.shape)}."
            )
        if tuple(node_position.shape[:3]) != (b, t, n):
            raise ValueError(
                "node_position and animal_features are not aligned: "
                f"{tuple(node_position.shape)} vs {tuple(animal_features.shape)}."
            )

        pair_features = self.compute_pair_features(node_position)
        group_features = self._group_features_from_pair(pair_features)
        i = self.pair_indices[:, 0]
        j = self.pair_indices[:, 1]
        feature_i = animal_features[:, :, i]
        feature_j = animal_features[:, :, j]

        # The shared directed encoder sees six minimal quantities.  Swapping
        # pair endpoints only swaps the two encoded tensors; mean/max removes
        # that order while retaining both directions.
        directed_i = torch.stack(
            [
                pair_features[..., 0],
                pair_features[..., 1],
                pair_features[..., 2],
                pair_features[..., 4],
                pair_features[..., 6],
                pair_features[..., 7],
            ],
            dim=-1,
        )
        directed_j = torch.stack(
            [
                pair_features[..., 0],
                pair_features[..., 1],
                pair_features[..., 3],
                pair_features[..., 5],
                pair_features[..., 6],
                pair_features[..., 7],
            ],
            dim=-1,
        )
        relation_i = self.directed_relation_encoder(directed_i)
        relation_j = self.directed_relation_encoder(directed_j)
        relation_mean = 0.5 * (relation_i + relation_j)
        relation_max = torch.maximum(relation_i, relation_j)

        pose_mean = 0.5 * (feature_i + feature_j)
        pose_difference = torch.abs(feature_i - feature_j)
        pair_tokens = self.pair_token_projection(
            torch.cat(
                [pose_mean, pose_difference, relation_mean, relation_max],
                dim=-1,
            )
        )
        group_feature_tokens = self.group_feature_encoder(group_features)
        if not torch.isfinite(pair_tokens).all():
            raise RuntimeError("Pair token encoder produced NaN or Inf.")
        return {
            "pair_tokens": pair_tokens,
            "pair_features": pair_features,
            "group_features": group_features,
            "group_feature_tokens": group_feature_tokens,
        }

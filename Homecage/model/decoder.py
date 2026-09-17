import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedSlotTrajectoryDecoder(nn.Module):
    """Decode an unordered set of whole-window animal trajectories from one z.

    The learned slots only let one shared decoder produce distinct outputs.
    They are not animal identities; assignment to target tracks is handled by
    the whole-window permutation-invariant reconstruction loss.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        num_animals: int,
        num_keypoints: int,
        body_center_idx: int,
        clip_len: int,
        output_dim: int = 2,
        slot_embedding_dim: int = 16,
        time_embedding_dim: int = 16,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_animals = int(num_animals)
        self.num_keypoints = int(num_keypoints)
        self.body_center_idx = int(body_center_idx)
        self.clip_len = int(clip_len)
        self.output_dim = int(output_dim)
        self.slot_embedding_dim = int(slot_embedding_dim)
        self.time_embedding_dim = int(time_embedding_dim)
        self.num_layers = int(max(1, num_layers))
        self.dropout = float(max(0.0, dropout))

        self.slot_embedding = nn.Embedding(self.num_animals, self.slot_embedding_dim)
        self.time_embedding = nn.Parameter(
            torch.empty(self.clip_len, self.time_embedding_dim)
        )
        self.initial_hidden = nn.Sequential(
            nn.Linear(
                self.latent_dim + self.slot_embedding_dim,
                self.num_layers * self.hidden_dim,
            ),
            nn.Tanh(),
        )
        self.temporal_decoder = nn.GRU(
            input_size=self.time_embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
        )
        self.coordinate_head = nn.Linear(
            self.hidden_dim,
            self.num_keypoints * self.output_dim,
        )

        nn.init.normal_(self.slot_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.time_embedding, mean=0.0, std=0.02)
        nn.init.xavier_normal_(self.coordinate_head.weight, gain=0.1)
        nn.init.zeros_(self.coordinate_head.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2 or z.shape[-1] != self.latent_dim:
            raise ValueError(
                f"Expected window latent [B,{self.latent_dim}], got {tuple(z.shape)}"
            )
        b = int(z.shape[0])
        n = self.num_animals

        slots = self.slot_embedding.weight.unsqueeze(0).expand(b, -1, -1)
        z_slots = z.unsqueeze(1).expand(-1, n, -1)
        context = torch.cat([z_slots, slots], dim=-1)
        h0 = self.initial_hidden(context)
        h0 = h0.view(b, n, self.num_layers, self.hidden_dim)
        h0 = h0.permute(2, 0, 1, 3).reshape(
            self.num_layers, b * n, self.hidden_dim
        ).contiguous()

        time_input = self.time_embedding.unsqueeze(0).expand(b * n, -1, -1)
        decoded, _ = self.temporal_decoder(time_input, h0)
        coords = self.coordinate_head(decoded)
        coords = coords.view(
            b,
            n,
            self.clip_len,
            self.num_keypoints,
            self.output_dim,
        )
        coords = coords.permute(0, 2, 1, 3, 4).contiguous()

        # The loader defines positions relative to the three-animal Body_C
        # centroid. Enforce the same known coordinate constraint at output.
        group_center = coords[:, :, :, self.body_center_idx, :].mean(
            dim=2, keepdim=True
        )
        return coords - group_center.unsqueeze(3)


class GroupSpeedDecoder(nn.Module):
    """Decode the full non-negative group-speed sequence from one window z."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        clip_len: int,
        time_embedding_dim: int = 16,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.clip_len = int(clip_len)
        self.time_embedding_dim = int(time_embedding_dim)
        self.num_layers = int(max(1, num_layers))
        self.dropout = float(max(0.0, dropout))

        self.time_embedding = nn.Parameter(
            torch.empty(self.clip_len, self.time_embedding_dim)
        )
        self.initial_hidden = nn.Sequential(
            nn.Linear(self.latent_dim, self.num_layers * self.hidden_dim),
            nn.Tanh(),
        )
        self.temporal_decoder = nn.GRU(
            input_size=self.time_embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
        )
        self.output = nn.Linear(self.hidden_dim, 1)

        nn.init.normal_(self.time_embedding, mean=0.0, std=0.02)
        nn.init.xavier_normal_(self.output.weight, gain=0.1)
        nn.init.zeros_(self.output.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2 or z.shape[-1] != self.latent_dim:
            raise ValueError(
                f"Expected window latent [B,{self.latent_dim}], got {tuple(z.shape)}"
            )
        b = int(z.shape[0])
        h0 = self.initial_hidden(z)
        h0 = h0.view(b, self.num_layers, self.hidden_dim)
        h0 = h0.permute(1, 0, 2).contiguous()
        time_input = self.time_embedding.unsqueeze(0).expand(b, -1, -1)
        decoded, _ = self.temporal_decoder(time_input, h0)
        return F.softplus(self.output(decoded))

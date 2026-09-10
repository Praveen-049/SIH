"""Spatio-Temporal Graph Neural Network for meteorological anomaly tracking.

Implements genuine spherical graph message-passing convolution and temporal recurrence.
Processes spherical nodes across forecast lead times to predict extreme anomaly probabilities
and trajectory displacement vectors.
"""

from __future__ import annotations

import os
import sys

# Ensure Windows Python 3.14 loads PyTorch C++ DLLs cleanly
if sys.platform == "win32":
    torch_lib_dir = os.path.join(
        os.path.dirname(sys.executable), "..", "Lib", "site-packages", "torch", "lib"
    )
    if os.path.exists(torch_lib_dir):
        try:
            os.add_dll_directory(torch_lib_dir)
        except Exception:
            pass

import torch
import torch.nn as nn
import torch.nn.functional as F


class SphericalGraphConv(nn.Module):
    """Spatial message-passing convolution on spherical mesh edges.

    Aggregates neighbor features weighted by geodesic distances and edge learnable weights.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.node_transform = nn.Linear(in_features, out_features, bias=False)
        self.neighbor_transform = nn.Linear(in_features, out_features, bias=False)
        self.edge_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(
        self,
        node_features: torch.Tensor,       # (num_nodes, in_features)
        edge_index: torch.Tensor,          # (2, num_edges)
        edge_distances_norm: torch.Tensor, # (num_edges, 1) normalized geodesic distances
    ) -> torch.Tensor:
        num_nodes = node_features.shape[0]
        src_nodes = edge_index[0]
        dst_nodes = edge_index[1]

        # 1. Transform self features
        h_self = self.node_transform(node_features)

        # 2. Transform source neighbor features
        h_neighbors = self.neighbor_transform(node_features[src_nodes])

        # 3. Modulate by edge distance kernel
        edge_weights = self.edge_mlp(edge_distances_norm)
        weighted_messages = h_neighbors * edge_weights

        # 4. Scatter-add aggregation onto destination nodes
        aggregated = torch.zeros(
            (num_nodes, self.out_features),
            device=node_features.device,
            dtype=node_features.dtype,
        )
        aggregated.index_add_(0, dst_nodes, weighted_messages)

        # Degree normalization
        degrees = torch.zeros(
            (num_nodes, 1), device=node_features.device, dtype=node_features.dtype
        )
        degrees.index_add_(0, dst_nodes, torch.ones_like(src_nodes, dtype=node_features.dtype).unsqueeze(1))
        degrees = torch.clamp(degrees, min=1.0)
        aggregated = aggregated / degrees

        return h_self + aggregated + self.bias


class SpatioTemporalAnomalyGNN(nn.Module):
    """Genuine Spatio-Temporal Graph Neural Network for extreme weather anomaly prediction.

    Architecture:
    1. Input Encoder: Encodes multi-variable atmospheric observations + spherical coordinates.
    2. Spatial Spherical Convolution: Multi-layer message passing over geodesic spherical mesh.
    3. Temporal Sequence Processing: Recurrent GRU across forecast lead times (T+24, 48, 72, 96).
    4. Prediction Heads:
       - Node-level extreme anomaly probability (Sigmoid)
       - Node-level intensity delta (Linear)
       - Global event centroid trajectory offset (Linear)
    """

    def __init__(
        self,
        in_channels: int = 10,  # 6 weather variables + anomaly + EFI + spherical coords (x,y,z)
        hidden_dim: int = 64,
        temporal_layers: int = 1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim

        # Feature projection
        self.encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Spherical graph convolution layers
        self.conv1 = SphericalGraphConv(hidden_dim, hidden_dim)
        self.conv2 = SphericalGraphConv(hidden_dim, hidden_dim)

        # Temporal sequence processing across lead times
        self.temporal_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=temporal_layers,
            batch_first=True,
        )

        # Output Heads
        self.anomaly_prob_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self.intensity_delta_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # Global motion prediction head (maps mean pooled node state to dlat, dlon velocity)
        self.trajectory_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2),  # [dlat_deg/24h, dlon_deg/24h]
        )

    def forward(
        self,
        x_sequence: torch.Tensor,           # (B, T, N, in_channels)
        edge_index: torch.Tensor,          # (2, num_edges)
        edge_distances_norm: torch.Tensor, # (num_edges, 1)
    ) -> dict[str, torch.Tensor]:
        """Forward pass through spatio-temporal GNN."""
        batch_size, timesteps, num_nodes, _ = x_sequence.shape

        # 1. Spatial GNN over each timestep
        spatial_outputs: list[torch.Tensor] = []
        for t in range(timesteps):
            xt = x_sequence[:, t]  # (B, N, in_channels)
            # Flatten batch for graph conv
            xt_flat = xt.reshape(batch_size * num_nodes, self.in_channels)
            h = self.encoder(xt_flat)

            # Replicate edges across batch
            if batch_size > 1:
                # Offset edges for batch
                edge_offsets = torch.arange(batch_size, device=edge_index.device).repeat_interleave(
                    edge_index.shape[1]
                ) * num_nodes
                batch_edges = edge_index.repeat(1, batch_size) + edge_offsets.unsqueeze(0)
                batch_dist = edge_distances_norm.repeat(batch_size, 1)
            else:
                batch_edges = edge_index
                batch_dist = edge_distances_norm

            h = F.gelu(self.conv1(h, batch_edges, batch_dist))
            h = F.gelu(self.conv2(h, batch_edges, batch_dist))
            h = h.reshape(batch_size, num_nodes, self.hidden_dim)
            spatial_outputs.append(h)

        # Stack over time: (B, T, N, hidden_dim)
        spatial_seq = torch.stack(spatial_outputs, dim=1)

        # 2. Temporal sequence processing: reshape to (B * N, T, hidden_dim)
        spatial_seq_perm = spatial_seq.permute(0, 2, 1, 3).reshape(
            batch_size * num_nodes, timesteps, self.hidden_dim
        )
        temporal_out, _ = self.temporal_gru(spatial_seq_perm)
        # Reshape back to (B, T, N, hidden_dim)
        temporal_features = temporal_out.reshape(
            batch_size, num_nodes, timesteps, self.hidden_dim
        ).permute(0, 2, 1, 3)

        # 3. Output predictions
        last_timestep_feat = temporal_features[:, -1]  # (B, N, hidden_dim)

        anomaly_prob = self.anomaly_prob_head(last_timestep_feat).squeeze(-1)  # (B, N)
        intensity_delta = self.intensity_delta_head(last_timestep_feat).squeeze(-1)  # (B, N)

        # Global pooling for trajectory motion offset
        global_pooled = torch.mean(last_timestep_feat, dim=1)  # (B, hidden_dim)
        trajectory_offset = self.trajectory_head(global_pooled)  # (B, 2)

        return {
            "anomaly_probability": anomaly_prob,
            "intensity_delta": intensity_delta,
            "trajectory_offset_deg": trajectory_offset,
            "temporal_node_embeddings": temporal_features,
        }

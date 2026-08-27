"""fNIRS graph-conditioned gated K/V adapters for a frozen CBraMod backbone.

The official CBraMod source is not modified. Each selected CrissCross block
keeps its original spatial/temporal self-attention and adds a parallel,
zero-gated cross-attention whose K/V memory comes from an fNIRS graph prompt.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def normalized_adjacency(node_count: int, edge_index: torch.Tensor) -> torch.Tensor:
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"Expected edge_index [2,E], got {tuple(edge_index.shape)}")
    adjacency = torch.zeros(node_count, node_count, dtype=torch.float32)
    adjacency[edge_index[0], edge_index[1]] = 1.0
    adjacency[edge_index[1], edge_index[0]] = 1.0
    adjacency.fill_diagonal_(1.0)
    degree = adjacency.sum(dim=1).clamp_min(1.0)
    inverse_sqrt = degree.rsqrt()
    return inverse_sqrt[:, None] * adjacency * inverse_sqrt[None, :]


class DenseGraphConv(nn.Module):
    def __init__(self, dimension: int, dropout: float) -> None:
        super().__init__()
        self.self_projection = nn.Linear(dimension, dimension)
        self.neighbor_projection = nn.Linear(dimension, dimension, bias=False)
        self.norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        neighbors = torch.einsum("ij,bjd->bid", adjacency, x)
        update = self.self_projection(x) + self.neighbor_projection(neighbors)
        return self.norm(x + self.dropout(F.gelu(update)))


class SGFormerGraphEncoder(nn.Module):
    """One-layer SGFormer adapted to a batch of 36-node fNIRS graphs.

    This follows Eq. (2)-(4) and the official implementation: one-head global
    linear attention with Frobenius-normalized Q/K, a residual self path, and
    a shallow local GCN branch mixed at the output.
    """

    def __init__(
        self,
        dimension: int,
        dropout: float,
        attention_residual_weight: float = 0.5,
        graph_weight: float = 0.8,
    ) -> None:
        super().__init__()
        if not 0.0 <= attention_residual_weight <= 1.0:
            raise ValueError("SGFormer attention residual weight must be in [0,1]")
        if not 0.0 <= graph_weight <= 1.0:
            raise ValueError("SGFormer graph weight must be in [0,1]")
        self.attention_residual_weight = float(attention_residual_weight)
        self.graph_weight = float(graph_weight)
        self.input_layer = nn.Sequential(
            nn.Linear(dimension, dimension),
            nn.LayerNorm(dimension),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.query = nn.Linear(dimension, dimension)
        self.key = nn.Linear(dimension, dimension)
        self.value = nn.Linear(dimension, dimension)
        self.global_norm = nn.LayerNorm(dimension)
        # SGFormer retains a shallow graph network in parallel with global
        # attention. This is one local graph-convolution block, not the former
        # two serial GCN blocks.
        self.local_gcn = DenseGraphConv(dimension, dropout)
        self.output_norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _frobenius_normalize(x: torch.Tensor) -> torch.Tensor:
        norm = torch.linalg.vector_norm(x, dim=(1, 2), keepdim=True)
        return x / norm.clamp_min(torch.finfo(x.dtype).eps)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected SGFormer input [B,N,D], got {tuple(x.shape)}")
        nodes = x.shape[1]
        z0 = self.input_layer(x)
        query = self._frobenius_normalize(self.query(z0))
        key = self._frobenius_normalize(self.key(z0))
        value = self.value(z0)

        # Official SGFormer linear attention, evaluated independently for each
        # graph in the batch. Algebraically equivalent to Eq. (3), but avoids
        # materializing the N x N all-pair matrix.
        key_value = torch.einsum("bnd,bne->bde", key, value)
        numerator = (
            torch.einsum("bnd,bde->bne", query, key_value)
            + nodes * value
        )
        key_sum = key.sum(dim=1)
        denominator = (
            torch.einsum("bnd,bd->bn", query, key_sum).unsqueeze(-1)
            + nodes
        )
        global_attention = numerator / denominator
        beta = self.attention_residual_weight
        global_output = self.global_norm(
            beta * global_attention + (1.0 - beta) * z0
        )
        global_output = self.dropout(global_output)

        local_output = self.local_gcn(z0, adjacency)
        alpha = self.graph_weight
        return self.output_norm(
            (1.0 - alpha) * global_output + alpha * local_output
        )


class FNIRSGraphPrompt(nn.Module):
    """Map one HbO/HbR trial graph to 36 node-wise prompt tokens."""

    def __init__(
        self,
        positions_3d: torch.Tensor,
        edge_index: torch.Tensor,
        chromophores: int,
        chromophore_encoder_mode: str = "separate_concat",
        prompt_stream_mode: str = "split_spatial_temporal",
        prompt_dimension: int = 200,
        graph_dimension: int = 128,
        dropout: float = 0.1,
        sgformer_attention_residual_weight: float = 0.5,
        sgformer_graph_weight: float = 0.8,
    ) -> None:
        super().__init__()
        if positions_3d.shape != (36, 3):
            raise ValueError(
                f"Expected 36 fNIRS positions, got {tuple(positions_3d.shape)}"
            )
        geometry = positions_3d.float()
        geometry = (geometry - geometry.mean(dim=0, keepdim=True)) / (
            geometry.std(dim=0, keepdim=True).clamp_min(1e-6)
        )
        self.register_buffer("positions_3d", geometry)
        self.register_buffer(
            "adjacency",
            normalized_adjacency(36, edge_index.long()),
        )
        self.chromophore_encoder_mode = chromophore_encoder_mode
        if prompt_stream_mode not in {"shared", "split_spatial_temporal"}:
            raise ValueError(f"Unknown prompt_stream_mode: {prompt_stream_mode}")
        self.prompt_stream_mode = prompt_stream_mode
        if chromophore_encoder_mode == "joint":
            self.temporal_encoder = nn.Sequential(
                nn.Conv1d(chromophores, 64, kernel_size=5, padding=2),
                nn.GroupNorm(8, 64),
                nn.GELU(),
                nn.Conv1d(64, graph_dimension, kernel_size=5, padding=2),
                nn.GroupNorm(8, graph_dimension),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.hbo_temporal_encoder = None
            self.hbr_temporal_encoder = None
        elif chromophore_encoder_mode == "separate_concat":
            if chromophores != 2:
                raise ValueError(
                    "separate_concat requires exactly two ordered channels: HbO, HbR"
                )
            if graph_dimension % 2 != 0 or (graph_dimension // 2) % 8 != 0:
                raise ValueError(
                    "separate_concat requires graph_dimension/2 divisible by 8"
                )
            branch_dimension = graph_dimension // 2

            def make_chromophore_encoder() -> nn.Sequential:
                # Two independent encoders with the same architecture but no
                # shared parameters. Keeping 64 hidden channels makes the
                # combined parameter count close to the former joint encoder.
                return nn.Sequential(
                    nn.Conv1d(1, 64, kernel_size=5, padding=2),
                    nn.GroupNorm(8, 64),
                    nn.GELU(),
                    nn.Conv1d(64, branch_dimension, kernel_size=5, padding=2),
                    nn.GroupNorm(8, branch_dimension),
                    nn.GELU(),
                    nn.AdaptiveAvgPool1d(1),
                )

            self.temporal_encoder = None
            self.hbo_temporal_encoder = make_chromophore_encoder()
            self.hbr_temporal_encoder = make_chromophore_encoder()
        else:
            raise ValueError(
                f"Unknown chromophore_encoder_mode: {chromophore_encoder_mode}"
            )
        self.geometry_encoder = nn.Sequential(
            nn.Linear(3, graph_dimension),
            nn.GELU(),
            nn.Linear(graph_dimension, graph_dimension),
        )
        self.node_embedding = nn.Embedding(36, graph_dimension)
        self.sgformer = SGFormerGraphEncoder(
            dimension=graph_dimension,
            dropout=dropout,
            attention_residual_weight=sgformer_attention_residual_weight,
            graph_weight=sgformer_graph_weight,
        )
        self.projection = nn.Sequential(
            nn.LayerNorm(graph_dimension),
            nn.Linear(graph_dimension, prompt_dimension),
        )

    def forward(
        self,
        fnirs: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if fnirs.ndim != 4 or fnirs.shape[1] != 36:
            raise ValueError(f"Expected fNIRS [B,36,M,T], got {tuple(fnirs.shape)}")
        batch, nodes, modalities, samples = fnirs.shape
        if self.chromophore_encoder_mode == "joint":
            temporal = self.temporal_encoder(
                fnirs.reshape(batch * nodes, modalities, samples)
            ).squeeze(-1)
        else:
            if modalities != 2:
                raise ValueError(
                    "separate_concat expects fNIRS channel order [HbO,HbR]"
                )
            hbo = fnirs[:, :, 0, :].reshape(batch * nodes, 1, samples)
            hbr = fnirs[:, :, 1, :].reshape(batch * nodes, 1, samples)
            hbo_features = self.hbo_temporal_encoder(hbo).squeeze(-1)
            hbr_features = self.hbr_temporal_encoder(hbr).squeeze(-1)
            # Fixed feature order: first HbO, then HbR. No mixing occurs until
            # after each chromophore has produced its own 64-D representation.
            temporal = torch.cat([hbo_features, hbr_features], dim=-1)
        temporal = temporal.reshape(batch, nodes, -1)
        geometry = self.geometry_encoder(self.positions_3d).unsqueeze(0)
        node_ids = torch.arange(nodes, device=fnirs.device)
        spatial_features = (
            temporal + geometry + self.node_embedding(node_ids).unsqueeze(0)
        )
        spatial_features = self.sgformer(spatial_features, self.adjacency)
        # Keep every graph node as a K/V memory token. Pooling before
        # cross-attention would leave a singleton memory whose softmax weight
        # is always one, degenerating the mechanism into a broadcast V bias.
        spatial_prompt = self.projection(spatial_features)
        if self.prompt_stream_mode == "shared":
            temporal_prompt = spatial_prompt
        else:
            # The temporal prompt contains only node-wise HbO/HbR temporal
            # features. It excludes geometry, node identity, and SGFormer.
            # Reusing the same 128->200 projection keeps parameter count fixed.
            temporal_prompt = self.projection(temporal)
        graph_representation = spatial_prompt.mean(dim=1)
        return (
            spatial_prompt,
            temporal_prompt,
            graph_representation,
            spatial_features,
            temporal,
        )


class GatedCrissCrossKVAdapter(nn.Module):
    """Parallel spatial/temporal K/V cross-attention for one CBraMod block."""

    def __init__(
        self,
        model_dimension: int = 200,
        heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if model_dimension % 2:
            raise ValueError("CBraMod model dimension must be even")
        self.half_dimension = model_dimension // 2
        half_heads = heads // 2
        self.spatial_k = nn.Linear(model_dimension, self.half_dimension, bias=False)
        self.spatial_v = nn.Linear(model_dimension, self.half_dimension, bias=False)
        self.temporal_k = nn.Linear(model_dimension, self.half_dimension, bias=False)
        self.temporal_v = nn.Linear(model_dimension, self.half_dimension, bias=False)
        self.cross_attention_s = nn.MultiheadAttention(
            self.half_dimension,
            half_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attention_t = nn.MultiheadAttention(
            self.half_dimension,
            half_heads,
            dropout=dropout,
            batch_first=True,
        )
        # Exact zero means the initial model is mathematically identical to
        # the pretrained CBraMod backbone.
        self.spatial_gate = nn.Parameter(torch.zeros(()))
        self.temporal_gate = nn.Parameter(torch.zeros(()))
        self.dropout = nn.Dropout(dropout)

    def gate_values(self) -> dict[str, float]:
        return {
            "spatial": float(torch.tanh(self.spatial_gate).detach().cpu()),
            "temporal": float(torch.tanh(self.temporal_gate).detach().cpu()),
        }

    def _prompt_update(
        self,
        normalized_x: torch.Tensor,
        spatial_prompt: torch.Tensor,
        temporal_prompt: torch.Tensor,
    ) -> torch.Tensor:
        batch, channels, patches, dimension = normalized_x.shape
        if dimension != self.half_dimension * 2:
            raise ValueError(
                f"Expected CBraMod dimension {self.half_dimension * 2}, got {dimension}"
            )
        if (
            spatial_prompt.ndim != 3
            or spatial_prompt.shape[0] != batch
            or spatial_prompt.shape[2] != dimension
        ):
            raise ValueError(
                "Expected spatial prompt "
                f"[B,K,{dimension}], got {tuple(spatial_prompt.shape)}"
            )
        if (
            temporal_prompt.ndim != 3
            or temporal_prompt.shape[0] != batch
            or temporal_prompt.shape[2] != dimension
        ):
            raise ValueError(
                "Expected temporal prompt "
                f"[B,K,{dimension}], got {tuple(temporal_prompt.shape)}"
            )
        spatial_prompt_count = spatial_prompt.shape[1]
        temporal_prompt_count = temporal_prompt.shape[1]
        spatial_query = (
            normalized_x[:, :, :, :self.half_dimension]
            .transpose(1, 2)
            .contiguous()
            .view(batch * patches, channels, self.half_dimension)
        )
        temporal_query = (
            normalized_x[:, :, :, self.half_dimension:]
            .contiguous()
            .view(batch * channels, patches, self.half_dimension)
        )
        spatial_key = (
            self.spatial_k(spatial_prompt)
            .unsqueeze(1)
            .expand(batch, patches, spatial_prompt_count, self.half_dimension)
            .reshape(batch * patches, spatial_prompt_count, self.half_dimension)
        )
        spatial_value = (
            self.spatial_v(spatial_prompt)
            .unsqueeze(1)
            .expand(batch, patches, spatial_prompt_count, self.half_dimension)
            .reshape(batch * patches, spatial_prompt_count, self.half_dimension)
        )
        temporal_key = (
            self.temporal_k(temporal_prompt)
            .unsqueeze(1)
            .expand(batch, channels, temporal_prompt_count, self.half_dimension)
            .reshape(batch * channels, temporal_prompt_count, self.half_dimension)
        )
        temporal_value = (
            self.temporal_v(temporal_prompt)
            .unsqueeze(1)
            .expand(batch, channels, temporal_prompt_count, self.half_dimension)
            .reshape(batch * channels, temporal_prompt_count, self.half_dimension)
        )
        spatial_update = self.cross_attention_s(
            spatial_query,
            spatial_key,
            spatial_value,
            need_weights=False,
        )[0]
        temporal_update = self.cross_attention_t(
            temporal_query,
            temporal_key,
            temporal_value,
            need_weights=False,
        )[0]
        spatial_update = (
            spatial_update
            .view(batch, patches, channels, self.half_dimension)
            .transpose(1, 2)
        )
        temporal_update = temporal_update.view(
            batch, channels, patches, self.half_dimension
        )
        spatial_update = torch.tanh(self.spatial_gate) * spatial_update
        temporal_update = torch.tanh(self.temporal_gate) * temporal_update
        return self.dropout(torch.cat((spatial_update, temporal_update), dim=-1))

    def forward(
        self,
        base_layer: nn.Module,
        x: torch.Tensor,
        spatial_prompt: torch.Tensor,
        temporal_prompt: torch.Tensor,
    ) -> torch.Tensor:
        normalized = base_layer.norm1(x)
        base_attention = base_layer._sa_block(
            normalized,
            attn_mask=None,
            key_padding_mask=None,
            is_causal=False,
        )
        x = x + base_attention + self._prompt_update(
            normalized, spatial_prompt, temporal_prompt
        )
        return x + base_layer._ff_block(base_layer.norm2(x))


class GatedCBraModGraphPrompt(nn.Module):
    """Frozen CBraMod plus last-block gated fNIRS K/V adapters."""

    def __init__(
        self,
        backbone: nn.Module,
        positions_3d: torch.Tensor,
        edge_index: torch.Tensor,
        chromophores: int,
        chromophore_encoder_mode: str,
        prompt_stream_mode: str,
        prompt_layer_indices: list[int],
        class_count: int = 2,
        graph_dimension: int = 128,
        dropout: float = 0.1,
        sgformer_attention_residual_weight: float = 0.5,
        sgformer_graph_weight: float = 0.8,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        layer_count = len(backbone.encoder.layers)
        if not prompt_layer_indices:
            raise ValueError("At least one prompt layer is required")
        if len(set(prompt_layer_indices)) != len(prompt_layer_indices):
            raise ValueError("Prompt layer indices must be unique")
        if min(prompt_layer_indices) < 0 or max(prompt_layer_indices) >= layer_count:
            raise ValueError(
                f"Prompt layers {prompt_layer_indices} outside 0..{layer_count - 1}"
            )
        self.prompt_layer_indices = tuple(sorted(prompt_layer_indices))
        self.graph_prompt = FNIRSGraphPrompt(
            positions_3d=positions_3d,
            edge_index=edge_index,
            chromophores=chromophores,
            chromophore_encoder_mode=chromophore_encoder_mode,
            prompt_stream_mode=prompt_stream_mode,
            prompt_dimension=200,
            graph_dimension=graph_dimension,
            dropout=dropout,
            sgformer_attention_residual_weight=sgformer_attention_residual_weight,
            sgformer_graph_weight=sgformer_graph_weight,
        )
        self.adapters = nn.ModuleDict({
            str(index): GatedCrissCrossKVAdapter(
                model_dimension=200,
                heads=8,
                dropout=dropout,
            )
            for index in self.prompt_layer_indices
        })
        # Shared official all-patch downstream head. Sharing avoids duplicating
        # its approximately 120M parameters for EEG and Fusion.
        self.classifier = nn.Sequential(
            nn.Linear(30 * 10 * 200, 10 * 200),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(10 * 200, 200),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(200, class_count),
        )
        self.fnirs_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(200, class_count),
        )

    def freeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def gate_values(self) -> dict[str, dict[str, float]]:
        return {
            index: adapter.gate_values()
            for index, adapter in self.adapters.items()
        }

    def _encode(
        self,
        eeg: torch.Tensor,
        spatial_prompt: torch.Tensor | None,
        temporal_prompt: torch.Tensor | None,
    ) -> torch.Tensor:
        if (spatial_prompt is None) != (temporal_prompt is None):
            raise ValueError("Spatial and temporal prompts must be both set or both None")
        features = self.backbone.patch_embedding(eeg)
        for index, layer in enumerate(self.backbone.encoder.layers):
            key = str(index)
            if spatial_prompt is not None and key in self.adapters:
                features = self.adapters[key](
                    layer, features, spatial_prompt, temporal_prompt
                )
            else:
                features = layer(features)
        return self.backbone.proj_out(features)

    def forward(
        self,
        eeg: torch.Tensor,
        fnirs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if eeg.ndim != 4 or tuple(eeg.shape[1:]) != (30, 10, 200):
            raise ValueError(f"Expected EEG [B,30,10,200], got {tuple(eeg.shape)}")
        (
            spatial_prompt,
            temporal_prompt,
            fnirs_representation,
            spatial_node_features,
            temporal_node_features,
        ) = self.graph_prompt(fnirs)
        eeg_features = self._encode(
            eeg, spatial_prompt=None, temporal_prompt=None
        )
        fusion_features = self._encode(
            eeg,
            spatial_prompt=spatial_prompt,
            temporal_prompt=temporal_prompt,
        )
        return {
            "eeg": self.classifier(eeg_features.flatten(start_dim=1)),
            "fnirs": self.fnirs_head(fnirs_representation),
            "fusion": self.classifier(fusion_features.flatten(start_dim=1)),
            "prompt": spatial_prompt,
            "spatial_prompt": spatial_prompt,
            "temporal_prompt": temporal_prompt,
            "node_features": spatial_node_features,
            "temporal_node_features": temporal_node_features,
        }

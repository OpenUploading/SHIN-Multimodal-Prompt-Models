"""fNIRS graph-conditioned gated K/V adapters for a frozen CBraMod backbone.

The official CBraMod source is not modified. Each selected CrissCross block
keeps its original spatial/temporal self-attention and adds a parallel,
zero-gated cross-attention whose K/V memory comes from an fNIRS graph prompt.
"""

from __future__ import annotations

import math

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


class TwoLayerGCNEncoder(nn.Module):
    """Two serial normalized graph-convolution blocks for encoder ablation."""

    def __init__(self, dimension: int, dropout: float) -> None:
        super().__init__()
        self.gcn1 = DenseGraphConv(dimension, dropout)
        self.gcn2 = DenseGraphConv(dimension, dropout)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        return self.gcn2(self.gcn1(x, adjacency), adjacency)


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


class SGFormerGlobalEncoder(nn.Module):
    """One-layer SGFormer global attention without an explicit GNN branch.

    The first NVE experiment deliberately tests whether structured node
    attributes plus all-pair attention are sufficient.  It therefore does not
    use triangulation edges or a parallel GCN, keeping the new factor isolated.
    """

    def __init__(
        self,
        dimension: int,
        dropout: float,
        attention_residual_weight: float = 0.5,
    ) -> None:
        super().__init__()
        if not 0.0 <= attention_residual_weight <= 1.0:
            raise ValueError("SGFormer attention residual weight must be in [0,1]")
        self.attention_residual_weight = float(attention_residual_weight)
        self.input_layer = nn.Sequential(
            nn.Linear(dimension, dimension),
            nn.LayerNorm(dimension),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.query = nn.Linear(dimension, dimension)
        self.key = nn.Linear(dimension, dimension)
        self.value = nn.Linear(dimension, dimension)
        self.output_norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _frobenius_normalize(x: torch.Tensor) -> torch.Tensor:
        norm = torch.linalg.vector_norm(x, dim=(1, 2), keepdim=True)
        return x / norm.clamp_min(torch.finfo(x.dtype).eps)

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected SGFormer input [B,N,D], got {tuple(x.shape)}")
        nodes = x.shape[1]
        z0 = self.input_layer(x)
        query = self._frobenius_normalize(self.query(z0))
        key = self._frobenius_normalize(self.key(z0))
        value = self.value(z0)
        key_value = torch.einsum("bnd,bne->bde", key, value)
        numerator = torch.einsum("bnd,bde->bne", query, key_value) + nodes * value
        denominator = (
            torch.einsum("bnd,bd->bn", query, key.sum(dim=1)).unsqueeze(-1)
            + nodes
        )
        attended = numerator / denominator.clamp_min(torch.finfo(x.dtype).eps)
        beta = self.attention_residual_weight
        return self.output_norm(
            beta * self.dropout(attended) + (1.0 - beta) * z0
        )


class TwoLayerSGFormerBottleneckEncoder(nn.Module):
    """Two global SGFormer layers separated by a residual bottleneck."""

    def __init__(
        self,
        dimension: int,
        dropout: float,
        attention_residual_weight: float = 0.5,
        bottleneck_dimension: int | None = None,
    ) -> None:
        super().__init__()
        bottleneck_dimension = bottleneck_dimension or dimension // 2
        if not 0 < bottleneck_dimension < dimension:
            raise ValueError(
                "SGFormer bottleneck dimension must be between zero and the "
                f"model dimension, got {bottleneck_dimension} for {dimension}"
            )
        self.bottleneck_dimension = int(bottleneck_dimension)
        self.layer1 = SGFormerGlobalEncoder(
            dimension=dimension,
            dropout=dropout,
            attention_residual_weight=attention_residual_weight,
        )
        self.bottleneck = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(dimension, self.bottleneck_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.bottleneck_dimension, dimension),
            nn.Dropout(dropout),
        )
        self.bottleneck_output_norm = nn.LayerNorm(dimension)
        self.layer2 = SGFormerGlobalEncoder(
            dimension=dimension,
            dropout=dropout,
            attention_residual_weight=attention_residual_weight,
        )

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor | None = None,
    ) -> torch.Tensor:
        first = self.layer1(x)
        bridged = self.bottleneck_output_norm(first + self.bottleneck(first))
        return self.layer2(bridged)


class FNIRSTAttention(nn.Module):
    """Published fNIRS-T attention: eight 64-D heads for a 64-D token."""

    def __init__(self, dimension: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.heads = heads
        self.head_dimension = 64
        inner = heads * self.head_dimension
        self.scale = self.head_dimension ** -0.5
        self.qkv = nn.Linear(dimension, inner * 3, bias=False)
        self.output = nn.Sequential(nn.Linear(inner, dimension), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        def heads(t: torch.Tensor) -> torch.Tensor:
            return t.reshape(batch, tokens, self.heads, self.head_dimension).transpose(1, 2)
        q, k, v = heads(q), heads(k), heads(v)
        attention = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        output = torch.matmul(attention.softmax(dim=-1), v)
        output = output.transpose(1, 2).reshape(batch, tokens, -1)
        return self.output(output)


class FNIRSTBlock(nn.Module):
    def __init__(self, dimension: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(dimension)
        self.attention = FNIRSTAttention(dimension, heads, dropout)
        self.norm_ffn = nn.LayerNorm(dimension)
        self.ffn = nn.Sequential(
            nn.Linear(dimension, dimension), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dimension, dimension), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm_attention(x))
        return x + self.ffn(self.norm_ffn(x))


class FNIRSTBackbone(nn.Module):
    """Feature-only adaptation of the published fNIRS-T dual-path backbone.

    Both published Conv2d stems use ``in_channels=2``. HbO and HbR are thus
    mixed from the first learned convolution rather than encoded separately.
    The classifier is removed and the two paths produce 36 node-wise 128-D
    tokens for MoPE.
    """

    def __init__(
        self,
        sampling_points: int,
        node_count: int = 36,
        token_dimension: int = 64,
        depth: int = 6,
        heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if sampling_points < 30:
            raise ValueError("fNIRS-T requires at least 30 temporal samples")
        embedded_width = ((sampling_points - 30) // 4 + 1) * 8
        self.patch_stem = nn.Conv2d(2, 8, kernel_size=(5, 30), stride=(1, 4))
        self.channel_stem = nn.Conv2d(2, 8, kernel_size=(1, 30), stride=(1, 4))
        self.patch_projection = nn.Sequential(
            nn.Linear(embedded_width, token_dimension), nn.LayerNorm(token_dimension)
        )
        self.channel_projection = nn.Sequential(
            nn.Linear(embedded_width, token_dimension), nn.LayerNorm(token_dimension)
        )
        patch_count = node_count - 5 + 1
        self.patch_position = nn.Parameter(
            torch.randn(1, patch_count, token_dimension) * 0.02
        )
        self.channel_position = nn.Parameter(
            torch.randn(1, node_count, token_dimension) * 0.02
        )
        self.patch_transformer = nn.Sequential(*[
            FNIRSTBlock(token_dimension, heads, dropout) for _ in range(depth)
        ])
        self.channel_transformer = nn.Sequential(*[
            FNIRSTBlock(token_dimension, heads, dropout) for _ in range(depth)
        ])
        self.output_norm = nn.LayerNorm(token_dimension * 2)

    @staticmethod
    def _tokens(x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 1, 3).flatten(start_dim=2)

    def forward(self, fnirs: torch.Tensor) -> torch.Tensor:
        if fnirs.ndim != 4 or fnirs.shape[1] != 2:
            raise ValueError(
                f"Expected fNIRS-T input [B,2,N,T], got {tuple(fnirs.shape)}"
            )
        patch = self.patch_projection(self._tokens(self.patch_stem(fnirs)))
        channel = self.channel_projection(self._tokens(self.channel_stem(fnirs)))
        patch = self.patch_transformer(
            patch + self.patch_position[:, :patch.shape[1]]
        )
        channel = self.channel_transformer(
            channel + self.channel_position[:, :channel.shape[1]]
        )
        patch_global = patch.mean(dim=1, keepdim=True).expand(
            -1, channel.shape[1], -1
        )
        return self.output_norm(torch.cat((channel, patch_global), dim=-1))


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


class FNIRSNVESGFormerPrompt(FNIRSGraphPrompt):
    """Explicit neurovascular-event attributes followed by global SGFormer.

    Spatial tokens are not learned directly from an opaque temporal encoder.
    Every node is first represented by 23 named HbO/HbR descriptors: four
    coarse temporal-bin means per chromophore, seven whole-trial event
    statistics per chromophore, and their Pearson coupling.  A small MLP maps
    this explicit intermediate representation to the SGFormer dimension.

    The temporal K/V stream remains the established independent HbO and HbR
    Conv1d encoding.  This makes the first experiment a focused test of the
    structured NVE representation in the spatial stream.
    """

    DESCRIPTOR_NAMES = tuple(
        [f"{chrom}_bin_{index}_mean" for chrom in ("hbo", "hbr") for index in range(4)]
        + [
            f"{chrom}_{name}"
            for chrom in ("hbo", "hbr")
            for name in (
                "mean", "std", "slope", "late_minus_early",
                "mean_abs_derivative", "maximum", "minimum",
            )
        ]
        + ["hbo_hbr_pearson_correlation"]
    )
    TEMPORAL_REGION_NAMES = (
        "prefrontal",
        "occipital",
        "left_anterior_sensorimotor",
        "left_posterior_sensorimotor",
        "right_anterior_sensorimotor",
        "right_posterior_sensorimotor",
    )
    # SHIN's fixed 36-channel fNIRS order. The six disjoint regions cover
    # every node exactly once and use no task labels or fitted statistics.
    TEMPORAL_REGION_NODE_INDICES = (
        tuple(range(0, 9)),
        tuple(range(9, 12)),
        (13, 14, 15, 16, 17, 21, 22),
        (12, 18, 19, 20, 23),
        (24, 26, 27, 28, 29, 34, 35),
        (25, 30, 31, 32, 33),
    )

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
        sgformer_graph_weight: float = 0.0,
        nve_spatial_encoder: str = "sgformer",
        temporal_prompt_mode: str = "node_summary",
    ) -> None:
        if chromophores != 2 or chromophore_encoder_mode != "separate_concat":
            raise ValueError(
                "NVE-SGFormer requires ordered HbO/HbR and separate temporal encoders"
            )
        super().__init__(
            positions_3d=positions_3d,
            edge_index=edge_index,
            chromophores=chromophores,
            chromophore_encoder_mode=chromophore_encoder_mode,
            prompt_stream_mode=prompt_stream_mode,
            prompt_dimension=prompt_dimension,
            graph_dimension=graph_dimension,
            dropout=dropout,
            sgformer_attention_residual_weight=sgformer_attention_residual_weight,
            sgformer_graph_weight=sgformer_graph_weight,
        )
        if temporal_prompt_mode not in {
            "node_summary", "aligned_10_tokens", "overlap_3s_10_tokens",
            "overlap_3s_region_time_tokens", "overlap_3s_node_time_tokens",
        }:
            raise ValueError(
                "temporal_prompt_mode must be 'node_summary', "
                "'aligned_10_tokens', 'overlap_3s_10_tokens', or "
                "'overlap_3s_region_time_tokens', or "
                "'overlap_3s_node_time_tokens', got "
                f"{temporal_prompt_mode!r}"
            )
        self.temporal_prompt_mode = temporal_prompt_mode
        if temporal_prompt_mode == "aligned_10_tokens":
            # Preserve the exact two Conv1d encoders and all their parameters;
            # only change the parameter-free final pooling from one whole-trial
            # summary to ten ordered one-second bins (100 samples at 10 Hz).
            self.hbo_temporal_encoder[-1] = nn.AdaptiveAvgPool1d(10)
            self.hbr_temporal_encoder[-1] = nn.AdaptiveAvgPool1d(10)
            position = torch.arange(10, dtype=torch.float32).unsqueeze(1)
            frequencies = torch.exp(
                torch.arange(0, graph_dimension, 2, dtype=torch.float32)
                * (-math.log(10000.0) / graph_dimension)
            )
            encoding = torch.zeros(10, graph_dimension, dtype=torch.float32)
            encoding[:, 0::2] = torch.sin(position * frequencies)
            encoding[:, 1::2] = torch.cos(position * frequencies)
            self.register_buffer(
                "temporal_position_encoding", encoding.unsqueeze(0)
            )
        else:
            if temporal_prompt_mode in {
                "overlap_3s_10_tokens", "overlap_3s_region_time_tokens",
                "overlap_3s_node_time_tokens",
            }:
                position = torch.arange(10, dtype=torch.float32).unsqueeze(1)
                frequencies = torch.exp(
                    torch.arange(0, graph_dimension, 2, dtype=torch.float32)
                    * (-math.log(10000.0) / graph_dimension)
                )
                encoding = torch.zeros(10, graph_dimension, dtype=torch.float32)
                encoding[:, 0::2] = torch.sin(position * frequencies)
                encoding[:, 1::2] = torch.cos(position * frequencies)
                self.register_buffer(
                    "temporal_position_encoding", encoding.unsqueeze(0)
                )
            else:
                self.register_buffer("temporal_position_encoding", None)
        if temporal_prompt_mode == "overlap_3s_region_time_tokens":
            region_count = len(self.TEMPORAL_REGION_NODE_INDICES)
            covered_nodes = sorted(
                node
                for region in self.TEMPORAL_REGION_NODE_INDICES
                for node in region
            )
            if covered_nodes != list(range(36)):
                raise RuntimeError(
                    "Temporal region definition must cover all 36 nodes exactly once"
                )
            region_position = torch.arange(
                region_count, dtype=torch.float32
            ).unsqueeze(1)
            region_frequencies = torch.exp(
                torch.arange(0, graph_dimension, 2, dtype=torch.float32)
                * (-math.log(10000.0) / graph_dimension)
            )
            region_encoding = torch.zeros(
                region_count, graph_dimension, dtype=torch.float32
            )
            region_encoding[:, 0::2] = torch.sin(
                region_position * region_frequencies
            )
            region_encoding[:, 1::2] = torch.cos(
                region_position * region_frequencies
            )
            self.register_buffer(
                "temporal_region_position_encoding",
                region_encoding.unsqueeze(0).unsqueeze(2),
            )
        else:
            self.register_buffer("temporal_region_position_encoding", None)
        if nve_spatial_encoder not in {
            "sgformer", "two_layer_sgformer_bottleneck", "hybrid_sgformer",
            "identity", "two_layer_gcn"
        }:
            raise ValueError(
                "nve_spatial_encoder must be 'sgformer', "
                "'two_layer_sgformer_bottleneck', 'hybrid_sgformer', "
                "'identity', or 'two_layer_gcn', got "
                f"{nve_spatial_encoder!r}"
            )
        self.nve_spatial_encoder = nve_spatial_encoder
        if nve_spatial_encoder == "sgformer":
            self.sgformer = SGFormerGlobalEncoder(
                dimension=graph_dimension,
                dropout=dropout,
                attention_residual_weight=sgformer_attention_residual_weight,
            )
        elif nve_spatial_encoder == "two_layer_sgformer_bottleneck":
            self.sgformer = TwoLayerSGFormerBottleneckEncoder(
                dimension=graph_dimension,
                bottleneck_dimension=graph_dimension // 2,
                dropout=dropout,
                attention_residual_weight=sgformer_attention_residual_weight,
            )
        elif nve_spatial_encoder == "hybrid_sgformer":
            self.sgformer = SGFormerGraphEncoder(
                dimension=graph_dimension,
                dropout=dropout,
                attention_residual_weight=sgformer_attention_residual_weight,
                graph_weight=sgformer_graph_weight,
            )
        elif nve_spatial_encoder == "two_layer_gcn":
            self.sgformer = TwoLayerGCNEncoder(graph_dimension, dropout)
        else:
            self.sgformer = nn.Identity()
        self.nve_projection = nn.Sequential(
            nn.LayerNorm(len(self.DESCRIPTOR_NAMES)),
            nn.Linear(len(self.DESCRIPTOR_NAMES), graph_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(graph_dimension, graph_dimension),
            nn.LayerNorm(graph_dimension),
        )

    @staticmethod
    def _event_descriptors(fnirs: torch.Tensor) -> torch.Tensor:
        # Input has already been normalized with train-subject statistics.
        _, _, modalities, samples = fnirs.shape
        if modalities != 2 or samples < 8:
            raise ValueError("NVE descriptors require [B,36,2,T] with T >= 8")
        per_chromophore: list[torch.Tensor] = []
        bin_means: list[torch.Tensor] = []
        time = torch.linspace(-1.0, 1.0, samples, device=fnirs.device, dtype=fnirs.dtype)
        slope_denominator = time.square().sum().clamp_min(1e-8)
        edge_width = max(1, samples // 10)
        for index in range(2):
            signal = fnirs[:, :, index, :]
            chunks = torch.tensor_split(signal, 4, dim=-1)
            bin_means.extend(chunk.mean(dim=-1) for chunk in chunks)
            centered = signal - signal.mean(dim=-1, keepdim=True)
            slope = (centered * time).sum(dim=-1) / slope_denominator
            late_minus_early = (
                signal[..., -edge_width:].mean(dim=-1)
                - signal[..., :edge_width].mean(dim=-1)
            )
            per_chromophore.extend((
                signal.mean(dim=-1),
                signal.std(dim=-1, unbiased=False),
                slope,
                late_minus_early,
                signal.diff(dim=-1).abs().mean(dim=-1),
                signal.amax(dim=-1),
                signal.amin(dim=-1),
            ))
        hbo = fnirs[:, :, 0, :]
        hbr = fnirs[:, :, 1, :]
        hbo_centered = hbo - hbo.mean(dim=-1, keepdim=True)
        hbr_centered = hbr - hbr.mean(dim=-1, keepdim=True)
        correlation = (hbo_centered * hbr_centered).mean(dim=-1) / (
            hbo_centered.square().mean(dim=-1).sqrt()
            * hbr_centered.square().mean(dim=-1).sqrt()
        ).clamp_min(1e-6)
        descriptors = torch.stack(bin_means + per_chromophore + [correlation], dim=-1)
        return torch.nan_to_num(descriptors, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _overlapping_three_second_windows(signal: torch.Tensor) -> torch.Tensor:
        """Return ten centered 3 s windows with a 1 s stride and zero padding.

        The trial is fixed to ten one-second positions. For a 10 Hz, 100-sample
        input, padding one second on both sides and unfolding 30-sample windows
        every 10 samples produces windows [-1,2), [0,3), ..., [8,11). The same
        ratio is used for any input whose sample count is divisible by ten.
        """
        if signal.ndim != 3 or signal.shape[1] != 1:
            raise ValueError(
                f"Expected chromophore signal [B*N,1,T], got {tuple(signal.shape)}"
            )
        samples = signal.shape[-1]
        if samples % 10 != 0:
            raise ValueError(
                "overlap_3s_10_tokens requires a ten-second input with an "
                f"integer number of samples per second, got T={samples}"
            )
        samples_per_second = samples // 10
        padded = F.pad(signal, (samples_per_second, samples_per_second), value=0.0)
        windows = padded.unfold(
            dimension=-1,
            size=3 * samples_per_second,
            step=samples_per_second,
        )
        if windows.shape[-2] != 10:
            raise RuntimeError(
                f"Expected ten overlapping windows, got {windows.shape[-2]}"
            )
        return windows

    def forward(
        self,
        fnirs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if fnirs.ndim != 4 or tuple(fnirs.shape[1:3]) != (36, 2):
            raise ValueError(f"Expected ordered fNIRS [B,36,2,T], got {tuple(fnirs.shape)}")
        batch, nodes, _, samples = fnirs.shape
        branch_mode = getattr(self, "prompt_branch_mode", "both")
        if branch_mode == "spatial_only" or self.prompt_stream_mode == "shared":
            # A shared stream reuses the NVE spatial Prompt in both adapters,
            # so the chromophore CNN path is deliberately not executed.
            # The placeholder preserves the diagnostic/output API only.
            temporal = fnirs.new_zeros(batch, nodes, self.projection[1].in_features)
        else:
            hbo = fnirs[:, :, 0, :].reshape(batch * nodes, 1, samples)
            hbr = fnirs[:, :, 1, :].reshape(batch * nodes, 1, samples)
            if self.temporal_prompt_mode in {
                "overlap_3s_10_tokens", "overlap_3s_region_time_tokens",
                "overlap_3s_node_time_tokens",
            }:
                hbo_windows = self._overlapping_three_second_windows(hbo)
                hbr_windows = self._overlapping_three_second_windows(hbr)
                window_samples = hbo_windows.shape[-1]
                hbo_features = self.hbo_temporal_encoder(
                    hbo_windows.reshape(batch * nodes * 10, 1, window_samples)
                ).squeeze(-1).reshape(batch, nodes, 10, -1)
                hbr_features = self.hbr_temporal_encoder(
                    hbr_windows.reshape(batch * nodes * 10, 1, window_samples)
                ).squeeze(-1).reshape(batch, nodes, 10, -1)
            else:
                hbo_features = self.hbo_temporal_encoder(hbo)
                hbr_features = self.hbr_temporal_encoder(hbr)
            if self.temporal_prompt_mode == "node_summary":
                temporal = torch.cat((
                    hbo_features.squeeze(-1),
                    hbr_features.squeeze(-1),
                ), dim=-1).reshape(batch, nodes, -1)
            elif self.temporal_prompt_mode == "overlap_3s_10_tokens":
                # [B,N,10,64] per chromophore -> ten ordered temporal tokens.
                # Node averaging stays identical to the former time-token
                # control, isolating the effect of the overlapping 3 s window.
                temporal = torch.cat(
                    (hbo_features, hbr_features), dim=-1
                ).mean(dim=1)
                temporal = temporal + self.temporal_position_encoding.to(
                    dtype=temporal.dtype
                )
            elif self.temporal_prompt_mode == "overlap_3s_node_time_tokens":
                # Preserve every node at every time position instead of
                # averaging nodes or grouping them into regions. Node-major
                # ordering gives 36 x 10 = 360 temporal K/V tokens. Explicit
                # node and time encodings keep both axes identifiable after
                # flattening, while the spatial SGFormer branch stays separate.
                node_time_features = torch.cat(
                    (hbo_features, hbr_features), dim=-1
                )
                node_position = self.node_embedding(
                    torch.arange(nodes, device=fnirs.device)
                ).view(1, nodes, 1, -1)
                node_time_features = (
                    node_time_features
                    + node_position
                    + self.temporal_position_encoding.to(
                        dtype=node_time_features.dtype
                    ).unsqueeze(1)
                )
                temporal = node_time_features.reshape(batch, nodes * 10, -1)
            elif self.temporal_prompt_mode == "overlap_3s_region_time_tokens":
                node_time_features = torch.cat(
                    (hbo_features, hbr_features), dim=-1
                )
                regional_time_features = torch.stack(
                    tuple(
                        node_time_features[:, region_indices, :, :].mean(dim=1)
                        for region_indices in self.TEMPORAL_REGION_NODE_INDICES
                    ),
                    dim=1,
                )
                regional_time_features = (
                    regional_time_features
                    + self.temporal_position_encoding.to(
                        dtype=regional_time_features.dtype
                    ).unsqueeze(1)
                    + self.temporal_region_position_encoding.to(
                        dtype=regional_time_features.dtype
                    )
                )
                # Region-major ordering: six regions x ten time positions.
                temporal = regional_time_features.reshape(batch, 60, -1)
            else:
                # [B*N,64,10] -> [B,N,10,64] for each chromophore.
                hbo_tokens = hbo_features.reshape(
                    batch, nodes, -1, 10
                ).permute(0, 1, 3, 2)
                hbr_tokens = hbr_features.reshape(
                    batch, nodes, -1, 10
                ).permute(0, 1, 3, 2)
                # Spatial information remains in the NVE path.  The temporal
                # path averages nodes to yield exactly ten ordered time tokens.
                temporal = torch.cat(
                    (hbo_tokens, hbr_tokens), dim=-1
                ).mean(dim=1)
                temporal = temporal + self.temporal_position_encoding.to(
                    dtype=temporal.dtype
                )

        nve_descriptors = self._event_descriptors(fnirs)
        geometry = self.geometry_encoder(self.positions_3d).unsqueeze(0)
        node_ids = torch.arange(nodes, device=fnirs.device)
        nve_nodes = (
            self.nve_projection(nve_descriptors)
            + geometry
            + self.node_embedding(node_ids).unsqueeze(0)
        )
        spatial_features = (
            self.sgformer(nve_nodes, self.adjacency)
            if self.nve_spatial_encoder in {
                "hybrid_sgformer", "two_layer_gcn"
            }
            else self.sgformer(nve_nodes)
        )
        spatial_prompt = self.projection(spatial_features)
        if branch_mode == "spatial_only":
            temporal_prompt = spatial_prompt.new_zeros(spatial_prompt.shape)
        else:
            temporal_prompt = (
                spatial_prompt
                if self.prompt_stream_mode == "shared"
                else self.projection(temporal)
            )
        graph_representation = spatial_prompt.mean(dim=1)
        return (
            spatial_prompt,
            temporal_prompt,
            graph_representation,
            spatial_features,
            temporal,
        )


class FNIRSCNNGlobalPrompt(FNIRSGraphPrompt):
    """Matched no-NVE control using independent HbO/HbR CNN node features.

    The temporal encoders, geometry, Node-ID embedding, prompt projection and
    Global-only SGFormer are kept identical to the NVE model.  The only
    conceptual change is that the spatial stream receives learned CNN node
    features instead of the 23 explicit NVE descriptors.
    """

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
        sgformer_graph_weight: float = 0.0,
    ) -> None:
        if chromophores != 2 or chromophore_encoder_mode != "separate_concat":
            raise ValueError(
                "CNN-SGFormer control requires ordered independent HbO/HbR encoders"
            )
        if sgformer_graph_weight != 0.0:
            raise ValueError("CNN-SGFormer matched control is Global-only")
        super().__init__(
            positions_3d=positions_3d,
            edge_index=edge_index,
            chromophores=chromophores,
            chromophore_encoder_mode=chromophore_encoder_mode,
            prompt_stream_mode=prompt_stream_mode,
            prompt_dimension=prompt_dimension,
            graph_dimension=graph_dimension,
            dropout=dropout,
            sgformer_attention_residual_weight=sgformer_attention_residual_weight,
            sgformer_graph_weight=sgformer_graph_weight,
        )
        # Replace the legacy global+local SGFormer with the exact same
        # Global-only encoder class used by the NVE main model.
        self.sgformer = SGFormerGlobalEncoder(
            dimension=graph_dimension,
            dropout=dropout,
            attention_residual_weight=sgformer_attention_residual_weight,
        )


class FNIRSGraphPromptMoPE(FNIRSGraphPrompt):
    """Three-component conditional prompt with MoPE dynamic experts.

    Reference: Conditional Prompt Tuning for Multimodal Fusion
    (arXiv:2312.03734) + the MoPE extension.  The prompt entering the SGFormer
    spatial stream is decomposed as

        P = P_static + tanh(g_dynamic) * P_dynamic + tanh(g_mapped) * P_mapped

    * P_static  = a learned per-node static prompt;
    * P_dynamic = dense routing of a trial-level fNIRS condition psi over K
      experts:  r = softmax(W_r psi / tau + noise),  P_d = sum_k r_k E_k;
    * P_mapped  = node-wise features from the fNIRS-T backbone.

    The temporal prompt stream and the gated K/V adapters (blocks 8-11) are
    unchanged.  All component gates are zero-initialised, so the initial
    model is identical to the frozen EEG baseline (the adapter gates are also
    zero), and the dynamic expert routing is optional via prompt_components.
    """

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
        mope_expert_count: int = 4,
        mope_temperature: float = 0.1,
        mope_router_noise_std: float = 0.00390625,
        mope_importance_threshold: float = 0.05,
        mope_condition_dim: int = 128,
        mope_top_k: int | None = None,
        prompt_components: str = "all",
    ) -> None:
        super().__init__(
            positions_3d=positions_3d,
            edge_index=edge_index,
            chromophores=chromophores,
            chromophore_encoder_mode=chromophore_encoder_mode,
            prompt_stream_mode=prompt_stream_mode,
            prompt_dimension=prompt_dimension,
            graph_dimension=graph_dimension,
            dropout=dropout,
            sgformer_attention_residual_weight=sgformer_attention_residual_weight,
            sgformer_graph_weight=sgformer_graph_weight,
        )
        if prompt_components not in {"all", "static_only"}:
            raise ValueError(
                f"Unknown prompt_components: {prompt_components!r}"
            )
        if mope_expert_count < 2 or mope_temperature <= 0:
            raise ValueError(
                "MoPE requires expert_count >= 2 and temperature > 0"
            )
        self.prompt_components = prompt_components
        self.expert_count = int(mope_expert_count)
        self.temperature = float(mope_temperature)
        self.router_noise_std = float(mope_router_noise_std)
        self.importance_threshold = float(mope_importance_threshold)
        if mope_top_k is not None and not 1 <= int(mope_top_k) <= self.expert_count:
            raise ValueError("mope_top_k must be between 1 and expert_count")
        self.top_k = None if mope_top_k is None else int(mope_top_k)
        self.node_count = 36

        # Replace the former independent HbO/HbR CNN + SGFormer path.  The
        # fNIRS-T stems consume [HbO,HbR] jointly and return T [B,36,128].
        self.temporal_encoder = None
        self.hbo_temporal_encoder = None
        self.hbr_temporal_encoder = None
        self.geometry_encoder = None
        self.node_embedding = None
        self.sgformer = None
        self.fnirs_t = FNIRSTBackbone(
            sampling_points=100,
            node_count=self.node_count,
            token_dimension=graph_dimension // 2,
            depth=6,
            heads=8,
            dropout=dropout,
        )

        # Trial-level condition encoder: node-wise temporal features are
        # pooled across nodes, then projected to the router condition space.
        self.condition_proj = nn.Sequential(
            nn.Linear(graph_dimension, mope_condition_dim),
            nn.LayerNorm(mope_condition_dim),
        )
        self.router = nn.Linear(mope_condition_dim, self.expert_count)
        self.prompt_experts = nn.Parameter(
            torch.empty(
                self.expert_count, self.node_count, graph_dimension
            )
        )
        self.static_prompt = nn.Parameter(
            torch.empty(self.node_count, graph_dimension)
        )
        # Zero-initialised component gates keep the initial model equivalent
        # to the original (temporal + geometry + node-id) prompt.
        self.gate_dynamic = nn.Parameter(torch.zeros(()))
        self.gate_mapped = nn.Parameter(torch.zeros(()))
        self._routing_scores: torch.Tensor | None = None

        nn.init.normal_(self.prompt_experts, std=0.02)
        nn.init.normal_(self.static_prompt, std=0.02)
        nn.init.normal_(self.router.weight, std=1e-3)
        nn.init.zeros_(self.router.bias)
        for module in self.condition_proj:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)

    def route(self, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Produce new sparse-capable expert coefficients for every trial.

        Softmax weights cannot be sparsified with L1 because their L1 norm is
        always one.  Sigmoid coefficients are nonnegative but unnormalised, so
        an L1 penalty can actually switch unnecessary experts off.
        """
        clean_logits = self.router(condition) / self.temperature
        routed_logits = clean_logits
        if self.training and self.router_noise_std > 0:
            routed_logits = routed_logits + (
                torch.randn_like(routed_logits) * self.router_noise_std
            )
        if self.top_k is None:
            return torch.sigmoid(routed_logits), torch.sigmoid(clean_logits)

        def topk_softmax(logits: torch.Tensor) -> torch.Tensor:
            probabilities = torch.softmax(logits, dim=-1)
            _, indices = torch.topk(probabilities, k=self.top_k, dim=-1)
            mask = torch.zeros_like(probabilities).scatter_(-1, indices, 1.0)
            selected = probabilities * mask
            return selected / selected.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        # Every forward pass recomputes a separate Top-K selection for each
        # trial. Only the selected experts receive task gradients for it.
        return topk_softmax(routed_logits), topk_softmax(clean_logits)

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
        _, nodes, modalities, samples = fnirs.shape
        if modalities != 2 or samples != 100:
            raise ValueError(
                "fNIRS-T MoPE expects [B,36,2,100] ordered as [HbO,HbR]"
            )
        # fNIRS-T expects modality before node: [B,2,36,100].
        temporal = self.fnirs_t(fnirs.permute(0, 2, 1, 3).contiguous())
        spatial_features = self.static_prompt.unsqueeze(0).expand(
            temporal.shape[0], -1, -1
        )
        if self.prompt_components == "all":
            # Trial-level condition from the pooled temporal features.
            condition = self.condition_proj(temporal.mean(dim=1))
            routing_scores, clean_scores = self.route(condition)
            self._routing_scores = clean_scores
            dynamic = torch.einsum(
                "bk,knd->bnd", routing_scores, self.prompt_experts
            )
            spatial_features = (
                spatial_features
                + torch.tanh(self.gate_dynamic) * dynamic
                + torch.tanh(self.gate_mapped) * temporal
            )
        else:
            self._routing_scores = None
            spatial_features = spatial_features + temporal
        spatial_prompt = self.projection(spatial_features)
        if self.prompt_stream_mode == "shared":
            temporal_prompt = spatial_prompt
        else:
            temporal_prompt = self.projection(temporal)
        graph_representation = spatial_prompt.mean(dim=1)
        return (
            spatial_prompt,
            temporal_prompt,
            graph_representation,
            spatial_features,
            temporal,
        )

    def importance_loss(self) -> torch.Tensor:
        """L1 penalty on per-trial expert coefficients."""
        if self._routing_scores is None:
            return self.router.weight.new_zeros(())
        if self.top_k is not None:
            # Top-K weights are renormalised to sum to one, so their L1 norm is
            # constant and must not be reported or added to the objective.
            return self.router.weight.new_zeros(())
        return self._routing_scores.abs().sum(dim=-1).mean()

    @torch.no_grad()
    def routing_statistics(self) -> dict[str, float]:
        if self._routing_scores is None:
            return {}
        scores = self._routing_scores.detach()
        mean_usage = scores.mean(dim=0)
        probabilities = scores / scores.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        entropy = -(
            probabilities.clamp_min(1e-8) * probabilities.clamp_min(1e-8).log()
        ).sum(dim=-1).mean()
        normalized_entropy = entropy / math.log(self.expert_count)
        coefficient_of_variation = (
            mean_usage.std(unbiased=False) / mean_usage.mean().clamp_min(1e-8)
        )
        return {
            "normalized_entropy": float(normalized_entropy.item()),
            "importance_cv": float(coefficient_of_variation.item()),
            "maximum_mean_usage": float(mean_usage.max().item()),
            "minimum_mean_usage": float(mean_usage.min().item()),
            "top_k": self.top_k,
            "mean_active_experts": float(
                (scores > 0).sum(dim=-1).float().mean().item()
            ),
        }

    def component_gate_values(self) -> dict[str, float]:
        return {
            "dynamic": float(torch.tanh(self.gate_dynamic).detach().cpu()),
            "mapped": float(torch.tanh(self.gate_mapped).detach().cpu()),
        }


class GatedCrissCrossKVAdapter(nn.Module):
    """Parallel spatial/temporal K/V cross-attention for one CBraMod block."""

    def __init__(
        self,
        model_dimension: int = 200,
        heads: int = 8,
        dropout: float = 0.1,
        temporal_kv_policy: str = "all",
        temporal_future_steps: int = 3,
    ) -> None:
        super().__init__()
        if model_dimension % 2:
            raise ValueError("CBraMod model dimension must be even")
        self.half_dimension = model_dimension // 2
        if temporal_kv_policy not in {"all", "current_and_future"}:
            raise ValueError(
                "temporal_kv_policy must be 'all' or 'current_and_future'"
            )
        if not 0 <= int(temporal_future_steps) <= 9:
            raise ValueError("temporal_future_steps must be in [0,9]")
        self.temporal_kv_policy = temporal_kv_policy
        self.temporal_future_steps = int(temporal_future_steps)
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

    def temporal_attention_mask(
        self,
        query_count: int,
        prompt_count: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Mask node-major 36x10 tokens to current/future fNIRS times.

        A True entry is blocked by ``nn.MultiheadAttention``. For EEG query
        time t, only fNIRS times t..t+future_steps are visible. All 36 nodes
        at an allowed time remain available.
        """
        if self.temporal_kv_policy == "all":
            return None
        if query_count != 10 or prompt_count != 36 * query_count:
            raise ValueError(
                "current_and_future temporal K/V requires ten EEG queries and "
                f"36x10 node-major Prompt tokens, got Q={query_count}, "
                f"K={prompt_count}"
            )
        query_time = torch.arange(query_count, device=device).unsqueeze(1)
        key_time = torch.arange(prompt_count, device=device).remainder(query_count)
        allowed = (
            (key_time >= query_time)
            & (key_time <= query_time + self.temporal_future_steps)
        )
        return ~allowed

    def _prompt_update(
        self,
        normalized_x: torch.Tensor,
        spatial_prompt: torch.Tensor,
        temporal_prompt: torch.Tensor,
        prompt_branch_mode: str = "both",
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
        if prompt_branch_mode not in {"both", "spatial_only", "temporal_only"}:
            raise ValueError(f"Unknown prompt_branch_mode: {prompt_branch_mode!r}")
        use_spatial = prompt_branch_mode in {"both", "spatial_only"}
        use_temporal = prompt_branch_mode in {"both", "temporal_only"}
        if use_spatial:
            spatial_prompt_count = spatial_prompt.shape[1]
            spatial_query = (
                normalized_x[:, :, :, :self.half_dimension]
                .transpose(1, 2).contiguous()
                .view(batch * patches, channels, self.half_dimension)
            )
            spatial_key = (
                self.spatial_k(spatial_prompt).unsqueeze(1)
                .expand(batch, patches, spatial_prompt_count, self.half_dimension)
                .reshape(batch * patches, spatial_prompt_count, self.half_dimension)
            )
            spatial_value = (
                self.spatial_v(spatial_prompt).unsqueeze(1)
                .expand(batch, patches, spatial_prompt_count, self.half_dimension)
                .reshape(batch * patches, spatial_prompt_count, self.half_dimension)
            )
            spatial_update = self.cross_attention_s(
                spatial_query, spatial_key, spatial_value, need_weights=False
            )[0]
            spatial_update = (
                spatial_update.view(batch, patches, channels, self.half_dimension)
                .transpose(1, 2)
            )
            spatial_update = torch.tanh(self.spatial_gate) * spatial_update
        else:
            spatial_update = normalized_x.new_zeros(
                batch, channels, patches, self.half_dimension
            )
        if use_temporal:
            temporal_prompt_count = temporal_prompt.shape[1]
            temporal_query = (
                normalized_x[:, :, :, self.half_dimension:].contiguous()
                .view(batch * channels, patches, self.half_dimension)
            )
            temporal_key = (
                self.temporal_k(temporal_prompt).unsqueeze(1)
                .expand(batch, channels, temporal_prompt_count, self.half_dimension)
                .reshape(batch * channels, temporal_prompt_count, self.half_dimension)
            )
            temporal_value = (
                self.temporal_v(temporal_prompt).unsqueeze(1)
                .expand(batch, channels, temporal_prompt_count, self.half_dimension)
                .reshape(batch * channels, temporal_prompt_count, self.half_dimension)
            )
            temporal_update = self.cross_attention_t(
                temporal_query,
                temporal_key,
                temporal_value,
                attn_mask=self.temporal_attention_mask(
                    patches, temporal_prompt_count, temporal_query.device
                ),
                need_weights=False,
            )[0].view(batch, channels, patches, self.half_dimension)
            temporal_update = torch.tanh(self.temporal_gate) * temporal_update
        else:
            temporal_update = normalized_x.new_zeros(
                batch, channels, patches, self.half_dimension
            )
        return self.dropout(torch.cat((spatial_update, temporal_update), dim=-1))

    def forward(
        self,
        base_layer: nn.Module,
        x: torch.Tensor,
        spatial_prompt: torch.Tensor,
        temporal_prompt: torch.Tensor,
        prompt_branch_mode: str = "both",
    ) -> torch.Tensor:
        normalized = base_layer.norm1(x)
        base_attention = base_layer._sa_block(
            normalized,
            attn_mask=None,
            key_padding_mask=None,
            is_causal=False,
        )
        x = x + base_attention + self._prompt_update(
            normalized, spatial_prompt, temporal_prompt, prompt_branch_mode
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
        prompt_generator: str = "original",
        mope_expert_count: int = 4,
        mope_temperature: float = 0.1,
        mope_router_noise_std: float = 0.00390625,
        mope_importance_threshold: float = 0.05,
        mope_condition_dim: int = 128,
        mope_top_k: int | None = None,
        prompt_components: str = "all",
        prompt_branch_mode: str = "both",
        nve_spatial_encoder: str = "sgformer",
        temporal_prompt_mode: str = "node_summary",
        temporal_kv_policy: str = "all",
        temporal_future_steps: int = 3,
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
        self.prompt_generator = prompt_generator
        if prompt_branch_mode not in {"both", "spatial_only", "temporal_only"}:
            raise ValueError(f"Unknown prompt_branch_mode: {prompt_branch_mode!r}")
        self.prompt_branch_mode = prompt_branch_mode
        prompt_kwargs = dict(
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
        if prompt_generator == "mope":
            self.graph_prompt = FNIRSGraphPromptMoPE(
                **prompt_kwargs,
                mope_expert_count=mope_expert_count,
                mope_temperature=mope_temperature,
                mope_router_noise_std=mope_router_noise_std,
                mope_importance_threshold=mope_importance_threshold,
                mope_condition_dim=mope_condition_dim,
                mope_top_k=mope_top_k,
                prompt_components=prompt_components,
            )
        elif prompt_generator == "nve_sgformer":
            self.graph_prompt = FNIRSNVESGFormerPrompt(
                **prompt_kwargs,
                nve_spatial_encoder=nve_spatial_encoder,
                temporal_prompt_mode=temporal_prompt_mode,
            )
        elif prompt_generator == "cnn_sgformer":
            self.graph_prompt = FNIRSCNNGlobalPrompt(**prompt_kwargs)
        elif prompt_generator == "original":
            self.graph_prompt = FNIRSGraphPrompt(**prompt_kwargs)
        else:
            raise ValueError(
                "prompt_generator must be 'original', 'mope', or "
                "'nve_sgformer', or 'cnn_sgformer', got "
                f"{prompt_generator!r}"
            )
        self.graph_prompt.prompt_branch_mode = prompt_branch_mode
        self.adapters = nn.ModuleDict({
            str(index): GatedCrissCrossKVAdapter(
                model_dimension=200,
                heads=8,
                dropout=dropout,
                temporal_kv_policy=temporal_kv_policy,
                temporal_future_steps=temporal_future_steps,
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
        self.enforce_disabled_branch_freezing()

    def enforce_disabled_branch_freezing(self) -> None:
        """Keep structurally disabled Prompt branches non-trainable."""
        if getattr(self.graph_prompt, "prompt_stream_mode", None) == "shared":
            # Shared mode routes the NVE spatial Prompt into both adapters;
            # the separate HbO/HbR temporal encoders are intentionally unused.
            for name in ("hbo_temporal_encoder", "hbr_temporal_encoder", "temporal_encoder"):
                module = getattr(self.graph_prompt, name, None)
                if module is not None:
                    for parameter in module.parameters():
                        parameter.requires_grad = False
        if self.prompt_branch_mode == "spatial_only":
            for name in ("hbo_temporal_encoder", "hbr_temporal_encoder", "temporal_encoder"):
                module = getattr(self.graph_prompt, name, None)
                if module is not None:
                    for parameter in module.parameters():
                        parameter.requires_grad = False
            for adapter in self.adapters.values():
                for module in (adapter.temporal_k, adapter.temporal_v, adapter.cross_attention_t):
                    for parameter in module.parameters():
                        parameter.requires_grad = False
                adapter.temporal_gate.requires_grad = False
        elif self.prompt_branch_mode == "temporal_only":
            for adapter in self.adapters.values():
                for module in (adapter.spatial_k, adapter.spatial_v, adapter.cross_attention_s):
                    for parameter in module.parameters():
                        parameter.requires_grad = False
                adapter.spatial_gate.requires_grad = False

    def freeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def gate_values(self) -> dict[str, dict[str, float]]:
        return {
            index: adapter.gate_values()
            for index, adapter in self.adapters.items()
        }

    def prompt_importance_loss(self) -> torch.Tensor:
        if hasattr(self.graph_prompt, "importance_loss"):
            return self.graph_prompt.importance_loss()
        return self.classifier[0].weight.new_zeros(())

    def routing_statistics(self) -> dict[str, float]:
        if hasattr(self.graph_prompt, "routing_statistics"):
            return self.graph_prompt.routing_statistics()
        return {}

    def prompt_component_gates(self) -> dict[str, float]:
        if hasattr(self.graph_prompt, "component_gate_values"):
            return self.graph_prompt.component_gate_values()
        return {}

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
                    layer, features, spatial_prompt, temporal_prompt,
                    self.prompt_branch_mode,
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

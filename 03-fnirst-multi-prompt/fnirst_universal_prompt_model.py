# -*- coding: utf-8 -*-
"""fNIRS-T prompts with a backbone-agnostic residual cross-attention adapter."""

from __future__ import annotations

import math

import torch
from torch import nn


class FNIRSTAttention(nn.Module):
    """Published fNIRS-T attention: eight 64-D heads for a 64-D token."""

    def __init__(self, dimension: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.heads = int(heads)
        self.head_dimension = 64
        inner = self.heads * self.head_dimension
        self.scale = self.head_dimension ** -0.5
        self.qkv = nn.Linear(dimension, inner * 3, bias=False)
        self.output = nn.Sequential(nn.Linear(inner, dimension), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def split_heads(value: torch.Tensor) -> torch.Tensor:
            return value.reshape(
                batch, tokens, self.heads, self.head_dimension
            ).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
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
            nn.Linear(dimension, dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dimension, dimension),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm_attention(x))
        return x + self.ffn(self.norm_ffn(x))


class FNIRSTBackbone(nn.Module):
    """Feature-only fNIRS-T with its channel, local-patch and joint outputs."""

    def __init__(
        self,
        sampling_points: int = 100,
        node_count: int = 36,
        token_dimension: int = 64,
        depth: int = 6,
        heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if sampling_points < 30:
            raise ValueError("fNIRS-T requires at least 30 temporal samples")
        self.sampling_points = int(sampling_points)
        self.node_count = int(node_count)
        embedded_width = ((sampling_points - 30) // 4 + 1) * 8
        # Original fNIRS-T behavior: HbO and HbR are mixed immediately.
        self.patch_stem = nn.Conv2d(2, 8, kernel_size=(5, 30), stride=(1, 4))
        self.channel_stem = nn.Conv2d(2, 8, kernel_size=(1, 30), stride=(1, 4))
        self.patch_projection = nn.Sequential(
            nn.Linear(embedded_width, token_dimension),
            nn.LayerNorm(token_dimension),
        )
        self.channel_projection = nn.Sequential(
            nn.Linear(embedded_width, token_dimension),
            nn.LayerNorm(token_dimension),
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

    def forward(self, fnirs: torch.Tensor) -> dict[str, torch.Tensor]:
        expected = (2, self.node_count, self.sampling_points)
        if fnirs.ndim != 4 or tuple(fnirs.shape[1:]) != expected:
            raise ValueError(
                f"Expected complete fNIRS [B,{expected[0]},{expected[1]},"
                f"{expected[2]}], got {tuple(fnirs.shape)}"
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
        joint = self.output_norm(torch.cat((channel, patch_global), dim=-1))
        return {"channel": channel, "patch": patch, "joint": joint}


class SingleUnifiedPrompt(nn.Module):
    """Step 1: one 36-token prompt from the standard fNIRS-T joint output."""

    def __init__(
        self,
        model_dimension: int,
        dropout: float,
        sampling_points: int = 100,
    ) -> None:
        super().__init__()
        self.fnirs_t = FNIRSTBackbone(
            sampling_points=sampling_points, dropout=dropout
        )
        self.projection = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, model_dimension),
        )

    def forward(self, fnirs: torch.Tensor) -> dict[str, torch.Tensor | None]:
        features = self.fnirs_t(fnirs)
        prompt = self.projection(features["joint"])
        return {
            "prompt": prompt,
            "token_log_prior": None,
            "router_weights": None,
            "representation": prompt.mean(dim=1),
            "features": features,
        }

    def regularization_loss(self) -> torch.Tensor:
        return self.projection[1].weight.new_zeros(())

    def routing_statistics(self) -> dict[str, float | list[float]]:
        return {}


class LearnedContextSamplePrompt(nn.Module):
    """Merged task/universal context concatenated with trial fNIRS tokens.

    The context tokens are ordinary trainable parameters shared by every trial
    within one task run.  They intentionally merge the roles of a universal
    prompt and a task-adaptation prompt, which avoids trying to identify two
    static prompt groups from the small SHIN split.  The sample tokens remain
    explicitly dynamic and are generated from the current 0--10 s fNIRS trial.

    Channel/patch features are internal to fNIRS-T.  Only its 36 joint tokens
    are exposed as sample prompts, so the external decomposition has the clear
    form [learned context ; current-sample evidence].
    """

    def __init__(
        self,
        model_dimension: int,
        dropout: float,
        context_tokens: int = 8,
        sampling_points: int = 100,
    ) -> None:
        super().__init__()
        if context_tokens < 1:
            raise ValueError("context_tokens must be positive")
        self.fnirs_t = FNIRSTBackbone(
            sampling_points=sampling_points, dropout=dropout
        )
        self.context_tokens = int(context_tokens)
        self.learned_context = nn.Parameter(
            torch.empty(self.context_tokens, model_dimension)
        )
        self.context_type = nn.Parameter(torch.empty(1, 1, model_dimension))
        self.sample_type = nn.Parameter(torch.empty(1, 1, model_dimension))
        self.sample_projection = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, model_dimension),
        )
        nn.init.normal_(self.learned_context, std=0.02)
        nn.init.normal_(self.context_type, std=0.02)
        nn.init.normal_(self.sample_type, std=0.02)

    def forward(self, fnirs: torch.Tensor) -> dict[str, torch.Tensor | None]:
        features = self.fnirs_t(fnirs)
        sample_prompt = (
            self.sample_projection(features["joint"]) + self.sample_type
        )
        context_prompt = (
            self.learned_context.unsqueeze(0).expand(fnirs.shape[0], -1, -1)
            + self.context_type
        )
        prompt = torch.cat((context_prompt, sample_prompt), dim=1)
        return {
            "prompt": prompt,
            "token_log_prior": None,
            "router_weights": None,
            # The auxiliary fNIRS head must be driven by current-trial evidence,
            # not by the static context that is identical for every sample.
            "representation": sample_prompt.mean(dim=1),
            "features": features,
        }

    def regularization_loss(self) -> torch.Tensor:
        return self.learned_context.new_zeros(())

    def routing_statistics(self) -> dict[str, float | list[float]]:
        return {}


class WeightedMultiPrompt(nn.Module):
    """Step 2: channel, local-patch and joint prompts with sample-wise weights.

    The three sources are not replicas:
      * channel: 36 channel-wise fNIRS-T tokens;
      * patch: 32 overlapping five-node local-patch tokens;
      * joint: 36 channel tokens augmented by the global patch representation.

    Tokens are concatenated.  The router weight is converted to a log attention
    prior, normalized by each source's token count so a longer source does not
    receive more total prior mass merely because it contains more tokens.
    """

    SOURCE_NAMES = ("channel", "patch", "joint")
    SOURCE_LENGTHS = (36, 32, 36)

    def __init__(
        self,
        model_dimension: int,
        dropout: float,
        router_hidden: int = 32,
        router_temperature: float = 1.0,
        entropy_lambda: float = 0.001,
        sampling_points: int = 100,
    ) -> None:
        super().__init__()
        if router_temperature <= 0:
            raise ValueError("router_temperature must be positive")
        if entropy_lambda < 0:
            raise ValueError("entropy_lambda must be nonnegative")
        self.fnirs_t = FNIRSTBackbone(
            sampling_points=sampling_points, dropout=dropout
        )
        self.channel_projection = nn.Sequential(
            nn.LayerNorm(64), nn.Linear(64, model_dimension)
        )
        self.patch_projection = nn.Sequential(
            nn.LayerNorm(64), nn.Linear(64, model_dimension)
        )
        self.joint_projection = nn.Sequential(
            nn.LayerNorm(128), nn.Linear(128, model_dimension)
        )
        self.router = nn.Sequential(
            nn.LayerNorm(64 + 64 + 128),
            nn.Linear(64 + 64 + 128, router_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(router_hidden, len(self.SOURCE_NAMES)),
        )
        # Exact uniform initialization makes Step 2 begin as an unbiased concat.
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)
        self.router_temperature = float(router_temperature)
        self.entropy_lambda = float(entropy_lambda)
        self._last_weights: torch.Tensor | None = None

    def forward(self, fnirs: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.fnirs_t(fnirs)
        prompts = (
            self.channel_projection(features["channel"]),
            self.patch_projection(features["patch"]),
            self.joint_projection(features["joint"]),
        )
        condition = torch.cat([
            features["channel"].mean(dim=1),
            features["patch"].mean(dim=1),
            features["joint"].mean(dim=1),
        ], dim=-1)
        weights = torch.softmax(
            self.router(condition) / self.router_temperature, dim=-1
        )
        self._last_weights = weights
        prompt = torch.cat(prompts, dim=1)
        log_prior_parts = []
        for source, token_count in enumerate(self.SOURCE_LENGTHS):
            source_log_mass = torch.log(weights[:, source].clamp_min(1e-8))
            per_token_log_mass = source_log_mass - math.log(token_count)
            log_prior_parts.append(
                per_token_log_mass.unsqueeze(1).expand(-1, token_count)
            )
        token_log_prior = torch.cat(log_prior_parts, dim=1)
        representation = sum(
            weights[:, index:index + 1] * item.mean(dim=1)
            for index, item in enumerate(prompts)
        )
        return {
            "prompt": prompt,
            "token_log_prior": token_log_prior,
            "router_weights": weights,
            "representation": representation,
            "features": features,
        }

    def regularization_loss(self) -> torch.Tensor:
        if self._last_weights is None:
            return self.router[-1].weight.new_zeros(())
        entropy = -(
            self._last_weights
            * torch.log(self._last_weights.clamp_min(1e-8))
        ).sum(dim=-1).mean()
        return self.entropy_lambda * entropy

    def routing_statistics(self) -> dict[str, float | list[float]]:
        if self._last_weights is None:
            return {}
        weights = self._last_weights.detach()
        effective = 1.0 / weights.square().sum(dim=-1)
        return {
            "mean": weights.mean(dim=0).cpu().tolist(),
            "std": weights.std(dim=0, unbiased=False).cpu().tolist(),
            "min": weights.min(dim=0).values.cpu().tolist(),
            "max": weights.max(dim=0).values.cpu().tolist(),
            "effective_prompt_count_mean": float(effective.mean().cpu()),
        }


class UniversalResidualPromptAdapter(nn.Module):
    """Generic Q=EEG, K/V=fNIRS residual adapter over flattened tokens."""

    def __init__(
        self,
        dimension: int = 200,
        heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.heads = int(heads)
        self.query_norm = nn.LayerNorm(dimension)
        self.prompt_norm = nn.LayerNorm(dimension)
        self.attention = nn.MultiheadAttention(
            dimension, heads, dropout=dropout, batch_first=True
        )
        self.output_norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        eeg_tokens: torch.Tensor,
        prompt: torch.Tensor,
        token_log_prior: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if eeg_tokens.ndim != 3 or prompt.ndim != 3:
            raise ValueError("Adapter expects EEG and Prompt as [B,N,D]")
        attention_mask = None
        if token_log_prior is not None:
            if token_log_prior.shape != prompt.shape[:2]:
                raise ValueError(
                    f"Bad token prior {token_log_prior.shape} for {prompt.shape}"
                )
            batch, queries, _ = eeg_tokens.shape
            attention_mask = token_log_prior[:, None, None, :].expand(
                batch, self.heads, queries, prompt.shape[1]
            ).reshape(batch * self.heads, queries, prompt.shape[1])
        update, _ = self.attention(
            query=self.query_norm(eeg_tokens),
            key=self.prompt_norm(prompt),
            value=self.prompt_norm(prompt),
            attn_mask=attention_mask,
            need_weights=False,
        )
        update = self.dropout(self.output_norm(update))
        return eeg_tokens + torch.tanh(self.gate) * update


def official_all_patch_classifier(
    dimension: int = 200,
    channels: int = 30,
    patches: int = 10,
    class_count: int = 2,
    dropout: float = 0.1,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(channels * patches * dimension, 10 * dimension),
        nn.ELU(),
        nn.Dropout(dropout),
        nn.Linear(10 * dimension, dimension),
        nn.ELU(),
        nn.Dropout(dropout),
        nn.Linear(dimension, class_count),
    )


class FrozenCBraModFNIRSTPrompt(nn.Module):
    """Fixed CBraMod/head plus a selected universal fNIRS-T prompt."""

    def __init__(
        self,
        backbone: nn.Module,
        classifier: nn.Module,
        prompt_mode: str,
        dropout: float = 0.1,
        router_hidden: int = 32,
        router_temperature: float = 1.0,
        entropy_lambda: float = 0.001,
        context_tokens: int = 8,
        fnirs_sampling_points: int = 100,
    ) -> None:
        super().__init__()
        if prompt_mode == "single":
            self.prompt_generator = SingleUnifiedPrompt(
                200, dropout, fnirs_sampling_points
            )
        elif prompt_mode == "multi_weighted":
            self.prompt_generator = WeightedMultiPrompt(
                200,
                dropout,
                router_hidden,
                router_temperature,
                entropy_lambda,
                fnirs_sampling_points,
            )
        elif prompt_mode == "learned_context_sample":
            self.prompt_generator = LearnedContextSamplePrompt(
                200, dropout, context_tokens, fnirs_sampling_points
            )
        else:
            raise ValueError(
                "prompt_mode must be 'single', 'multi_weighted', or "
                "'learned_context_sample'"
            )
        self.prompt_mode = prompt_mode
        self.fnirs_sampling_points = int(fnirs_sampling_points)
        self.backbone = backbone
        self.classifier = classifier
        self.adapter = UniversalResidualPromptAdapter(200, 8, dropout)
        self.fnirs_head = nn.Sequential(
            nn.LayerNorm(200), nn.Dropout(dropout), nn.Linear(200, 2)
        )
        self.freeze_reference()

    def freeze_reference(self) -> None:
        for module in (self.backbone, self.classifier):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad = False

    def train(self, mode: bool = True) -> "FrozenCBraModFNIRSTPrompt":
        super().train(mode)
        self.backbone.eval()
        self.classifier.eval()
        return self

    def _encode_eeg(self, eeg: torch.Tensor) -> torch.Tensor:
        features = self.backbone.patch_embedding(eeg)
        for layer in self.backbone.encoder.layers:
            features = layer(features)
        return self.backbone.proj_out(features)

    def forward(self, eeg: torch.Tensor, fnirs: torch.Tensor) -> dict[str, torch.Tensor]:
        if eeg.ndim != 4 or tuple(eeg.shape[1:]) != (30, 10, 200):
            raise ValueError(f"Expected EEG [B,30,10,200], got {tuple(eeg.shape)}")
        expected_fnirs = (36, 2, self.fnirs_sampling_points)
        if fnirs.ndim != 4 or tuple(fnirs.shape[1:]) != expected_fnirs:
            raise ValueError(
                f"Expected fNIRS [B,{expected_fnirs[0]},{expected_fnirs[1]},"
                f"{expected_fnirs[2]}], got {tuple(fnirs.shape)}"
            )
        encoded = self._encode_eeg(eeg)
        eeg_tokens = encoded.flatten(start_dim=1, end_dim=2)
        prompt_output = self.prompt_generator(
            fnirs.permute(0, 2, 1, 3).contiguous()
        )
        fusion_tokens = self.adapter(
            eeg_tokens,
            prompt_output["prompt"],
            prompt_output["token_log_prior"],
        )
        eeg_logits = self.classifier(encoded.flatten(start_dim=1))
        fusion_logits = self.classifier(fusion_tokens.flatten(start_dim=1))
        return {
            "eeg": eeg_logits,
            "fnirs": self.fnirs_head(prompt_output["representation"]),
            "fusion": fusion_logits,
            "prompt": prompt_output["prompt"],
            "router_weights": prompt_output["router_weights"],
            "eeg_tokens": eeg_tokens,
            "fusion_tokens": fusion_tokens,
        }

    def regularization_loss(self) -> torch.Tensor:
        return self.prompt_generator.regularization_loss()

    def routing_statistics(self) -> dict[str, float | list[float]]:
        return self.prompt_generator.routing_statistics()

    def gate_value(self) -> float:
        return float(torch.tanh(self.adapter.gate).detach().cpu())

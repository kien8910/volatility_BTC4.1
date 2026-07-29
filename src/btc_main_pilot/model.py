from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .config import MainPilotConfig


def sinusoidal_encoding(length: int, d_model: int, device: torch.device) -> torch.Tensor:
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    divisor = torch.exp(
        torch.arange(0, d_model, 2, device=device, dtype=torch.float32)
        * (-math.log(10_000.0) / d_model)
    )
    encoding = torch.zeros(length, d_model, device=device)
    encoding[:, 0::2] = torch.sin(position * divisor)
    encoding[:, 1::2] = torch.cos(position * divisor)
    return encoding


class PreNormSelfAttentionBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model, heads, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(x)
        attended, _ = self.attention(
            normalized, normalized, normalized, need_weights=False
        )
        x = x + self.dropout1(attended)
        return x + self.dropout2(self.ffn(self.norm2(x)))


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.query_norm = nn.LayerNorm(d_model)
        self.context_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model, heads, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if key_padding_mask is not None:
            if key_padding_mask.shape != context.shape[:2]:
                raise ValueError("key_padding_mask shape does not match news context")
            if bool(key_padding_mask.all(dim=1).any()):
                raise AssertionError("At least one attention row has every news key masked")
            if bool(key_padding_mask[:, -1].any()):
                raise AssertionError("The learned null-news token must remain unmasked")
        attended, _ = self.attention(
            self.query_norm(query),
            self.context_norm(context),
            self.context_norm(context),
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        output = query + self.dropout1(attended)
        return output + self.dropout2(self.ffn(self.norm2(output)))


class MainPatchTST(nn.Module):
    def __init__(
        self,
        config: MainPilotConfig,
        target_mean: float,
        target_scale: float,
        unconditional_mean_rv: float,
        forecast_queries: int,
    ):
        super().__init__()
        d = config.d_model
        self.config = config
        self.forecast_query_count = forecast_queries
        self.register_buffer(
            "target_mean", torch.tensor(float(target_mean), dtype=torch.float64)
        )
        self.register_buffer(
            "target_scale", torch.tensor(float(target_scale), dtype=torch.float64)
        )
        self.fine_projection = nn.Linear(config.fine_patch_length + 1, d)
        self.coarse_projection = nn.Linear(config.coarse_patch_length + 1, d)
        self.channel_embedding = nn.Embedding(config.channels, d)
        self.scale_embedding = nn.Embedding(2, d)
        self.market_time_projection = nn.Linear(4, d, bias=False)
        self.market_blocks = nn.ModuleList(
            [
                PreNormSelfAttentionBlock(
                    d, config.attention_heads, config.ffn_dim, config.dropout
                )
                for _ in range(config.patch_layers)
            ]
        )
        self.channel_gate = nn.Linear(d, 1)

        slow_fast_input = config.pca_dim + 3
        self.news_slow_fast_projection = nn.Linear(slow_fast_input, d)
        self.news_scalar_projection = nn.Linear(config.daily_scalars, d, bias=False)
        self.news_type_embedding = nn.Embedding(2, d)
        self.news_blocks = nn.ModuleList(
            [
                PreNormSelfAttentionBlock(
                    d, config.attention_heads, config.ffn_dim, config.dropout
                )
                for _ in range(config.news_layers)
            ]
        )
        self.null_news_token = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.normal_(self.null_news_token, std=0.02)

        self.cross_blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    d, config.attention_heads, config.ffn_dim, config.dropout
                )
                for _ in range(config.cross_layers)
            ]
        )
        self.forecast_queries = nn.Parameter(torch.empty(1, forecast_queries, d))
        nn.init.normal_(self.forecast_queries, std=0.02)
        self.pool_attention = nn.MultiheadAttention(
            d, config.attention_heads, dropout=config.dropout, batch_first=True
        )
        self.pool_projection: nn.Module
        if forecast_queries == 4:
            self.pool_projection = nn.Linear(4 * d, d)
        elif forecast_queries == 1:
            self.pool_projection = nn.Identity()
        else:
            raise ValueError("Forecast query count must be exactly 4 or 1")
        self.forecast_head = nn.Sequential(
            nn.Linear(d + 2, 32),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(32, 1),
        )
        output_layer = self.forecast_head[-1]
        assert isinstance(output_layer, nn.Linear)
        nn.init.zeros_(output_layer.weight)
        initial_z = (
            math.log(float(unconditional_mean_rv)) - float(target_mean)
        ) / float(target_scale)
        nn.init.constant_(output_layer.bias, initial_z)

    def _encode_market_scale(
        self,
        patches: torch.Tensor,
        patch_logrv: torch.Tensor,
        time_features: torch.Tensor,
        scale_id: int,
    ) -> torch.Tensor:
        # patches: [B,C,N,P], shared patch scalar: [B,N]
        batch, channels, tokens, _ = patches.shape
        metadata = patch_logrv[:, None, :, None].expand(-1, channels, -1, -1)
        input_patch = torch.cat([patches, metadata], dim=-1)
        projection = self.fine_projection if scale_id == 0 else self.coarse_projection
        embedded = projection(input_patch)
        position = sinusoidal_encoding(tokens, self.config.d_model, patches.device)
        channel_ids = torch.arange(channels, device=patches.device)
        embedded = (
            embedded
            + position[None, None, :, :]
            + self.channel_embedding(channel_ids)[None, :, None, :]
            + self.scale_embedding.weight[scale_id][None, None, None, :]
            + self.market_time_projection(time_features)[:, None, :, :]
        )
        encoded = embedded.reshape(batch * channels, tokens, -1)
        for block in self.market_blocks:
            encoded = block(encoded)
        encoded = encoded.reshape(batch, channels, tokens, -1)
        gate = torch.softmax(self.channel_gate(encoded).squeeze(-1), dim=1)
        return torch.sum(encoded * gate.unsqueeze(-1), dim=1)

    def _encode_news(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        semantic_slow = batch["semantic_slow"]
        semantic_fast = batch["semantic_fast"]
        sentiment_slow = batch["sentiment_slow"]
        sentiment_fast = batch["sentiment_fast"]
        scalars = batch["daily_scalars"]
        slow = self.news_slow_fast_projection(
            torch.cat([semantic_slow, sentiment_slow], dim=-1)
        )
        fast = self.news_slow_fast_projection(
            torch.cat([semantic_fast, sentiment_fast], dim=-1)
        ) + self.news_scalar_projection(scalars)
        length = slow.shape[1]
        position = sinusoidal_encoding(length, self.config.d_model, slow.device)
        slow = slow + position[None, :, :] + self.news_type_embedding.weight[0]
        fast = fast + position[None, :, :] + self.news_type_embedding.weight[1]
        interleaved = torch.stack([slow, fast], dim=2).reshape(
            slow.shape[0], length * 2, -1
        )
        for block in self.news_blocks:
            interleaved = block(interleaved)
        null = self.null_news_token.expand(interleaved.shape[0], -1, -1)
        return torch.cat([interleaved, null], dim=1)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        fine = self._encode_market_scale(
            batch["fine"],
            batch["fine_patch_logrv"],
            batch["fine_time"],
            scale_id=0,
        )
        coarse = self._encode_market_scale(
            batch["coarse"],
            batch["coarse_patch_logrv"],
            batch["coarse_time"],
            scale_id=1,
        )
        market = torch.cat([fine, coarse], dim=1)
        news = self._encode_news(batch)
        for block in self.cross_blocks:
            market = block(market, news)
        queries = self.forecast_queries.expand(market.shape[0], -1, -1)
        pooled, _ = self.pool_attention(
            queries, market, market, need_weights=False
        )
        if self.forecast_query_count == 4:
            pooled = pooled.reshape(pooled.shape[0], -1)
        else:
            pooled = pooled[:, 0, :]
        latent = self.pool_projection(pooled)
        forecast_input = torch.cat([latent, batch["head_scalars"]], dim=-1)
        forecast_z = self.forecast_head(forecast_input).squeeze(-1)
        predicted_log_rv = (
            self.target_mean + self.target_scale * forecast_z.double()
        )
        predicted_rv = torch.exp(predicted_log_rv)
        return {
            "forecast_z": forecast_z,
            "predicted_log_rv": predicted_log_rv,
            "predicted_rv": predicted_rv,
            "market_attention_output": market,
        }


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


@dataclass
class BuiltModel:
    model: MainPatchTST
    parameter_count: int
    forecast_queries: int


def build_model(
    config: MainPilotConfig,
    target_mean: float,
    target_scale: float,
    unconditional_mean_rv: float,
) -> BuiltModel:
    model = MainPatchTST(
        config,
        target_mean,
        target_scale,
        unconditional_mean_rv,
        forecast_queries=4,
    )
    count = trainable_parameter_count(model)
    queries = 4
    if count > config.parameter_budget:
        model = MainPatchTST(
            config,
            target_mean,
            target_scale,
            unconditional_mean_rv,
            forecast_queries=1,
        )
        count = trainable_parameter_count(model)
        queries = 1
    if not 20_000 <= count <= config.parameter_budget:
        raise AssertionError(
            f"Trainable parameter count {count} outside locked 20k-60k budget"
        )
    return BuiltModel(model=model, parameter_count=count, forecast_queries=queries)


def attention_backend_metadata(device: torch.device) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "implementation": "torch.nn.MultiheadAttention",
        "need_weights": False,
        "dispatch": "PyTorch SDPA automatic dispatch when eligible",
        "device": str(device),
    }
    if device.type == "cuda":
        metadata.update(
            {
                "flash_attention_available": bool(
                    torch.backends.cuda.is_flash_attention_available()
                ),
                "flash_sdp_enabled": bool(torch.backends.cuda.flash_sdp_enabled()),
                "memory_efficient_sdp_enabled": bool(
                    torch.backends.cuda.mem_efficient_sdp_enabled()
                ),
                "math_sdp_enabled": bool(torch.backends.cuda.math_sdp_enabled()),
                "device_name": torch.cuda.get_device_name(device),
            }
        )
    else:
        metadata["selected_backend"] = "CPU math attention"
    return metadata


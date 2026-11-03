"""Locally initialized, decoder-only Transformer for the SYNTRA brain.

This module depends only on PyTorch and the Python standard library. It never
downloads weights, contacts a model registry, or selects a hosted inference API.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

BOS_ID = 256
EOS_ID = 257
MIN_VOCAB_SIZE = 258


@dataclass(frozen=True)
class TransformerConfig:
    """Architecture settings saved alongside a model checkpoint."""

    vocab_size: int = 1024
    block_size: int = 1024
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    bias: bool = True

    def validate(self) -> None:
        integer_limits = {
            "vocab_size": (self.vocab_size, MIN_VOCAB_SIZE, 8192),
            "block_size": (self.block_size, 16, 16384),
            "n_layer": (self.n_layer, 1, 24),
            "n_head": (self.n_head, 1, 16),
            "n_embd": (self.n_embd, 32, 1024),
        }
        for name, (value, minimum, maximum) in integer_limits.items():
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be an integer between {minimum} and {maximum}.")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head.")
        if isinstance(self.dropout, bool) or not isinstance(self.dropout, (float, int)):
            raise ValueError("dropout must be a number between 0 and 0.5.")
        if not math.isfinite(float(self.dropout)) or not 0.0 <= self.dropout < 0.5:
            raise ValueError("dropout must be a finite number between 0 and 0.5.")
        if type(self.bias) is not bool:
            raise ValueError("bias must be a boolean.")

    def to_dict(self) -> dict:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "TransformerConfig":
        if not isinstance(value, dict):
            raise ValueError("Transformer configuration must be a JSON object.")
        accepted = {key: value[key] for key in cls.__dataclass_fields__ if key in value}
        config = cls(**accepted)
        config.validate()
        return config


def select_device(preference: str = "auto") -> torch.device:
    """Select CPU, CUDA, or Apple MPS without requiring a third-party runtime."""
    if preference not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("Device must be one of: auto, cpu, cuda, mps.")

    cuda_available = torch.cuda.is_available()
    mps_backend = getattr(torch.backends, "mps", None)
    mps_available = bool(mps_backend and mps_backend.is_available())

    if preference == "auto":
        if cuda_available:
            return torch.device("cuda")
        if mps_available:
            return torch.device("mps")
        return torch.device("cpu")
    if preference == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was requested but is not available in this PyTorch installation.")
    if preference == "mps" and not mps_available:
        raise RuntimeError("Apple MPS was requested but is not available on this device.")
    return torch.device(preference)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with an optimized SDPA path."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.attn_dropout = float(config.dropout)
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.projection = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, sequence, channels = x.shape
        qkv = self.qkv(x).view(batch, sequence, 3, self.n_head, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        if hasattr(F, "scaled_dot_product_attention"):
            attended = F.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=self.attn_dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            # Compatibility path for older supported PyTorch builds.
            scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_dim)
            causal_mask = torch.ones(
                sequence, sequence, device=x.device, dtype=torch.bool
            ).tril()
            scores = scores.masked_fill(~causal_mask, torch.finfo(scores.dtype).min)
            weights = F.softmax(scores, dim=-1)
            weights = F.dropout(
                weights,
                p=self.attn_dropout,
                training=self.training,
            )
            attended = weights @ value

        attended = attended.transpose(1, 2).contiguous().view(batch, sequence, channels)
        return self.residual_dropout(self.projection(attended))


class FeedForward(nn.Module):
    """Position-wise feed-forward network used by each Transformer block."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TransformerBlock(nn.Module):
    """Pre-layer-normalized Transformer block."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(config.n_embd)
        self.attention = CausalSelfAttention(config)
        self.norm_feed_forward = nn.LayerNorm(config.n_embd)
        self.feed_forward = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm_attention(x))
        x = x + self.feed_forward(self.norm_feed_forward(x))
        return x


class LocalTransformer(nn.Module):
    """A complete causal language model with locally random-initialized weights.

    Input and target tensors use shape [batch, sequence] and torch.long token IDs.
    The return value is (logits, loss); loss is None when targets are omitted.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_layer)
        )
        self.final_norm = nn.LayerNorm(config.n_embd)
        self.language_model_head = nn.Linear(
            config.n_embd, config.vocab_size, bias=False
        )
        # Tied embeddings reduce parameters and are standard for decoder LMs.
        self.language_model_head.weight = self.token_embedding.weight
        self.apply(self._initialize_weights)

        # Scale residual projections to keep deep stacks well-conditioned at init.
        residual_std = 0.02 / math.sqrt(2 * config.n_layer)
        for name, parameter in self.named_parameters():
            if name.endswith("attention.projection.weight") or name.endswith(
                "feed_forward.layers.2.weight"
            ):
                nn.init.normal_(parameter, mean=0.0, std=residual_std)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @property
    def parameter_count(self) -> int:
        """Number of trainable parameters (tied weights counted once)."""
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError("input_ids must be a torch.Tensor.")
        if input_ids.ndim != 2 or input_ids.shape[0] < 1 or input_ids.shape[1] < 1:
            raise ValueError("input_ids must have shape [batch, sequence] with non-empty dimensions.")
        if input_ids.dtype != torch.long:
            raise ValueError("input_ids must use torch.long token IDs.")
        if input_ids.shape[1] > self.config.block_size:
            raise ValueError(
                f"Input sequence length {input_ids.shape[1]} exceeds block_size {self.config.block_size}."
            )
        if torch.any(input_ids < 0) or torch.any(input_ids >= self.config.vocab_size):
            raise ValueError("input_ids contains a token outside the configured vocabulary.")
        if targets is not None:
            if not isinstance(targets, torch.Tensor) or targets.shape != input_ids.shape:
                raise ValueError("targets must be a tensor with the same shape as input_ids.")
            if targets.dtype != torch.long:
                raise ValueError("targets must use torch.long token IDs.")
            if targets.device != input_ids.device:
                raise ValueError("targets and input_ids must be on the same device.")

        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]
        x = self.embedding_dropout(x)
        for block in self.blocks:
            x = block(x)
        logits = self.language_model_head(self.final_norm(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
            )
        return logits, loss


__all__ = [
    "BOS_ID",
    "EOS_ID",
    "MIN_VOCAB_SIZE",
    "TransformerConfig",
    "select_device",
    "CausalSelfAttention",
    "FeedForward",
    "TransformerBlock",
    "LocalTransformer",
]

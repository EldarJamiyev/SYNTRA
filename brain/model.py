"""Locally initialized, decoder-only Transformer for the SYNTRA brain.

This module depends only on PyTorch and the Python standard library. It never
downloads weights, contacts a model registry, or selects a hosted inference API.
Supports classic LayerNorm/GELU/Absolute-Position blocks as well as optional
RMSNorm, Rotary Position Embeddings (RoPE), SwiGLU, and dense embedding pooling.
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


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True)
        x_normed = x.float() * torch.rsqrt(norm + self.eps)
        return (self.weight * x_normed).type_as(x)


def _apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    head_dim: int,
    theta: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    half = head_dim // 2
    freq_seq = torch.arange(0, half, dtype=torch.float32, device=q.device)
    inv_freq = 1.0 / (theta ** (freq_seq / half))
    angles = positions.float()[:, None] * inv_freq[None, :]
    cos = torch.cos(angles)[None, None, :, :]
    sin = torch.sin(angles)[None, None, :, :]

    def rotate(tensor: torch.Tensor) -> torch.Tensor:
        x1 = tensor[..., :half]
        x2 = tensor[..., half : 2 * half]
        rotated = torch.cat((-x2, x1), dim=-1)
        return (tensor.float() * cos.repeat(1, 1, 1, 2) + rotated.float() * sin.repeat(1, 1, 1, 2)).type_as(tensor)

    return rotate(q), rotate(k)


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
    use_rope: bool = False
    use_rmsnorm: bool = False
    use_swiglu: bool = False

    @property
    def seq_len(self) -> int:
        return self.block_size

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
        if type(self.use_rope) is not bool or type(self.use_rmsnorm) is not bool or type(self.use_swiglu) is not bool:
            raise ValueError("use_rope, use_rmsnorm, and use_swiglu must be booleans.")
        if self.use_rope and (self.n_embd // self.n_head) % 2 != 0:
            raise ValueError("head_dim (n_embd // n_head) must be even when use_rope is enabled.")

    def to_dict(self) -> dict:
        self.validate()
        result = {
            "vocab_size": self.vocab_size,
            "block_size": self.block_size,
            "n_layer": self.n_layer,
            "n_head": self.n_head,
            "n_embd": self.n_embd,
            "dropout": self.dropout,
            "bias": self.bias,
        }
        if self.use_rope:
            result["use_rope"] = True
        if self.use_rmsnorm:
            result["use_rmsnorm"] = True
        if self.use_swiglu:
            result["use_swiglu"] = True
        return result

    @classmethod
    def from_dict(cls, value: dict) -> "TransformerConfig":
        if not isinstance(value, dict):
            raise ValueError("Transformer configuration must be a JSON object.")
        normalized = dict(value)
        if "block_size" not in normalized and "seq_len" in normalized:
            normalized["block_size"] = normalized["seq_len"]
        accepted = {key: normalized[key] for key in cls.__dataclass_fields__ if key in normalized}
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
    """Multi-head causal self-attention with an optimized SDPA path and optional RoPE."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.use_rope = config.use_rope
        self.attn_dropout = float(config.dropout)
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.projection = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ):
        batch, sequence, channels = x.shape
        qkv = self.qkv(x).view(batch, sequence, 3, self.n_head, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        past_length = 0

        if past_key_value is not None:
            past_key, past_value = past_key_value
            expected_prefix = (batch, self.n_head)
            if (
                past_key.ndim != 4
                or past_value.shape != past_key.shape
                or past_key.shape[:2] != expected_prefix
                or past_key.shape[-1] != self.head_dim
                or past_key.device != key.device
                or past_value.device != value.device
            ):
                raise ValueError("Cached attention keys/values do not match the current input.")
            past_length = past_key.shape[-2]

        if self.use_rope:
            query_positions = torch.arange(past_length, past_length + sequence, device=x.device)
            query, key = _apply_rotary_emb(query, key, query_positions, self.head_dim)

        if past_key_value is not None:
            past_key, past_value = past_key_value
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if hasattr(F, "scaled_dot_product_attention"):
            if past_key_value is None:
                attended = F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    dropout_p=self.attn_dropout if self.training else 0.0,
                    is_causal=True,
                )
            else:
                key_positions = torch.arange(key.shape[-2], device=x.device)
                query_positions = past_length + torch.arange(sequence, device=x.device)
                causal_mask = key_positions[None, :] <= query_positions[:, None]
                attended = F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=causal_mask[None, None, :, :],
                    dropout_p=self.attn_dropout if self.training else 0.0,
                    is_causal=False,
                )
        else:
            scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_dim)
            key_positions = torch.arange(key.shape[-2], device=x.device)
            query_positions = past_length + torch.arange(sequence, device=x.device)
            causal_mask = key_positions[None, :] <= query_positions[:, None]
            scores = scores.masked_fill(
                ~causal_mask[None, None, :, :],
                torch.finfo(scores.dtype).min,
            )
            weights = F.softmax(scores, dim=-1)
            weights = F.dropout(weights, p=self.attn_dropout, training=self.training)
            attended = weights @ value

        attended = attended.transpose(1, 2).contiguous().view(batch, sequence, channels)
        output = self.residual_dropout(self.projection(attended))
        if use_cache:
            return output, (key, value)
        return output


class FeedForward(nn.Module):
    """Position-wise feed-forward network used by each Transformer block."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.use_swiglu = config.use_swiglu
        if self.use_swiglu:
            hidden = ((int((8 * config.n_embd) / 3) + 7) // 8) * 8
            self.w_gate = nn.Linear(config.n_embd, hidden, bias=config.bias)
            self.w_up = nn.Linear(config.n_embd, hidden, bias=config.bias)
            self.w_down = nn.Linear(hidden, config.n_embd, bias=config.bias)
            self.dropout = nn.Dropout(config.dropout)
        else:
            self.layers = nn.Sequential(
                nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
                nn.GELU(),
                nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
                nn.Dropout(config.dropout),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_swiglu:
            return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))
        return self.layers(x)


class TransformerBlock(nn.Module):
    """Pre-layer-normalized Transformer block."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        norm_cls = RMSNorm if config.use_rmsnorm else nn.LayerNorm
        self.norm_attention = norm_cls(config.n_embd)
        self.attention = CausalSelfAttention(config)
        self.norm_feed_forward = norm_cls(config.n_embd)
        self.feed_forward = FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ):
        attention_result = self.attention(
            self.norm_attention(x),
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        if use_cache:
            attention_output, present_key_value = attention_result
            x = x + attention_output
            x = x + self.feed_forward(self.norm_feed_forward(x))
            return x, present_key_value
        x = x + attention_result
        return x + self.feed_forward(self.norm_feed_forward(x))


class LocalTransformer(nn.Module):
    """A complete causal language model with locally random-initialized weights.

    Input and target tensors use shape [batch, sequence] and torch.long token IDs.
    The return value is (logits, loss); loss is None when targets are omitted.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        norm_cls = RMSNorm if config.use_rmsnorm else nn.LayerNorm

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_layer)
        )
        self.final_norm = norm_cls(config.n_embd)
        self.language_model_head = nn.Linear(
            config.n_embd, config.vocab_size, bias=False
        )
        # Tied embeddings reduce parameters and are standard for decoder LMs.
        self.language_model_head.weight = self.token_embedding.weight
        self.apply(self._initialize_weights)

        # Scale residual projections to keep deep stacks well-conditioned at init.
        residual_std = 0.02 / math.sqrt(2 * config.n_layer)
        for name, parameter in self.named_parameters():
            if (
                name.endswith("attention.projection.weight")
                or name.endswith("feed_forward.layers.2.weight")
                or name.endswith("feed_forward.w_down.weight")
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

    def _forward_hidden(
        self,
        input_ids: torch.Tensor,
        past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
        use_cache: bool = False,
    ):
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError("input_ids must be a torch.Tensor.")
        if input_ids.ndim != 2 or input_ids.shape[0] < 1 or input_ids.shape[1] < 1:
            raise ValueError("input_ids must have shape [batch, sequence] with non-empty dimensions.")
        if input_ids.dtype != torch.long:
            raise ValueError("input_ids must use torch.long token IDs.")
        if torch.any(input_ids < 0) or torch.any(input_ids >= self.config.vocab_size):
            raise ValueError("input_ids contains a token outside the configured vocabulary.")

        if past_key_values is not None and len(past_key_values) != len(self.blocks):
            raise ValueError("The cache must contain one key/value pair per Transformer block.")

        past_length = 0
        if past_key_values:
            past_length = past_key_values[0][0].shape[-2]
            for key, value in past_key_values:
                if key.ndim != 4 or value.shape != key.shape:
                    raise ValueError("Cached attention tensors have invalid dimensions.")
                if key.shape[0] != input_ids.shape[0] or key.shape[1] != self.config.n_head:
                    raise ValueError("Cached attention tensors do not match the current batch/head dimensions.")
                if key.device != input_ids.device or value.device != input_ids.device:
                    raise ValueError("Cached attention tensors must be on the same device as input_ids.")
                if key.shape[-2] != past_length:
                    raise ValueError("Cached attention layers have inconsistent sequence lengths.")

        if past_length + input_ids.shape[1] > self.config.block_size:
            raise ValueError(
                f"Input plus cached tokens exceeds block_size {self.config.block_size}."
            )

        positions = torch.arange(
            past_length,
            past_length + input_ids.shape[1],
            device=input_ids.device,
        )
        if self.config.use_rope:
            x = self.token_embedding(input_ids)
        else:
            x = self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]
        x = self.embedding_dropout(x)
        presents = []
        for index, block in enumerate(self.blocks):
            past = past_key_values[index] if past_key_values is not None else None
            if use_cache:
                x, present = block(x, past_key_value=past, use_cache=True)
                presents.append(present)
            else:
                x = block(x, past_key_value=past, use_cache=False)
        hidden = self.final_norm(x)
        if use_cache:
            return hidden, tuple(presents)
        return hidden

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
        use_cache: bool = False,
    ):
        if targets is not None:
            if not isinstance(targets, torch.Tensor) or targets.shape != input_ids.shape:
                raise ValueError("targets must be a tensor with the same shape as input_ids.")
            if targets.dtype != torch.long:
                raise ValueError("targets must use torch.long token IDs.")
            if targets.device != input_ids.device:
                raise ValueError("targets and input_ids must be on the same device.")
            valid_targets = targets != -100
            if torch.any((targets < 0) & valid_targets) or torch.any(targets >= self.config.vocab_size):
                raise ValueError("targets contains a token outside the configured vocabulary.")
            if not torch.any(valid_targets):
                raise ValueError("targets must contain at least one supervised token (not -100).")

        if use_cache:
            hidden, presents = self._forward_hidden(input_ids, past_key_values=past_key_values, use_cache=True)
        else:
            hidden = self._forward_hidden(input_ids, past_key_values=past_key_values, use_cache=False)
        logits = self.language_model_head(hidden)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                ignore_index=-100,
            )
        if use_cache:
            return logits, loss, tuple(presents)
        return logits, loss

    @torch.no_grad()
    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute L2-normalized semantic embeddings from pooled final hidden states."""
        self.eval()
        clipped = input_ids[:, -self.config.block_size:]
        hidden = self._forward_hidden(clipped, use_cache=False)
        mean_pooled = hidden.mean(dim=1)
        last_token = hidden[:, -1, :]
        combined = 0.7 * mean_pooled + 0.3 * last_token
        return F.normalize(combined.float(), p=2, dim=-1)


__all__ = [
    "BOS_ID",
    "EOS_ID",
    "MIN_VOCAB_SIZE",
    "RMSNorm",
    "TransformerConfig",
    "select_device",
    "CausalSelfAttention",
    "FeedForward",
    "TransformerBlock",
    "LocalTransformer",
]

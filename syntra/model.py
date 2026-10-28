"""SYNTRA decoder-only transformer implemented in PyTorch.

The architecture and forward pass are defined in this repository. PyTorch
provides tensor operations, optimized kernels, and automatic differentiation;
no pretrained architecture weights are loaded.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from syntra.config import SyntraConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, config: SyntraConfig) -> None:
        super().__init__()
        if config.n_embd % config.n_head:
            raise ValueError("n_embd must be divisible by n_head.")
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.projection = nn.Linear(config.n_embd, config.n_embd)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)
        mask = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, config.block_size, config.block_size), persistent=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, length, channels = inputs.shape
        qkv = self.qkv(inputs).view(
            batch, length, 3, self.n_head, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        allowed = self.causal_mask[:, :, :length, :length]
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=-1)
        weights = self.attention_dropout(weights)
        attended = weights @ value
        attended = attended.transpose(1, 2).contiguous().view(batch, length, channels)
        return self.residual_dropout(self.projection(attended))


class FeedForward(nn.Module):
    def __init__(self, config: SyntraConfig) -> None:
        super().__init__()
        hidden = 4 * config.n_embd
        self.layers = nn.Sequential(
            nn.Linear(config.n_embd, hidden),
            nn.GELU(),
            nn.Linear(hidden, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class TransformerBlock(nn.Module):
    def __init__(self, config: SyntraConfig) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(config.n_embd)
        self.attention = CausalSelfAttention(config)
        self.norm_feedforward = nn.LayerNorm(config.n_embd)
        self.feedforward = FeedForward(config)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs + self.attention(self.norm_attention(inputs))
        inputs = inputs + self.feedforward(self.norm_feedforward(inputs))
        return inputs


class SyntraModel(nn.Module):
    """Small GPT-style language model with randomly initialized weights."""

    def __init__(self, config: SyntraConfig) -> None:
        super().__init__()
        self.config = config.validate()
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.final_norm = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        token_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence].")
        batch, length = token_ids.shape
        if length > self.config.block_size:
            raise ValueError("Input sequence exceeds block_size=" + str(self.config.block_size) + ".")
        positions = torch.arange(length, device=token_ids.device)
        hidden = self.token_embedding(token_ids) + self.position_embedding(positions)
        hidden = self.embedding_dropout(hidden)
        for block in self.blocks:
            hidden = block(hidden)
        logits = self.lm_head(self.final_norm(hidden))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(batch * length, self.config.vocab_size),
                targets.reshape(batch * length),
            )
        return logits, loss

    def parameter_count(self, trainable_only: bool = True) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if not trainable_only or parameter.requires_grad
        )

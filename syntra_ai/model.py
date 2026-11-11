"""A decoder-only Transformer implemented for SYNTRA.
All parameters are initialized locally; no pretrained weights are fetched or loaded.
Supports classic LayerNorm/GELU/Absolute-Position blocks as well as modern
RMSNorm, Rotary Position Embeddings (RoPE), SwiGLU, nucleus (top-p) sampling,
repetition penalties, and dense hidden-state representation pooling.
"""
import math
from functools import wraps
import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig

BOS_ID = 256
EOS_ID = 257


def _preserve_module_modes(method):
    """Restore every module's exact training flag after temporary inference mode."""
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        module_modes = tuple((module, module.training) for module in self.modules())
        try:
            return method(self, *args, **kwargs)
        finally:
            # Calling self.train(previous_state) would overwrite intentionally mixed
            # child-module states. Restore the snapshot one module at a time instead.
            for module, was_training in module_modes:
                module.training = was_training

    return wrapped


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


class SwiGLUFeedForward(nn.Module):
    """Gated Linear Unit with SiLU activation for stronger code representation."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        hidden = int((8 * config.n_embd) / 3)
        hidden = ((hidden + 7) // 8) * 8
        bias = getattr(config, "bias", True)
        self.w_gate = nn.Linear(config.n_embd, hidden, bias=bias)
        self.w_up = nn.Linear(config.n_embd, hidden, bias=bias)
        self.w_down = nn.Linear(hidden, config.n_embd, bias=bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.use_rope = getattr(config, "use_rope", False)
        bias = getattr(config, "bias", True)
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=bias)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=bias)
        self.attn_dropout = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ):
        batch, length, channels = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.n_head, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        past_length = 0

        if past_key_value is not None:
            past_key, past_value = past_key_value
            if past_key.shape[:2] != k.shape[:2] or past_key.shape[-1] != k.shape[-1]:
                raise ValueError("Cached attention keys do not match the current batch/head dimensions.")
            past_length = past_key.shape[-2]
            if past_value.shape != past_key.shape:
                raise ValueError("Cached attention keys and values have inconsistent shapes.")

        if self.use_rope:
            query_positions = torch.arange(past_length, past_length + length, device=x.device)
            q, k = _apply_rotary_emb(q, k, query_positions, self.head_dim)

        if past_key_value is not None:
            past_key, past_value = past_key_value
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)

        if past_key_value is None:
            attended = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            key_positions = torch.arange(k.shape[-2], device=x.device)
            query_positions = past_length + torch.arange(length, device=x.device)
            causal_mask = key_positions[None, :] <= query_positions[:, None]
            attended = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=causal_mask,
                dropout_p=self.attn_dropout if self.training else 0.0,
                is_causal=False,
            )

        attended = attended.transpose(1, 2).contiguous().view(batch, length, channels)
        output = self.resid_dropout(self.proj(attended))
        if use_cache:
            return output, (k, v)
        return output


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        norm_cls = RMSNorm if getattr(config, "use_rmsnorm", False) else nn.LayerNorm
        bias = getattr(config, "bias", True)
        self.ln_1 = norm_cls(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = norm_cls(config.n_embd)
        if getattr(config, "use_swiglu", False):
            self.mlp = SwiGLUFeedForward(config)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(config.n_embd, 4 * config.n_embd, bias=bias),
                nn.GELU(),
                nn.Linear(4 * config.n_embd, config.n_embd, bias=bias),
                nn.Dropout(config.dropout),
            )

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ):
        attention_result = self.attn(self.ln_1(x), past_key_value, use_cache)
        if use_cache:
            attention_output, present_key_value = attention_result
            x = x + attention_output
            x = x + self.mlp(self.ln_2(x))
            return x, present_key_value
        x = x + attention_result
        return x + self.mlp(self.ln_2(x))


class SyntraTransformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        norm_cls = RMSNorm if getattr(config, "use_rmsnorm", False) else nn.LayerNorm
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.seq_len, config.n_embd)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.final_norm = norm_cls(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._initialize)

        residual_std = 0.02 / math.sqrt(2 * config.n_layer)
        for name, parameter in self.named_parameters():
            if name.endswith("attn.proj.weight") or name.endswith("mlp.2.weight") or name.endswith("mlp.w_down.weight"):
                nn.init.normal_(parameter, mean=0.0, std=residual_std)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def _forward_hidden(
        self,
        tokens: torch.Tensor,
        past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
        use_cache: bool = False,
    ):
        if tokens.ndim != 2:
            raise ValueError("Input tokens must have shape [batch, sequence].")
        if tokens.shape[1] < 1:
            raise ValueError("Input must contain at least one token.")
        if past_key_values is not None and len(past_key_values) != len(self.blocks):
            raise ValueError("The cache must contain one key/value pair per Transformer block.")

        past_length = 0
        if past_key_values:
            past_length = past_key_values[0][0].shape[-2]
            if any(pair[0].shape[-2] != past_length or pair[1].shape[-2] != past_length for pair in past_key_values):
                raise ValueError("Cached attention layers have inconsistent sequence lengths.")
        if past_length + tokens.shape[1] > self.config.seq_len:
            raise ValueError("Input plus cached tokens exceeds configured context length.")

        positions = torch.arange(past_length, past_length + tokens.shape[1], device=tokens.device)
        if getattr(self.config, "use_rope", False):
            x = self.token_embedding(tokens)
        else:
            x = self.token_embedding(tokens) + self.position_embedding(positions)[None, :, :]
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
        tokens: torch.Tensor,
        targets: torch.Tensor | None = None,
        past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
        use_cache: bool = False,
    ):
        if targets is not None:
            if not isinstance(targets, torch.Tensor) or targets.shape != tokens.shape:
                raise ValueError("targets must be a tensor with the same shape as input tokens.")
            if targets.dtype != torch.long:
                raise ValueError("targets must use torch.long token IDs.")
            if targets.device != tokens.device:
                raise ValueError("targets and input tokens must be on the same device.")
            supervised = targets != -100
            if torch.any((targets < 0) & supervised) or torch.any(targets >= self.config.vocab_size):
                raise ValueError("targets contains a token outside the configured vocabulary.")
            if not torch.any(supervised):
                raise ValueError("targets must contain at least one supervised token (not -100).")
        if use_cache:
            hidden, presents = self._forward_hidden(tokens, past_key_values=past_key_values, use_cache=True)
        else:
            hidden = self._forward_hidden(tokens, past_key_values=past_key_values, use_cache=False)
        logits = self.lm_head(hidden)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )
        if use_cache:
            return logits, loss, presents
        return logits, loss

    @_preserve_module_modes
    @torch.no_grad()
    def embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Compute L2-normalized semantic embeddings from pooled final hidden states."""
        self.eval()
        clipped = tokens[:, -self.config.seq_len:]
        hidden = self._forward_hidden(clipped, use_cache=False)
        mean_pooled = hidden.mean(dim=1)
        last_token = hidden[:, -1, :]
        combined = 0.7 * mean_pooled + 0.3 * last_token
        return F.normalize(combined.float(), p=2, dim=-1)

    @_preserve_module_modes
    @torch.no_grad()
    def generate(
        self,
        tokens: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        if not isinstance(tokens, torch.Tensor) or tokens.dtype != torch.long:
            raise ValueError("tokens must be a torch.long tensor with shape [batch, sequence].")
        if tokens.ndim != 2 or tokens.shape[0] < 1 or tokens.shape[1] < 1:
            raise ValueError("tokens must contain a non-empty batch and a non-empty sequence.")
        if tokens.device != self.token_embedding.weight.device:
            raise ValueError("tokens must be on the same device as the model parameters.")
        if torch.any((tokens < 0) | (tokens >= self.config.vocab_size)):
            raise ValueError("tokens contains an ID outside the configured vocabulary.")
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(float(temperature)) or not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be a finite number between 0.0 and 2.0.")
        if type(max_new_tokens) is not int or max_new_tokens < 1 or max_new_tokens > 4096:
            raise ValueError("max_new_tokens must be an integer between 1 and 4096.")
        if type(top_k) is not int or top_k < 1 or top_k > self.config.vocab_size:
            raise ValueError(f"top_k must be an integer between 1 and {self.config.vocab_size}.")
        if isinstance(top_p, bool) or not isinstance(top_p, (int, float)) or not math.isfinite(float(top_p)) or not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be a finite number between 0.0 (exclusive) and 1.0.")
        if isinstance(repetition_penalty, bool) or not isinstance(repetition_penalty, (int, float)) or not math.isfinite(float(repetition_penalty)) or not 1.0 <= repetition_penalty <= 2.0:
            raise ValueError("repetition_penalty must be a finite number between 1.0 and 2.0.")
        self.eval()
        output = tokens
        prompt_len = tokens.shape[1]
        context = output[:, -self.config.seq_len:]
        logits, _, past_key_values = self(context, use_cache=True)
        finished = torch.zeros(output.shape[0], dtype=torch.bool, device=output.device)
        for step in range(max_new_tokens):
            next_logits = logits[:, -1, :].float().clone()
            next_logits[:, BOS_ID] = float("-inf")

            if repetition_penalty > 1.0 and output.shape[1] > prompt_len:
                generated_slice = output[:, prompt_len:]
                for batch_idx in range(output.shape[0]):
                    unique_ids = torch.unique(generated_slice[batch_idx])
                    unique_ids = unique_ids[unique_ids != EOS_ID]
                    if unique_ids.numel() > 0:
                        selected = next_logits[batch_idx, unique_ids]
                        next_logits[batch_idx, unique_ids] = torch.where(
                            selected < 0,
                            selected * repetition_penalty,
                            selected / repetition_penalty,
                        )

            if temperature == 0.0:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                next_logits = next_logits / temperature
                if top_k < next_logits.shape[-1]:
                    threshold = torch.topk(next_logits, top_k, dim=-1).values[:, -1, None]
                    next_logits = next_logits.masked_fill(next_logits < threshold, float("-inf"))
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True, dim=-1)
                    cumulative_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                    remove_mask = cumulative_probs > top_p
                    remove_mask[:, 1:] = remove_mask[:, :-1].clone()
                    remove_mask[:, 0] = False
                    scatter_mask = torch.zeros_like(remove_mask).scatter(1, sorted_indices, remove_mask)
                    next_logits = next_logits.masked_fill(scatter_mask, float("-inf"))
                probabilities = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)

            # A finished row is held at EOS while the remaining batch continues.
            # This keeps the result rectangular without generating post-EOS content.
            next_token = torch.where(
                finished[:, None],
                torch.full_like(next_token, EOS_ID),
                next_token,
            )
            finished = finished | next_token[:, 0].eq(EOS_ID)
            output = torch.cat((output, next_token), dim=1)
            if bool(finished.all().item()) or step + 1 >= max_new_tokens:
                break

            cache_length = past_key_values[0][0].shape[-2]
            if cache_length < self.config.seq_len:
                logits, _, past_key_values = self(
                    next_token, past_key_values=past_key_values, use_cache=True
                )
            else:
                # Rebuild the rolling window at context saturation so learned absolute
                # positions remain identical to the original uncached generation path.
                context = output[:, -self.config.seq_len:]
                logits, _, past_key_values = self(context, use_cache=True)
        return output

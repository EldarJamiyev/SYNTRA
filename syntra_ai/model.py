"""A small decoder-only Transformer implemented for SYNTRA.
All parameters are initialized locally; no pretrained weights are fetched or loaded.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig

BOS_ID = 256
EOS_ID = 257


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=True)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=True)
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
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
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
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.seq_len, config.n_embd)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.final_norm = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor | None = None,
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
        logits = self.lm_head(self.final_norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        if use_cache:
            return logits, loss, tuple(presents)
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        tokens: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 40,
    ) -> torch.Tensor:
        if not 0.05 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0.05 and 2.0.")
        if max_new_tokens < 1 or max_new_tokens > 4096:
            raise ValueError("max_new_tokens must be between 1 and 4096.")
        if top_k < 1 or top_k > self.config.vocab_size:
            raise ValueError("top_k must be between 1 and 258.")
        self.eval()
        output = tokens
        context = output[:, -self.config.seq_len:]
        logits, _, past_key_values = self(context, use_cache=True)
        for step in range(max_new_tokens):
            next_logits = logits[:, -1, :] / temperature
            next_logits[:, BOS_ID] = float("-inf")
            if top_k < next_logits.shape[-1]:
                threshold = torch.topk(next_logits, top_k, dim=-1).values[:, -1, None]
                next_logits = next_logits.masked_fill(next_logits < threshold, float("-inf"))
            probabilities = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            output = torch.cat((output, next_token), dim=1)
            if int(next_token[0, 0].item()) == EOS_ID or step + 1 >= max_new_tokens:
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

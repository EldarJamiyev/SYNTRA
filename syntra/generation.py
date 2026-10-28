"""Autoregressive sampling for the SYNTRA model."""

from __future__ import annotations

import torch

from syntra.config import SyntraConfig
from syntra.model import SyntraModel
from syntra.tokenizer import EOS_ID, ByteTokenizer


@torch.no_grad()
def generate_tokens(
    model: SyntraModel,
    prompt_ids: list[int],
    config: SyntraConfig,
    *,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int = 40,
    device: str | torch.device = "cpu",
) -> list[int]:
    if not prompt_ids:
        prompt_ids = [1]
    if not 1 <= max_new_tokens <= 2048:
        raise ValueError("max_new_tokens must be between 1 and 2048.")
    if not 0.05 <= temperature <= 2.0:
        raise ValueError("temperature must be between 0.05 and 2.0.")
    if not 1 <= top_k <= config.vocab_size:
        raise ValueError("top_k must be between 1 and vocab_size.")

    model.eval()
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        context = ids[:, -config.block_size :]
        logits, _ = model(context)
        next_logits = logits[:, -1, :] / temperature
        cutoff = min(top_k, next_logits.shape[-1])
        threshold = torch.topk(next_logits, cutoff, dim=-1).values[:, -1, None]
        next_logits = next_logits.masked_fill(next_logits < threshold, float("-inf"))
        probabilities = torch.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probabilities, num_samples=1)
        ids = torch.cat((ids, next_id), dim=1)
        if int(next_id.item()) == EOS_ID:
            break
    return ids[0].tolist()


def generate_text(
    model: SyntraModel,
    prompt: str,
    config: SyntraConfig,
    *,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int = 40,
    device: str | torch.device = "cpu",
) -> str:
    tokenizer = ByteTokenizer()
    prompt_ids = tokenizer.encode(prompt, add_bos=True)
    output = generate_tokens(
        model,
        prompt_ids,
        config,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        device=device,
    )
    return tokenizer.decode(output)

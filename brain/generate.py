"""Temperature-controlled local inference with incremental token streaming."""
from __future__ import annotations

import argparse
from codecs import getincrementaldecoder
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Callable, Iterator
import torch
from torch.nn import functional as F

from .checkpoint import load_checkpoint
from .model import BOS_ID, EOS_ID, LocalTransformer, TransformerConfig, select_device
from .tokenizer import BytePairTokenizer

DEFAULT_CHECKPOINT = Path("checkpoints") / "syntra-brain.pt"
TokenCallback = Callable[[str, int, float], None]


@dataclass(frozen=True)
class GenerationResult:
    text: str
    tokens_generated: int
    tokens_per_second: float
    finish_reason: str


def load_for_inference(
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    device_name: str = "auto",
) -> tuple[LocalTransformer, BytePairTokenizer, torch.device, dict]:
    saved = load_checkpoint(checkpoint_path)
    config = TransformerConfig.from_dict(saved["config"])
    tokenizer = BytePairTokenizer.from_dict(saved["tokenizer"])
    device = select_device(device_name)
    model = LocalTransformer(config).to(device)
    model.load_state_dict(saved["model_state_dict"], strict=True)
    model.eval()
    return model, tokenizer, device, saved


def _token_bytes(tokenizer: BytePairTokenizer) -> list[bytes]:
    table = [bytes((token_id,)) for token_id in range(256)]
    for left, right in tokenizer.merges:
        if left >= len(table) or right >= len(table):
            raise ValueError("Tokenizer merge table contains an invalid dependency.")
        table.append(table[left] + table[right])
    return table


def _sample_next(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int | None,
    top_p: float,
    repetition_penalty: float,
    generated: list[int],
    generator: torch.Generator | None,
) -> torch.Tensor:
    scores = logits.float().clone()
    scores[:, BOS_ID] = float("-inf")

    if repetition_penalty != 1.0 and generated:
        previous = torch.tensor(sorted(set(generated)), device=scores.device, dtype=torch.long)
        previous = previous[previous != EOS_ID]
        if previous.numel():
            previous_scores = scores[:, previous]
            adjusted = torch.where(
                previous_scores < 0,
                previous_scores * repetition_penalty,
                previous_scores / repetition_penalty,
            )
            scores[:, previous] = adjusted

    if temperature == 0:
        return torch.argmax(scores, dim=-1, keepdim=True)

    scores = scores / temperature
    if top_k is not None and top_k < scores.shape[-1]:
        threshold = torch.topk(scores, top_k, dim=-1).values[:, -1, None]
        scores = scores.masked_fill(scores < threshold, float("-inf"))

    if top_p < 1.0:
        sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)
        cumulative = F.softmax(sorted_scores, dim=-1).cumsum(dim=-1)
        remove = cumulative > top_p
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        remove_mask = torch.zeros_like(remove).scatter(1, sorted_indices, remove)
        scores = scores.masked_fill(remove_mask, float("-inf"))

    probabilities = F.softmax(scores, dim=-1)
    if not torch.isfinite(probabilities).all().item() or (probabilities.sum(dim=-1) <= 0).any().item():
        raise RuntimeError("Sampling produced invalid probabilities.")
    return torch.multinomial(probabilities, num_samples=1, generator=generator)


def _token_events(
    model: LocalTransformer,
    tokenizer: BytePairTokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    top_p: float,
    repetition_penalty: float,
    seed: int | None,
    stop_event,
) -> Iterator[tuple[str, int, float, str | None]]:
    if not isinstance(prompt, str):
        raise TypeError("Prompt must be a string.")
    if not prompt.strip():
        raise ValueError("Prompt must contain at least one non-whitespace character.")
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 8192:
        raise ValueError("max_new_tokens must be between 1 and 8,192.")
    if not isinstance(temperature, (int, float)) or not 0 <= temperature <= 5:
        raise ValueError("temperature must be between 0 (greedy) and 5.")
    if top_k is not None and (type(top_k) is not int or not 1 <= top_k <= tokenizer.vocab_size):
        raise ValueError(f"top_k must be between 1 and {tokenizer.vocab_size}, or None.")
    if not isinstance(top_p, (int, float)) or not 0 < top_p <= 1:
        raise ValueError("top_p must be greater than 0 and at most 1.")
    if not isinstance(repetition_penalty, (int, float)) or not 1 <= repetition_penalty <= 2:
        raise ValueError("repetition_penalty must be between 1 and 2.")
    if seed is not None and (type(seed) is not int or not 0 <= seed <= 2**32 - 1):
        raise ValueError("seed must be between 0 and 4,294,967,295.")

    parameter = next(model.parameters())
    device = parameter.device
    config = model.config
    encoded_prompt = tokenizer.encode(prompt)
    context_ids = [BOS_ID, *encoded_prompt]
    prompt_truncated = len(context_ids) > config.block_size
    context_ids = context_ids[-config.block_size:]
    context = torch.tensor([context_ids], device=device, dtype=torch.long)

    generator: torch.Generator | None = None
    if seed is not None:
        if device.type in {"cpu", "cuda"}:
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)
        else:
            torch.manual_seed(seed)

    token_bytes = _token_bytes(tokenizer)
    decoder = getincrementaldecoder("utf-8")("replace")
    generated: list[int] = []
    started = perf_counter()
    reason = "length"

    model.eval()
    with torch.inference_mode():
        logits, _, cache = model(context, use_cache=True)
        for _ in range(max_new_tokens):
            if stop_event is not None and stop_event.is_set():
                reason = "cancelled"
                break
            next_token = _sample_next(
                logits[:, -1, :],
                temperature=float(temperature),
                top_k=top_k,
                top_p=float(top_p),
                repetition_penalty=float(repetition_penalty),
                generated=generated,
                generator=generator,
            )
            token_id = int(next_token.item())
            if token_id == EOS_ID:
                reason = "eos"
                break
            generated.append(token_id)
            if token_id >= len(token_bytes):
                raise RuntimeError("The model generated a token outside its saved tokenizer vocabulary.")
            text = decoder.decode(token_bytes[token_id], final=False)
            elapsed = max(perf_counter() - started, 1e-9)
            speed = len(generated) / elapsed
            yield text, len(generated), speed, None

            if len(generated) >= max_new_tokens:
                break

            cached_length = cache[0][0].shape[-2]
            if cached_length < config.block_size:
                logits, _, cache = model(
                    next_token,
                    past_key_values=cache,
                    use_cache=True,
                )
            else:
                # Rebuild only when the absolute-position context window is full.
                full_context = torch.cat(
                    (context, torch.tensor([generated], device=device, dtype=torch.long)),
                    dim=1,
                )[:, -config.block_size:]
                logits, _, cache = model(full_context, use_cache=True)

        final_text = decoder.decode(b"", final=True)
        if final_text:
            elapsed = max(perf_counter() - started, 1e-9)
            yield final_text, len(generated), len(generated) / elapsed, None
        elapsed = max(perf_counter() - started, 1e-9)
        yield "", len(generated), len(generated) / elapsed, reason


def stream_generate(
    model: LocalTransformer,
    tokenizer: BytePairTokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int | None = 40,
    top_p: float = 1.0,
    repetition_penalty: float = 1.05,
    seed: int | None = None,
    stop_event=None,
    on_token: TokenCallback | None = None,
) -> Iterator[str]:
    """Yield decoded text fragments as soon as generated UTF-8 bytes are complete."""
    for text, count, speed, finish_reason in _token_events(
        model,
        tokenizer,
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        seed=seed,
        stop_event=stop_event,
    ):
        if on_token:
            on_token(text, count, speed)
        if text:
            yield text


def generate(
    model: LocalTransformer,
    tokenizer: BytePairTokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int | None = 40,
    top_p: float = 1.0,
    repetition_penalty: float = 1.05,
    seed: int | None = None,
    stop_event=None,
    on_token: TokenCallback | None = None,
) -> GenerationResult:
    """Collect streaming output and return generated-token and throughput metrics."""
    pieces: list[str] = []
    token_count = 0
    speed = 0.0
    finish_reason = "length"

    def update(text: str, count: int, tokens_per_second: float) -> None:
        nonlocal token_count, speed
        token_count = count
        speed = tokens_per_second
        if on_token:
            on_token(text, count, tokens_per_second)

    for text, count, current_speed, reason in _token_events(
        model,
        tokenizer,
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        seed=seed,
        stop_event=stop_event,
    ):
        token_count = count
        speed = current_speed
        if text:
            pieces.append(text)
        if reason:
            finish_reason = reason
        if on_token:
            on_token(text, count, current_speed)

    return GenerationResult(
        text="".join(pieces),
        tokens_generated=token_count,
        tokens_per_second=round(speed, 2),
        finish_reason=finish_reason,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m brain.generate",
        description="Generate text from a locally trained SYNTRA checkpoint.",
    )
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        model, tokenizer, device, saved = load_for_inference(args.checkpoint, args.device)

        def progress(_text: str, count: int, speed: float) -> None:
            print(
                json.dumps({
                    "tokens_generated": count,
                    "tokens_per_second": round(speed, 2),
                    "device": str(device),
                    "checkpoint_step": saved["step"],
                }),
                file=sys.stderr,
                flush=True,
            )

        for fragment in stream_generate(
            model,
            tokenizer,
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            seed=args.seed,
            on_token=progress,
        ):
            print(fragment, end="", flush=True)
        print()
        return 0
    except (OSError, ValueError, RuntimeError, TypeError) as exc:
        print(f"SYNTRA generation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

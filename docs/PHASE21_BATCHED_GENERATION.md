# Phase 21 — Correct batched generation termination

## Scope

This change hardens the native PyTorch decoder's generation contract. It does not change model weights, tokenizer behavior, training data, checkpoint formats, inference providers, or the chat architecture.

## Defect

The previous generation loop stopped the entire batch when batch element zero emitted EOS. It also did not track completion for each row, so a row that emitted EOS while another row continued could receive additional generated tokens. The behavior was incorrect for batch inference and depended on batch ordering.

The generation entry point also relied on lower-level embedding and tensor operations to reject malformed input. Non-long token tensors, empty dimensions, invalid token IDs, and incompatible devices therefore failed late with less useful errors. Several sampling arguments accepted non-integral values that could fail deeper in the sampler.

## Implementation

- Track completion independently for every batch row.
- Freeze a completed row at EOS while remaining rows continue decoding.
- Emit EOS in already-completed rows while other rows continue. The returned tensor remains rectangular; EOS is the only padding token used by this generation contract.
- Stop the loop only when all rows have emitted EOS or the shared max_new_tokens budget is reached.
- Validate that input tokens are a non-empty rank-2 torch.long tensor, contain only IDs in the configured vocabulary, and reside on the same device as the model parameters.
- Validate integral generation limits and finite numeric sampling controls before model execution.
- Preserve the existing KV-cache path and rolling-context rebuild behavior.

## Regression coverage

The native model suite now includes:
- A two-row generation regression where row zero emits EOS immediately and row one must continue through two tokens before its own EOS.
- Assertions that the completed row is EOS-padded rather than generating additional content.
- Rejection tests for wrong token dtype, rank, empty batch/sequence, negative and out-of-vocabulary IDs, non-integral token budgets / top-k, and non-finite temperature.

These tests exercise the real Transformer forward/cache path while controlling only the next-token logits inside the test fixture to make row completion order deterministic.

## Boundaries

This is a decoding correctness improvement, not evidence of increased language understanding or training quality. It does not add beam search, variable-length tensor outputs, a PAD token, or batched chat streaming. EOS padding is internal to the returned rectangular batch; consumers should decode each row with EOS as a stopping token.

## Verification

The repository's native Python unit suite and CI workflow are the verification authority. This change must not be described as verified until those checks complete successfully on the exact PR head commit.

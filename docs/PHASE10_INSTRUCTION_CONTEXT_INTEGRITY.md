# Phase 10 — Lossless Instruction Context Integrity

## Defect addressed

The instruction JSONL loader previously truncated an overlong prompt to its newest `seq_len - 1` tokens. This could silently remove leading system constraints, task details, source context, or required output-format instructions before SFT. The behavior was especially risky because the training run could still complete and produce a checkpoint with no explicit indication that the conditioning prompt had been altered.

## Implementation

The loader now tokenizes each prompt with the selected checkpoint's exact BPE tokenizer and compares its length with the usable prompt budget (`seq_len - 1`, reserving one position for BOS).

- A prompt that fits is preserved in full.
- A prompt that exceeds the budget fails with its source line, measured token count, and available token budget.
- The loader does not silently clip the prompt, trim its beginning, or rewrite the prompt text.
- Long completion targets continue to use the existing overlapping causal windows. Previously supervised overlap positions remain masked so completion tokens and EOS are supervised once.

This is a fail-closed data-integrity change. It does not increase model context, alter the architecture, or introduce an external model/provider.

## Operator response

When the audit or fine-tuning command reports an overlong prompt, either shorten the prompt at its source while retaining all required instructions, or select a checkpoint with a larger context window and rerun the audit. Do not manually truncate the prompt tail or reuse a stale audit result after changing the dataset.

## Regression coverage

The new focused tests verify:
- Exact preservation of a prompt at the usable context boundary.
- Explicit rejection of a prompt one token beyond the available budget.
- Complete supervision of long completions across overlapping windows, including EOS.

The tests run as part of the existing Python test suite in GitHub Actions. No local test result is claimed until that workflow completes.

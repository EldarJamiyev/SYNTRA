# Native model: implementation and limits

SYNTRA's native runtime is a PyTorch decoder-only Transformer with learnable parameters, causal self-attention, residual/MLP blocks, cross-entropy objective, backpropagation, AdamW optimization, held-out validation, atomically written checkpoints, and autoregressive inference.

The initial weights are random. No pretrained checkpoints are bundled, downloaded, or loaded. The fixed tokenizer treats UTF-8 bytes as tokens and reserves token IDs 256 and 257 for BOS and EOS. A 1,024-token context is about 1,024 bytes, not 1,024 words. The default model is a small local training architecture rather than a competitive coding LLM.

The training data loader crawls only explicitly selected absolute directory roots. It skips common dependency/build/cache folders, symlinks, hidden directories, likely credential files, unsupported file suffixes, binary/non-UTF-8 content, and files larger than the per-file limit. It has a global token cap and an 80/20 held-out validation split. These heuristics are not a security classifier; inspect the directory and training files you choose.

The API has a narrow contract: `/model/status`, `/model/train`, `/model/train/status`, `/model/generate`, and `/model/finetune`. Long-running training occurs in a worker thread. Checkpoints carry model/optimizer state and measured train/validation loss; checkpoints are local artifacts and should never be committed.

## Supervised instruction fine-tuning

Prepare a UTF-8 JSONL dataset with one `{"prompt":"...","completion":"..."}` object per line. Only local files are read; no prompts/examples are sent to a hosted AI service. SYNTRA verifies required fields, uses the completion tokens as supervised targets, and masks prompt-token loss. Completion text is clipped to fit the small configured context window. Start from a trained checkpoint and use examples you own or are licensed to use. Measure performance on held-out examples before relying on it.

Example:
```jsonl
{"prompt":"Write a Python add function.","completion":"def add(a, b):\n    return a + b"}
{"prompt":"Write a TypeScript clamp helper.","completion":"export function clamp(n: number, low: number, high: number): number { return Math.min(high, Math.max(low, n)); }"}
```

A checkpoint proves only that optimization occurred. It does not prove instruction following or broad reasoning. The current local agent path invokes the trained model and puts actual generated continuation into the editor; it does not claim to create a correct patch automatically. Its raw output may be malformed or irrelevant. A future milestone should add syntax-aware patch extraction, multi-file planning/edit loops and iterative test-failure repair with explicit approval boundaries.

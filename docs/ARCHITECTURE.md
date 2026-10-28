# SYNTRA architecture

## Implemented native runtime

SYNTRA has a native Python/PyTorch runtime, independent of the retired browser byte-transition prototype. The model implementation, tokenizer, training loop, local checkpoint loader, inference API, CLI and project workspace are maintained in this repository. Initial neural weights are random; no pretrained checkpoint is bundled, downloaded or loaded. PyTorch provides tensor operations, automatic differentiation and optimized kernels.

## Data flow

1. The operator chooses an absolute local directory containing source and documentation files.
2. `syntra_ai.data` walks supported text formats and excludes common dependency/build folders, hidden directories, symlinks, likely secrets, non-UTF-8 files and oversized files. File and token caps bound the first laptop workflow.
3. UTF-8 source is represented as 256 byte values plus BOS and EOS, for a fixed 258-ID vocabulary.
4. `SyntraTransformer` uses learned token/position embeddings, causal multi-head self-attention, residual feed-forward blocks, LayerNorm and a tied token-output embedding.
5. Training minimizes next-byte cross-entropy with backpropagation, AdamW, gradient clipping and a sequential 80/20 train/validation split. Measured training/validation loss and model/optimizer state are saved atomically to a local checkpoint.
6. Optional supervised fine-tuning reads a local JSONL file with `prompt` and `completion` fields. Prompt targets are masked, so loss is applied to completion tokens rather than prompt bytes.
7. Generation loads the actual checkpoint and samples autoregressively from the neural model. It returns a raw byte continuation.
8. A loopback-only Python API runs on port 3002. A local Node workspace API on port 3001 bridges it to the React UI on port 3000.
9. The project workspace reads a user-selected codebase, stages a reviewable proposal, checks content hashes before proposing/applying, backs up files, records actions, and runs explicitly authorized build/test/lint/typecheck scripts.

## Actual limitations

- The tokenizer is deliberately simple, lossless byte encoding rather than BPE or another learned tokenizer. Context length is therefore measured in bytes: 1,024 tokens is about 1,024 bytes, not 1,024 words.
- The default randomly initialized model is approximately one million parameters and is a training-system foundation, not a competitive coding LLM. Quality depends on permitted training data, optimization, and task-based evaluation.
- The local generator performs raw continuation. It does not yet reliably follow natural-language instructions or produce complete multi-file patches. Instruction fine-tuning is supported but does not guarantee instruction following.
- The deterministic project plan is still a workflow template. A fully model-driven multi-file agent that uses tools, diagnoses test failures, repairs changes, and iterates has not been implemented.
- Falling loss only measures next-token prediction; it does not establish code correctness, security, or useful application-building capability.

## Security

The AI and workspace services bind to 127.0.0.1 and restrict browser origins. Workspace operations are bounded to an explicitly selected root. Likely credential files and symbolic links are blocked. Generated code is untrusted text: it is never automatically executed. Applying a proposed file change requires explicit confirmation, verifies the source hash, and creates a backup before replacing an existing file. Project test/build scripts may execute code and must be trusted before running.

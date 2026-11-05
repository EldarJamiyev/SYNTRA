# Model capabilities and evaluation

## Local inference

The supported one-click startup and application backend use only SYNTRA's native Python/PyTorch Transformer. There is no provider selector or alternate inference integration. The native runtime requires Python 3.12 or newer and a trained local checkpoint; it does not download pretrained weights or call a hosted AI API.

The native model is a small, randomly initialized decoder-only research model trained from local source text. It is useful for experimentation with tokenization, training, inference, and checkpoint mechanics. It is not a pretrained coding model and should not be represented as comparable to a modern 7B/14B coding model or Claude Code.

Before fine-tuning, run `python -m syntra_ai.cli audit-instructions --dataset <file.jsonl>`. This read-only preflight checks JSONL structure, duplicate prompts, prompt diversity, and whether a deterministic prompt-group train/validation split is possible. It does not judge semantic correctness, create a curated dataset, or prove that examples teach useful programming behavior. The audit uses a checkpoint tokenizer only when one is provided; otherwise it reports that the byte-level fallback was used for inspection.

SYNTRA's native model is a small, randomly initialized research model trained from the approved local corpus; it is not a pretrained coding model and its training does not guarantee strong instruction following. Semantic source retrieval and project-memory embeddings use this same native checkpoint. Project memory falls back to BM25 keyword ranking when native embeddings are unavailable or invalid. Source and memory embedding quality depends on the locally trained checkpoint and corpus; it is not pretrained semantic knowledge. Memory retrieval never embeds chat history.

Native prompt token counts use SYNTRA's local tokenizer. Any byte-ratio estimate exposed by compatibility or diagnostic paths must remain clearly marked as an estimate and must not be presented as exact model tokenization.

## Self-review

Patch review runs before verification and uses the same native checkpoint as generation. It is a second reasoning pass, not an independent-model guarantee. If review is enabled and no accepted verdict is produced, the candidate is blocked rather than silently treated as reviewed.

## Benchmark evidence

The benchmark command evaluates a local agent against versioned fixtures and stores machine-readable reports. The eight-task suite includes simple JavaScript utilities, a TypeScript compiler-checked contract task, a multi-file API change, and a Python task; it remains a development baseline rather than a broad coding benchmark. A passing SYNTRA unit-test suite validates SYNTRA's implementation and test harness; it does not establish coding ability of a live model. Run `npm run benchmark` on the target hardware with the actual installed model and preserve the JSON report to compare model, prompt, and agent changes. Benchmark process memory measurements do not include all system-level runtime allocations or GPU VRAM.

The current benchmark fixtures are a development baseline, not a representative coding benchmark. Broader coverage is still required for multi-file changes, repository debugging, dependency/build failures, API work, mixed-language workspaces, security fixes, and regression prevention.

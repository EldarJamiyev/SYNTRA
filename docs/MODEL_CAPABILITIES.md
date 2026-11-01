# Model capabilities and evaluation

## Local inference

SYNTRA can route inference to an installed local Ollama model or to its experimental native model service. Ollama model files remain local; the application does not need a hosted LLM API for this path.

The native model is a small, randomly initialized decoder-only research model trained from local source text. It is useful for experimentation with tokenization, training, inference, and checkpoint mechanics. It is not a pretrained coding model and should not be represented as comparable to a modern 7B/14B coding model or Claude Code.

For practical coding work, answer quality comes primarily from the installed pretrained local model selected by SYNTRA. Project-memory retrieval remains keyword-based unless `SYNTRA_EMBED_MODEL` is set to an installed local embedding model; in that case Ollama generates local vectors for saved user-authored memory notes and SYNTRA combines cosine similarity with BM25 ranking. The memory search falls back to BM25 on embedding errors and never embeds source files or chat history. Model selection uses the installed artifact size as a conservative first-pass filter and reserves 45% of physical RAM; disk size is not a reliable measure of inference RAM. SYNTRA also asks Ollama for the model-reported context length and queries the running-model list for resident size and VRAM use. When architecture metadata is present, SYNTRA estimates KV-cache bytes per token from layer count, embedding width, attention heads, and KV heads, then reports an advisory host-memory working-set estimate alongside total/available RAM. This estimate is explicitly partial: quantization, GPU offload, runtime allocations, KV-cache placement, concurrent workloads, and memory mapping can change actual requirements. It is not a guarantee that a model will load successfully.

Token counts used for Ollama prompt budgeting are UTF-8 byte-ratio estimates with a safety margin, not the model's tokenizer output. Provider-reported prompt/evaluation counts are the authoritative measurements when Ollama returns them. The provider surfaces the estimate type so the UI/report does not present it as exact tokenization.

## Self-review

Patch review runs before verification. By default it uses the same local model as generation, so it is a second reasoning pass rather than an independent model guarantee. Set `SYNTRA_REVIEW_MODEL` to a different installed local Ollama model to request a separate reviewer. The selected model must pass SYNTRA's local installation and memory-selection checks. If review is enabled and no accepted verdict is produced, the candidate is blocked rather than silently treated as reviewed.

## Benchmark evidence

The benchmark command evaluates a local agent against versioned fixtures and stores machine-readable reports. The seven-task suite includes simple JavaScript utilities, a multi-file API change, and a Python task; it remains a development baseline rather than a broad coding benchmark. A passing SYNTRA unit-test suite validates SYNTRA's implementation and test harness; it does not establish coding ability of a live model. Run `npm run benchmark` on the target hardware with the actual installed model and preserve the JSON report to compare model, prompt, and agent changes. Benchmark process memory measurements do not include the separate Ollama daemon, GPU VRAM, or all system-level runtime allocations.

The current benchmark fixtures are a development baseline, not a representative coding benchmark. Broader coverage is still required for multi-file changes, repository debugging, dependency/build failures, API work, mixed-language workspaces, security fixes, and regression prevention.

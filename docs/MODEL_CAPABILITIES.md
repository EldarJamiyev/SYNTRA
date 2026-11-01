# Model capabilities and evaluation

## Local inference

SYNTRA can route inference to an installed local Ollama model or to its experimental native model service. Ollama model files remain local; the application does not need a hosted LLM API for this path.

The native model is a small, randomly initialized decoder-only research model trained from local source text. It is useful for experimentation with tokenization, training, inference, and checkpoint mechanics. It is not a pretrained coding model and should not be represented as comparable to a modern 7B/14B coding model or Claude Code.

For practical coding work, answer quality comes primarily from the installed pretrained local model selected by SYNTRA. Model selection uses the installed artifact size as a conservative first-pass filter and reserves memory headroom; disk size is not a reliable measure of inference RAM. SYNTRA also asks Ollama for model metadata and clamps the configured context to the model-reported context length when that metadata is available.

Token counts used for Ollama budgeting are estimates, not the model's tokenizer output. The estimate deliberately includes a safety margin. Provider-reported prompt/evaluation counts are the authoritative measurements when Ollama returns them.

## Self-review

Patch review runs before verification. By default it uses the same local model as generation, so it is a second reasoning pass rather than an independent model guarantee. Set `SYNTRA_REVIEW_MODEL` to a different installed local Ollama model to request a separate reviewer. The selected model must pass SYNTRA's local installation and memory-selection checks. If review is enabled and no accepted verdict is produced, the candidate is blocked rather than silently treated as reviewed.

## Benchmark evidence

The benchmark command evaluates a local agent against versioned fixtures and stores machine-readable reports. A passing unit-test suite validates SYNTRA's implementation and test harness; it does not establish coding ability of a live model. Run `npm run benchmark` on the target hardware with the actual installed model and preserve the JSON report to compare model, prompt, and agent changes. Benchmark process memory measurements do not include the separate Ollama daemon, GPU VRAM, or all system-level runtime allocations.

The current benchmark fixtures are a development baseline, not a representative coding benchmark. Broader coverage is still required for multi-file changes, repository debugging, dependency/build failures, API work, mixed-language workspaces, security fixes, and regression prevention.

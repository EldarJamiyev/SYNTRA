# Changelog

All notable user-facing changes will be documented here.

## Unreleased

### Added
- Expand workspace secret scanning patterns and add approved, time-limited, file-SHA-256-bound detector exceptions with explicit revoke commands and audit redaction.
- A real browser-based React UI screenshot capture in CI and a visual architecture overview.
- Record measured host/native-device identity in raw-model benchmark reports and add protocol-/denominator-validated report comparison that blocks performance claims across incomplete or different hardware.
- Add a per-language raw-model comparison matrix with exact valid-fixture denominators, separately measured quality/performance outcomes, and explicit small-sample limitations.
- Expand the held-out raw-model benchmark from 14 to 21 real language fixtures, with three task types each for JavaScript, TypeScript, Go, and Rust plus nine Python tasks.
- Contribution, security, issue-reporting, and pull-request guidance.

### Changed
- Serialize resident model construction per checkpoint path, share one in-flight load among concurrent inference requests, and reject cache publication when checkpoint identity changes during loading.
- Enforce strict native embedding/evaluation device and integer-budget validation before checkpoint access; stop HTTP coercion for embed, evaluate, and curriculum request parameters.
- Share strict native generation parameter/device validation between streaming and non-streaming inference, before checkpoint loading; remove HTTP numeric coercion and reject non-boolean stream flags.
- Apply strict JSON decoding to native HTTP request bodies, rejecting duplicate object keys, non-standard/non-finite numbers, float overflow, and excessive nesting before endpoint dispatch.
- Validate explicit CUDA device indices in the shared local device selector before training/model allocation; preserve default `cuda` selection and fail clearly for unavailable indices.
- Validate native source-training and supervised fine-tuning request types/ranges, model configuration keys, and local device availability before queueing background work; reject missing-checkpoint source resumes instead of silently starting from random weights.
- Parse supervised-instruction JSONL strictly: reject duplicate object keys at any depth, non-standard/non-finite numeric values, float overflow, and nesting beyond the 128-level record limit with source-line validation errors.
- Extend checkpoint-backed status recovery to source pretraining, validate its optimizer/RNG/corpus provenance, and safely upgrade legacy progress metadata only after verifying the full checkpoint.
- Reconcile stale SFT JSON progress from the authoritative full checkpoint after interrupted writes, with checkpoint-identity verification and explicit stale-status reporting.
- Add dataset-bound supervised fine-tuning resume: restore AdamW, stage scheduler, and CPU/CUDA RNG state from an atomic per-update checkpoint, with strict dataset/settings/device validation and explicit CLI/API resume controls.
- Use a persisted global-step warmup/cosine schedule in the application’s native source-pretraining path, retain the legacy linear-warmup trajectory for older checkpoints, and reject non-finite pre-clipping gradient norms.
- Make native source-pretraining and fresh corpus evaluation document-isolated: deduplicate exact source copies, learn BPE merges from training documents only, preserve complete files under the token budget, and record holdout provenance.
- Enforce the Node.js version range required by the current Vite runtime in package metadata, startup validation, and CI.
- Clarify that no model-performance score or pretrained starter checkpoint is published until genuine artifacts and reports exist.

### Removed
- Remove the optional external inference bridge and provider-selection paths. Chat, coding, retrieval, embeddings, training, and benchmarks now use only SYNTRA's native local PyTorch runtime; unavailable checkpoints fail explicitly.

### Not included
- No pretrained SYNTRA checkpoint or model release asset is published by this change.
- No benchmark performance numbers are fabricated or inferred from software test results.

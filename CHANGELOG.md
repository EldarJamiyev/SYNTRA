# Changelog

All notable user-facing changes will be documented here.

## Unreleased

### Added
- Expand workspace secret scanning patterns and add approved, time-limited, file-SHA-256-bound detector exceptions with explicit revoke commands and audit redaction.
- A real browser-based React UI screenshot capture in CI and a visual architecture overview.
- Contribution, security, issue-reporting, and pull-request guidance.

### Changed
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

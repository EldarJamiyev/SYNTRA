# Phase 11 — SFT Dataset Provenance

## Problem

Instruction fine-tuning already records dataset quality statistics and training history, but the checkpoint did not identify the exact JSONL bytes used for SFT. A later operator could inspect a checkpoint and its quality report without proving which immutable dataset revision produced it. A file path alone would not solve this: the file can be edited or replaced after training.

## Implementation

The instruction loader computes SHA-256 and byte length from the exact byte sequence read through its existing approved-root, no-follow, regular-file and before/after identity checks. Every generated context window carries this dataset identity. Source-record indexing and train/validation splitting require all windows to agree on one dataset identity.

The read-only instruction audit exposes:
- `dataset_sha256`: SHA-256 of the exact JSONL bytes read
- `dataset_bytes`: exact byte length

Fine-tuning progress reports, full checkpoints, inference companions, and sidecar metadata retain the same values. No dataset path or record content is copied into the checkpoint solely for provenance. Replacing or editing the JSONL file produces a different digest on the next audit/training load.

## Integrity boundaries

- The digest identifies the bytes actually read by the loader, not a mutable pathname.
- The existing safe-open checks remain authoritative; hashing does not weaken path approval or file-change detection.
- The digest proves dataset identity, not dataset correctness, licensing, privacy suitability, or model capability.
- Audit and fine-tuning must use the checkpoint tokenizer when token-count/context compatibility is being assessed. A byte-fallback audit remains explicitly labeled as such.

## Regression coverage

- Compare loader and audit fingerprints with SHA-256 over the exact JSONL bytes.
- Modify the dataset and prove the fingerprint changes.
- Execute a real one-step CPU fine-tuning test and verify that the full checkpoint, inference companion, and returned training report retain the same digest and byte count.

No synthetic dataset, external model, hosted service, or checkpoint promotion is introduced.


## Strict JSONL record parsing

Instruction JSONL records are now parsed using strict JSON semantics rather than Python's permissive default decoder. Duplicate object member names are rejected at every nesting depth, including optional metadata; the loader no longer silently accepts a last-value-wins interpretation. Non-standard `NaN`/`Infinity` constants and numeric values that overflow to a non-finite float are rejected. Excessive JSON nesting is converted into a line-scoped dataset validation error instead of escaping as an unhandled parser recursion failure.

The record and file-size limits, required string `prompt`/`completion` fields, optional `split_group`/`source_path` metadata, exact-byte SHA-256 provenance, safe-open checks, and checkpoint-tokenizer context checks remain unchanged. This is input validation only; it does not rewrite datasets, discard records, alter split membership, or change the training objective.

Regression tests cover duplicate required keys, duplicate keys inside nested metadata, non-standard constants, floating-point overflow, deeply nested input, accepted valid metadata, and the read-only audit's invalid-input result.

# Phase 40 — Full-text native semantic embeddings

## Objective

Prevent native semantic indexing and project memory from embedding only the beginning of a long chunk or note when the checkpoint context is shorter than the supplied text.

## Implementation

- Tokenize every input with the tokenizer stored in the selected native checkpoint.
- Split the complete token sequence into overlapping windows. Each window contains at most the caller's requested `max_length` and the selected checkpoint's usable context (`seq_len - 1` tokens, leaving one position for BOS).
- Obtain each window vector from the same native PyTorch checkpoint. Combine window vectors with weights based on newly covered source tokens, then normalize the pooled vector.
- Keep request work bounded to 64 inputs, 100 KB per input, 131,072 aggregate input tokens, and 2,048 total windows. The token/window budget is computed before the first embedding forward pass.
- Return exact token counts, per-window token lengths, newly covered token counts, total windows, effective window length, pooling strategy, checkpoint identity, and `truncated: false`.
- Reject malformed or incomplete response evidence in the semantic-index and project-memory clients.
- Version semantic index and memory embedding caches so vectors built by the previous prefix-slicing behavior are not reused.

## Verification

The Python regression suite includes a real CPU/PyTorch checkpoint test for long-input coverage and an explicit proof that an over-budget window plan is rejected before `embed_tokens` runs. Node regressions cover response-evidence validation, semantic-index v1 rebuild, and project-memory cache schema migration.

These tests verify input coverage, bounds, and cache integrity. They do not establish that the native model's embeddings are semantically useful or that retrieval quality improved. A representative local checkpoint/retrieval evaluation remains required for that claim.

## Boundaries

- No model architecture, weights, tokenizer training algorithm, or training objective is changed.
- All embeddings continue to come from the selected local SYNTRA PyTorch checkpoint.
- No external embeddings, hosted APIs, model registries, or third-party inference providers are introduced.
- Existing user workspace operations remain read-only and bounded; tests use disposable temporary workspaces.

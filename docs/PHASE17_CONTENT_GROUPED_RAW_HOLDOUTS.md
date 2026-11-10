# Phase 17 — Content-Grouped Raw Dataset Holdouts

## Defect addressed

The raw-source holdout previously assigned files independently using a stable hash of each relative path. Two byte-identical files stored under different paths could therefore be placed on opposite sides of training and validation. Validation would then measure memorization of content already present in training rather than generalization to independent source content.

## Implementation

- Compute a SHA-256 identity from each decoded source body, excluding the synthetic path header used to label concatenated repository input.
- Group exact-content duplicates before deterministic holdout assignment.
- Rank content groups by a namespaced content digest, keeping the split reproducible without depending on path names.
- Keep every path belonging to a content group on the same side of the split.
- Reject a multi-file corpus containing fewer than two independent source contents when a genuine holdout is required.
- Preserve the existing single-file line-boundary fallback and the legacy non-holdout behavior.

This change intentionally detects exact duplicates only. It does not claim semantic or near-duplicate detection; structurally similar files can still share code across the split and require a future similarity-aware curation policy.

## Regression coverage

- Exact duplicate files under different paths are assigned together.
- A duplicate-only multi-file corpus cannot be presented as an independent train/validation split.
- Existing source-file isolation, deterministic split, structured-instruction split, tokenizer-fit, and training-resume tests remain in place.

## Boundaries

No training run, model-quality improvement, architecture change, checkpoint-format change, or inference-provider change is claimed. The change improves the validity of raw-corpus validation boundaries; measured model quality must still be established by an actual training run and evaluation against a fixed benchmark protocol.

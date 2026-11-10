# Phase 12 — SFT Dataset Lineage Through Evaluation and Promotion

## Objective

Phase 11 binds an instruction-fine-tuned checkpoint to the exact JSONL bytes used for SFT. Phase 12 extends that identity into the evidence used to evaluate and promote a candidate checkpoint. A benchmark result must not be detached from the training dataset provenance recorded beside the staged checkpoint.

## Implementation

- The raw-native-model benchmark reads the checkpoint sidecar's `instruction_dataset_sha256` and `instruction_dataset_bytes` fields and includes them in the report's `model` record.
- The benchmark evidence loader rejects malformed SHA-256 values, negative/non-integer byte counts, and incomplete digest/size pairs.
- For an SFT candidate, the promotion gate requires a regular checkpoint sidecar in the staged snapshot, verifies the sidecar's SHA-256 against the snapshot artifact manifest, validates its SFT training mode and dataset identity, and compares both provenance fields with the benchmark report.
- Promotion results expose the dataset digest and byte count so the activation record retains the same lineage.
- Pretraining and baseline candidates do not claim instruction-dataset provenance.

## Integrity boundaries

The checkpoint digest authenticates the model artifact. The snapshot manifest separately authenticates the sidecar bytes. The benchmark report records the dataset identity read from that sidecar. Promotion requires these records to agree. This is an integrity chain for accidental or inconsistent artifact substitution; it is not a digital signature against an attacker able to rewrite every local artifact and manifest.

The dataset digest identifies exact JSONL bytes, not dataset quality, licensing, privacy suitability, or model capability. A successful benchmark still proves only the held-out tasks and environment recorded in that report.

## Regression coverage

- Verify that benchmark checkpoint evidence returns exact SFT dataset identity from the sidecar.
- Reject malformed SHA-256, invalid byte counts, and incomplete provenance pairs.
- Verify promotion accepts matching report/sidecar identities.
- Reject report/sidecar digest mismatches and missing or unauthenticated SFT sidecars.
- Preserve existing full-suite, baseline-comparison, regression-prevention, checkpoint hash, and atomic promotion checks.

No external inference provider, hosted service, synthetic training data, or automatic checkpoint promotion is introduced.

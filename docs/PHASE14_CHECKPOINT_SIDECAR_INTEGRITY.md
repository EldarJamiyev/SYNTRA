# Phase 14 — Checkpoint Sidecar Integrity

## Objective

SYNTRA stores a full PyTorch checkpoint and a lightweight JSON sidecar. The checkpoint replacement and sidecar replacement are separate filesystem operations. If the process stops after replacing the checkpoint but before updating the sidecar, the old sidecar can remain beside a newer checkpoint. A lightweight metadata reader that trusts the JSON alone can then report stale training state as if it described the current artifact.

Checkpoint serialization can also fail with exceptions other than `OSError`. Restricting temporary-file cleanup to operating-system errors can leave incomplete temporary checkpoint files behind.

## Implementation

Each newly written sidecar now records:

- `checkpoint_sha256`: SHA-256 of the exact serialized checkpoint bytes
- `checkpoint_bytes`: exact checkpoint byte length
- `checkpoint`: resolved checkpoint path, retained as the artifact association

The digest is calculated from the completed checkpoint using bounded-memory streaming. The identity reader requires a non-empty regular file, rejects a final-component symlink, checks file identity and timestamps before and after reading, and verifies the observed byte count.

`read_checkpoint_metadata` recomputes the current checkpoint identity before returning a modern sidecar. It returns an empty object when the checkpoint is missing, changes during hashing, or no longer matches the sidecar digest, size, or associated path. Existing sidecars without the new identity fields remain readable as legacy, unverified metadata; they are not retroactively represented as verified.

Checkpoint serialization now removes its temporary file for any ordinary exception, not only `OSError`.

## Failure semantics

The checkpoint and sidecar remain two separate atomic replacements; this change does not claim a multi-file filesystem transaction. If checkpoint replacement succeeds and sidecar writing fails, the previous sidecar may remain on disk, but the metadata reader detects the mismatch and refuses to expose it as current metadata. The checkpoint itself remains independently loadable through the existing validated checkpoint loader.

The digest is an integrity consistency check, not a digital signature or authenticity guarantee. An actor able to rewrite both checkpoint and sidecar can recompute the digest. Promotion continues to use its separate benchmark, snapshot-manifest, and artifact verification gates.

## Regression coverage

- Verify sidecar SHA-256 and byte length against the actual serialized checkpoint.
- Modify checkpoint bytes and prove the stale sidecar is rejected.
- Force a non-`OSError` serialization failure and verify temporary checkpoint cleanup.
- Force sidecar replacement failure after a checkpoint update and verify stale metadata is not returned.

No model architecture, inference provider, dataset, training behavior, or promotion policy is changed. No model-quality improvement is claimed.

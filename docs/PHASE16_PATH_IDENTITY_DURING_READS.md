# Phase 16 — Bind file reads to selected path identity

## Objective

Close a time-of-check/time-of-use gap in two local file-integrity paths. A stable open file descriptor proves which bytes were read, but it does not prove that the selected pathname still refers to that same file when the operation completes. Training provenance and checkpoint promotion must fail closed if the pathname is replaced during the read.

## Implementation

- Extend the bounded local dataset reader to inspect the pathname again after reading.
- Require the post-read pathname to remain a regular file with the same device/inode, size, modification time, and change time as the opened descriptor.
- Harden checkpoint-promotion SHA-256 reads with descriptor-based opening, O_NOFOLLOW where available, and a post-read pathname identity check.
- Preserve existing byte limits, digest calculations, promotion evidence rules, and offline operation. No training, model architecture, inference-provider, or checkpoint format behavior changes.

## Regression coverage

- Replace a selected dataset pathname after its descriptor has been opened and assert that the reader rejects the operation.
- Confirm unchanged regular dataset files remain readable.
- Replace a staged checkpoint pathname during promotion hashing and assert that the digest operation rejects the artifact.
- Run the existing dataset, training, checkpoint-promotion, and full repository CI suites.

## Security boundary

This detects pathname replacement and ordinary metadata mutation during the read. It does not protect against a privileged actor capable of modifying the underlying file while preserving or restoring all observable metadata, nor does it authenticate the content's origin. SHA-256 still establishes content identity only when compared with a trusted expected digest. No model-quality or training-improvement claim is made.

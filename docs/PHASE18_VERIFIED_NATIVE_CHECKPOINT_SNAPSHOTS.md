# Phase 18 — Verified Native Checkpoint Snapshots

## Defect addressed

The active `syntra_ai` inference and training-resume paths performed a resource-size check and then passed the mutable checkpoint pathname directly to `torch.load()`. A separate pre-load check does not bind the bytes later deserialized to the file that was inspected. A concurrent path replacement could therefore substitute a different file after validation. The same path-resolution behavior could also erase evidence that the final path component was a symbolic link.

## Implementation

- Add one bounded native checkpoint snapshot loader for the active PyTorch runtime.
- Require a non-empty regular file and reject final-component symlinks where the platform exposes no-follow file opening.
- Evaluate the existing 40%-of-available-RAM checkpoint budget against the initial no-follow file identity inside the snapshot reader, removing the separate pathname-size check from inference and resume.
- Open the selected file once, verify descriptor identity against the initial path identity, and copy bounded chunks into a temporary snapshot.
- Re-check descriptor metadata and the selected pathname after reading; reject replacement, mutation, truncation, or growth during the read.
- Deserialize only the verified snapshot with `map_location="cpu"` and `weights_only=True`.
- Route both native inference checkpoint loading and training resume through the shared loader.
- Preserve the existing model formats, memory preflight, checkpoint selection, optimizer/resume logic, and inference behavior.

## Regression coverage

- A real PyTorch checkpoint loads from the verified snapshot.
- Final-component symlinks are rejected where supported.
- A deterministic pathname replacement after open is rejected before deserialization.
- The hard read-time byte limit and available-RAM checkpoint budget are enforced inside the verified snapshot path.
- Integration tests confirm that both native inference and training resume call the shared snapshot loader.

## Boundaries

This protects the local file-read/deserialization boundary. It does not authenticate who created a checkpoint, prove tensor values are benign, or defend against a privileged process that can mutate a file while restoring all observable metadata. It does not improve model quality or claim a training/inference performance gain. CI results are the validation source of truth.

# Phase 15 — Safe checkpoint loading

## Objective

Make checkpoint loading consume one bounded, stable regular-file snapshot. The loader must not follow a final-component symbolic link, must not trust a path stat followed by an unrelated pathname-based deserialization, and must not allow a file to grow beyond the configured byte budget during the read.

## Implementation

- Validate `max_bytes` as a positive integer; reject booleans and other numeric lookalikes.
- Inspect the supplied path with `follow_symlinks=False` and require a non-empty regular file.
- Open with `O_NOFOLLOW` where the platform provides it, then compare the opened descriptor's device/inode with the pre-open stat.
- Read in bounded chunks and stop immediately if the configured byte limit is crossed.
- Compare descriptor identity, size, modification time, and change time before and after the read; also verify that the pathname still resolves to the same regular-file identity.
- Copy the exact verified bytes into an automatically cleaned temporary file and pass that snapshot—not the mutable original pathname—to `torch.load(weights_only=True)`.
- Preserve CPU map-location and the existing post-deserialization schema checks.

## Regression coverage

The checkpoint integrity test suite now covers:
- Successful loading from the verified snapshot.
- Rejection of final-component symlinks.
- Positive-integer byte-limit validation, including rejection of booleans.
- Enforcement of the byte budget before deserialization.

These tests run in the repository's native PyTorch CI job. No local test execution or model training is claimed by this change.

## Platform and security boundaries

On platforms without `O_NOFOLLOW`, the pre-open and post-open file identity checks still reject a path substitution that changes the opened file identity. As with any pathname-based filesystem API, a privileged actor able to mutate the same underlying file concurrently may cause an operation to fail; the loader detects metadata/size changes and only deserializes the bytes copied from the opened descriptor. `weights_only=True` remains enabled. This hardening does not make an untrusted checkpoint authentic or guarantee its tensors are benign.

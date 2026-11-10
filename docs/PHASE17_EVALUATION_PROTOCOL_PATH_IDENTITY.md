# Phase 17 — Bind evaluation hashes to path identity

The raw-model benchmark protocol fingerprint is part of the evidence used to compare and promote checkpoints. The file hasher already opened files without following final-component symlinks, bounded their size, and verified that the opened descriptor did not change while hashing. It did not, however, verify that the selected pathname still referred to that same file when hashing completed.

The hasher now performs a final non-following path inspection and requires a regular file whose device, inode, size, modification time, and change time still match the original opened file. A replacement, removal, symlink substitution, or observable metadata change causes provenance generation to fail instead of recording a digest against a pathname that now names different content.

Regression coverage includes deterministic replacement of the selected path immediately after the second descriptor inspection, plus a stable-file control proving ordinary hashing remains supported. The change does not alter benchmark scoring, task selection, model behavior, or the protocol manifest schema.

This is a pathname-identity check, not cryptographic authentication of repository origin and not protection against privileged concurrent mutation capable of preserving or restoring observable metadata.

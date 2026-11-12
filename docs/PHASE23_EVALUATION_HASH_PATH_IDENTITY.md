# Phase 23 — Bind benchmark hashes to canonical path identity

## Defect

The raw-model benchmark protocol hasher verified that the opened descriptor remained stable while hashing. It did not verify that the selected pathname still referred to the same file after hashing. A replacement could therefore leave a valid digest associated with a path that now identified different bytes.

A final-component path check alone is also insufficient. A parent directory can be replaced with a symlink to another location containing a hard link to the same inode. Descriptor identity and even final-component inode equality would then remain unchanged while the pathname escapes the approved repository tree.

## Implementation

- Capture the selected file's canonical path before opening.
- Verify that the canonical path is inside the approved repository root.
- Preserve bounded regular-file checks, O_NOFOLLOW where available, and descriptor identity/metadata checks before and after reading.
- Reinspect the final path without following the last component after the read.
- Resolve the final canonical path and require it to equal the original canonical path and remain inside the approved repository root.
- Reject replacement, symlink substitution, parent-directory redirection, removal, mutation, and root escape instead of recording a misleading digest.
- Pass the canonical repository root into every protocol-file hashing call.

## Regression coverage

- Replacing a file after the descriptor's post-read inspection is rejected.
- Swapping a parent directory for a symlink to an outside directory that contains a hard link to the same inode is rejected.
- Stable regular-file hashing still returns the exact SHA-256 and byte count.
- Existing protocol-content sensitivity and symlink rejection tests remain intact.

The parent-directory test proves that same-inode equality is not treated as sufficient evidence of approved-path containment.

## CodeQL evidence retention

The CodeQL workflow now archives the exact SARIF output for each language as a short-lived workflow artifact. This provides inspectable evidence for code-scanning triage without depending on the connector's unavailable code-scanning-alert listing endpoint. It does not automatically dismiss or suppress findings.

## Boundaries

This is an evidence-integrity fix, not authentication of repository origin or a filesystem-wide transaction. It cannot prevent a privileged actor from mutating paths after the final inspection or defeating underlying filesystem identity guarantees.

## Verification

Native provenance tests, CodeQL, the full Python/Node regression suites, and the browser workflow are required before considering the change complete.

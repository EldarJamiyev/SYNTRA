# Phase 20 — Bind source-corpus bytes to final pathname identity

## Defect

The native source-corpus reader opened a file without following the final symlink component, verified the opened descriptor against the initial file inspection, and checked that the descriptor's identity and metadata remained stable during the read. Those checks proved that the opened file was stable; they did not prove that the selected pathname still named that file after the read.

A concurrent rename or replacement could therefore leave the descriptor reading the original bytes while the selected path already named a different file. The corpus could then retain a digest entry keyed by the selected path even though that path no longer identified the bytes being added to the corpus. This weakens the meaning of source-corpus provenance used by training checkpoints.

## Change

After the bounded descriptor read, `syntra_ai.data._read_safe_file` performs a non-following inspection of the selected path. The reader accepts content only when:

- Re-canonicalizing the selected path after the read produces the same canonical path as the one approved before opening; a parent-directory symlink or directory swap cannot redirect the final path outside the approved corpus root.

- The opened descriptor is still a regular file.
- The path itself still resolves to a regular file rather than a symlink or another file type.
- Device and inode identity match across the initial inspection, opened descriptor, final descriptor inspection, and final path inspection.
- Size, modification time, and change time remain stable across the descriptor and final path checks.
- The byte limit and UTF-8 decoding requirements remain satisfied.

Any observed replacement, removal, symlink substitution, metadata change, or read failure excludes that file from the corpus. Stable files retain the existing behavior. The corpus digest is updated only after the reader accepts the file, so rejected path races cannot enter the selected corpus provenance.

## Regression coverage

`tests/python/test_corpus_path_identity.py` exercises the real `load_corpus` integration:

- A deterministic replacement immediately after the descriptor's post-read identity check is excluded.
- A deterministic final-component symlink substitution is excluded.
- A parent-directory swap that redirects the same inode through an in-root symlink alias to an outside directory is rejected by the post-read canonical-path check.
- An unchanged regular source remains included.
- The source digest is independently reconstructed from the exact retained path and bytes, proving that rejected files do not contribute to the reported digest.

The race is injected by replacing the pathname between the post-read descriptor check and the final non-following path inspection. No sleeps, timing assumptions, fabricated corpus contents, or external services are used.

## Scope and limitations

This change binds the accepted corpus bytes to the selected pathname across the reader's final identity check. It does not provide a filesystem-wide transaction or protect against a privileged actor capable of orchestrating changes after the final inspection. Checkpoint format, tokenizer behavior, corpus selection policy, training objectives, and model architecture are unchanged.

# Phase 24 — Canonical filesystem boundaries and safe error responses

## Confirmed CodeQL findings

The full-branch CodeQL scan on `main` reported 18 Python `py/path-injection` findings across `data.py`, `safe_paths.py`, `curriculum.py`, and `instruction.py`, plus one JavaScript `js/stack-trace-exposure` finding in the chat-cancellation integration-test server.

## Path-handling changes

- Canonical corpus paths are normalized into the exact variables checked against approved-root boundaries and passed to `stat`, `open`, and traversal sinks.
- Corpus root, traversal base, child directory, and candidate-file paths are checked at their local filesystem boundaries.
- Corpus and fine-tuning dataset resolvers now use direct prefix guards on the canonical path variable before directory/file inspection.
- Curriculum output parent directories and final destinations are checked on the exact canonical values used for directory creation, temporary-file placement, and atomic replacement.
- Instruction JSONL reads now verify the opened descriptor matches the pathname before reading and revalidate final canonical path, inode, size, and timestamps after reading.

## Regression coverage

- A regular instruction dataset replaced after opening is rejected.
- Redirecting the selected dataset's parent directory to an outside symlink—even where that outside path is a hard link to the same inode—is rejected.
- A stable JSONL file inside an explicitly approved root remains readable.
- Existing corpus path-identity, symlink, traversal, and output-boundary tests remain required.

## Error disclosure

The chat-cancellation mock server now returns a fixed generic error response rather than sending exception messages or stack details back through the test HTTP response.

## Verification

Require the full native Python, frontend/Node, Windows and browser suites, CodeQL Python and JavaScript/TypeScript analyses, and dependency audit on the exact PR head. Inspect the new SARIF output after the scan; do not dismiss alerts merely to reduce the dashboard count.

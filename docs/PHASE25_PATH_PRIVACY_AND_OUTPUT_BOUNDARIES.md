# Phase 25 — Path privacy and output boundary ownership

## Findings addressed

The post-merge full-branch CodeQL SARIF identified two remaining path-injection flows in the curriculum-output resolver and two clear-text data flows caused by approved external corpus paths being persisted in evaluation metadata and printed by the curriculum CLI.

## Output path lifecycle

- `resolve_curriculum_output_path` now validates and returns a canonical approved destination without creating directories or probing the destination parent as a filesystem sink.
- `_write_jsonl_atomically` owns creation of missing parent directories only after its own canonical approved-root check, then re-resolves and revalidates the created parent before writing and publishing.
- The resolver keeps the existing workspace/project boundary, filename restrictions, and symlink rejection.

## Path privacy

- Evaluation run manifests no longer store absolute corpus, checkpoint, profile, workspace, or curriculum output paths.
- The selected corpus root is represented by an opaque approved-root label and a SHA-256 fingerprint of its canonical path.
- Nested profile and curriculum-result metadata are redacted before persistence so generated `dataset_path`, `workspace_roots`, and checkpoint fields cannot reintroduce absolute paths.
- Curriculum CLI output is an explicit allowlisted summary of numeric counts and validation metrics; raw curriculum result objects, absolute paths, root labels, and free-text path-bearing fields are never serialized.
- The `inspect-data` CLI sample list reports source filenames rather than absolute filesystem paths.

## Regression coverage

- Path resolver returns a valid nested destination without creating its parent.
- The real atomic JSONL writer creates that approved destination and writes the requested records.
- Run-manifest metadata redaction removes private root/path fragments from nested settings and curriculum results.
- Corpus-root fingerprint remains stable and has a fixed SHA-256 representation.
- CLI curriculum summary preserves bounded counts and validation metrics without emitting any filesystem path, root identifier, or raw result object.

## Boundaries

This change does not alter training data, model architecture, checkpoints, inference, dataset content, or the Python API's internal use of canonical paths. Full paths remain available to the actual local file operations; only persisted or printed metadata is redacted.

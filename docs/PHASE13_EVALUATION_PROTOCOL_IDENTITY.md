# Phase 13 — Content-Addressed Evaluation Protocol Identity

## Objective

A benchmark suite version and task-ID list do not uniquely identify the evaluation that produced a result. Fixture source, hidden tests, patch validation, or the harness can change without changing either label. Phase 13 binds each raw-model benchmark report to the exact checked-in inputs that determine its prompts and measured outcomes.

## Fingerprinted inputs

The protocol identity covers:

- The raw-model suite manifest
- The benchmark runner and language environment definitions
- The patch bridge and TypeScript test runner
- All checked-in fixture and hidden-evaluation files
- The local agent/patch runtime source tree, including its workspace and tool-runtime dependencies
- The locked Node dependency manifest

Each regular file is recorded by repository-relative path, exact byte length, and SHA-256. Entries are sorted and serialized deterministically before the aggregate SHA-256 is calculated. Symbolic links, non-regular files, missing required inputs, files over the per-file limit, and aggregate file/byte-limit violations fail closed.

The benchmark captures this identity before execution and recomputes it after all tasks finish. If any covered input changes during the run, no valid report is emitted. The report records the protocol schema version, aggregate digest, file count, and total bytes.

## Promotion gate

Checkpoint promotion recomputes the identity from the current checkout and requires an exact match with the benchmark report. Existing suite-version, complete-task-set, checkpoint-hash, snapshot-manifest, Docker-isolation, and SFT dataset-lineage checks remain in force.

A report from an older fixture or hidden-test revision is therefore rejected even if its suite version and task IDs are unchanged. Operators must rerun the benchmark against the current evaluation protocol before promoting a checkpoint.

## Integrity boundaries

The fingerprint establishes content identity and detects accidental or inconsistent changes between evaluation and promotion. It is not a digital signature and does not defend against an attacker who can rewrite the report and all relevant repository inputs. Container image digests continue to be recorded separately in the benchmark report's isolation evidence.

The fingerprint does not establish that the benchmark is statistically representative, that hidden tests are comprehensive, or that the model has broad coding ability. A successful result remains limited to the exact tasks, fixtures, tests, model checkpoint, and execution environments recorded in the report.

## Regression coverage

- Exact hidden-test changes alter the aggregate protocol digest.
- Exact fixture-source changes alter the aggregate protocol digest.
- Symbolic links inside evaluation inputs are rejected.
- Promotion rejects a report with a stale or altered protocol fingerprint.
- Existing benchmark, SFT lineage, regression-prevention, and checkpoint-promotion checks remain required.

No external inference provider, synthetic evaluation result, model training, or automatic checkpoint promotion is introduced.

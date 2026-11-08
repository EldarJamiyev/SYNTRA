# SYNTRA coding benchmarks

SYNTRA maintains two separate benchmark tracks. They answer different questions and their scores must not be combined.

## 1. Full coding-agent benchmark

The existing Node.js suite measures SYNTRA's repository-aware coding workflow: workspace discovery, read-only source retrieval, planning, structured patch generation, second-pass review, candidate verification, and proposal staging.

Run after starting the native service and preparing a trained checkpoint:

```bash
npm run benchmark -- --list
npm run benchmark
npm run benchmark -- --task utility-clamp
npm run benchmark:reports -- history
```

The versioned manifest is `benchmarks/suite.v1.json` (version 1.3.0) and currently contains 11 deterministic tasks:

| Task | Language / area | Capability |
| --- | --- | --- |
| `utility-clamp` | node-script | Implement a bounded numeric clamp helper |
| `string-slugify` | node-script | Normalize strings into URL-safe slugs |
| `config-deep-merge` | node-script | Merge nested configuration objects safely |
| `positive-integer` | node-script | Validate positive safe integers |
| `array-chunk` | node-script | Split arrays into fixed-size chunks |
| `http-header-lookup` | node-script | Implement case-insensitive safe HTTP header lookup |
| `async-retry` | node-script | Implement bounded asynchronous retry with selective retry policy |
| `multi-file-cache` | node-script | Implement and export a reusable in-memory cache across modules |
| `python-slugify` | python-unittest | Implement deterministic Python slug normalization |
| `go-clamp` | go-test | Implement integer clamp in a Go package |
| `rust-key-normalize` | rust-test | Implement normalized cache-key handling in Rust |

Each fixture must show its expected baseline failure before the agent is invoked. The runner requires a task-specific diagnostic; a generic non-zero exit code is insufficient. A task is complete only when the changed-path set exactly matches the manifest and candidate verification passes. Fixture work happens in disposable copies; the checked-in fixture and selected workspace remain unchanged.

## 2. Raw native-model benchmark

The direct-model track measures the native PyTorch checkpoint without calling the repository-aware agent. It does not include agent planning, repository retrieval, review, correction loops, or proposal orchestration. This separates model-generation capability from agent-system capability.

Run:

```bash
# List tasks without loading PyTorch or a checkpoint
npm run benchmark:model -- --list

# Evaluate one task
npm run benchmark:model -- --task python-clamp

# Run all raw-model tasks
npm run benchmark:model
```

The current suite is `tests/benchmark_engine/suite.v1.json`, version 1.0.0, and contains five Python fixtures:

| Task | Capability | Target behavior |
| --- | --- | --- |
| `python-slugify` | String normalization | Collapse non-ASCII/non-alphanumeric separator runs into one hyphen |
| `python-clamp` | Numeric boundaries | Clamp below/above range while preserving inclusive in-range values |
| `python-deep-merge` | Configuration | Recursively merge dictionaries without mutating inputs |
| `python-duration-format` | Time formatting | Format hours, minutes, and seconds without empty units |
| `python-array-chunk` | Data structures | Preserve order and split sequences at exact chunk boundaries |

For every task, the manifest names a fixture workspace, expected changed-path set, hidden target test, and independent hidden regression test. The fixture source is copied into a temporary model workspace. The hidden tests remain under `tests/benchmark_engine/hidden/`, are not copied into the model workspace, and are not included in the prompt. Before generation, the target test must fail with an assertion and the regression test must pass. Invalid baselines are reported as invalid fixtures and are not scored as model failures.

The native runtime's `tokenize_text` and `stream_generate` APIs are called directly against the selected checkpoint. Prompts exceeding that checkpoint's context window are rejected rather than silently truncated. Generation is deterministic (greedy decoding). The returned text must be one strict `SYNTRA_AGENT_PATCH_V1` JSON object without trailing prose. Candidate edits pass through SYNTRA's existing `validatePatch` and `writeCandidateChanges` functions, then a Python AST syntax check, target test, and regression test.

### Isolation and execution

Default execution is Docker-only and requires a pre-existing local `python:3.12-slim` image. The runner does not pull images. Each test container uses `--network none`, a read-only root, read-only workspace/evaluation mounts, a non-root user, dropped Linux capabilities, `no-new-privileges`, and bounded memory, CPU, PIDs, and temporary storage. Docker still trusts the Docker daemon and host kernel; it is not a virtual-machine boundary.

For a trusted local environment only, host execution can be explicitly enabled:

```bash
npm run benchmark:model -- --isolation host --allow-host-execution
```

Host execution is not sandboxed and runs tests with the current operating-system user's permissions. The command requires both flags so this risk cannot be selected accidentally.

### Direct-model measurements

The machine-readable output is written to:

```text
data/benchmarks/latest_run.json
```

The file is intentionally Git-ignored. It records:

- The checkpoint SHA-256, native checkpoint identity, training step, training mode, and selected device.
- Training and validation loss plus the bounded loss-history points actually saved in checkpoint metadata.
- Per-task baseline validity, strict patch validity, syntax status, target completion, and regression prevention.
- Prompt/output token counts, generation tokens per second, and separate generation, patch, syntax, target-test, regression-test, and total timings.
- Aggregate task-completion, patch-correctness, regression-prevention, and latency statistics.

The `OVERFITTING_LOSS_INTELLIGENCE_DIVERGENCE` warning triggers only when the saved training loss is at or below the configured threshold (default 1.0), at least three tasks were attempted, and zero target tasks completed. Configure the threshold with `--low-loss-threshold`. This is a warning heuristic, not a statistical diagnosis or proof that overfitting caused poor coding performance. Training-loss history is only recorded from checkpoints produced after history tracking was added; older checkpoints are reported as history-unavailable rather than backfilled.

## Comparability and limitations

Do not combine raw-model results with full-agent results. The direct track currently covers five small Python tasks; it is a first held-out regression suite, not a representative sample of unfamiliar repositories or an industrial capability benchmark. The existing eleven-task agent suite includes JavaScript, Python, Go, and Rust tasks, but those fixtures are still deliberately bounded and do not establish broad language competence.

For meaningful comparisons, retain the same suite version and task IDs, run the same checkpoint on the same hardware and device, and keep system load as similar as practical. Compare exact checkpoint identities and measured outputs. Do not infer skill improvement from training loss alone. Record a real baseline run before and after a training or agent change; the evaluator does not fabricate a score for missing checkpoints or failed tasks.

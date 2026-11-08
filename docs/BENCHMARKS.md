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

# Evaluate the TypeScript task
npm run benchmark:model -- --task typescript-object-mapper

# Run the complete 13-task matrix
npm run benchmark:model
```

The current suite is `tests/benchmark_engine/suite.v1.json`, version 1.3.0, and contains 13 held-out fixtures: nine Python, one JavaScript, one TypeScript, one Go, and one Rust task.

| Task | Language | Capability |
| --- | --- | --- |
| `python-slugify` | Python | Deterministic ASCII slug normalization |
| `python-clamp` | Python | Inclusive numeric clamping |
| `python-deep-merge` | Python | Recursive mapping merge without input mutation |
| `python-duration-format` | Python | Compact elapsed-time formatting |
| `python-array-chunk` | Python | Ordered fixed-size sequence chunks |
| `python-safe-path` | Python | Safe relative-path normalization |
| `python-header-lookup` | Python | Case-insensitive HTTP header lookup |
| `python-retry-backoff` | Python | Overflow-safe bounded exponential backoff |
| `javascript-event-normalizer` | JavaScript | Multi-file ESM event normalization, timestamp validation, and deep payload isolation |
| `typescript-object-mapper` | TypeScript | Multi-file generic key/value constraints and object mapping |
| `go-channel-sync` | Go | Worker-pool synchronization, cancellation, and race-safe collection |
| `rust-safe-slice` | Rust | Lifetime-bound UTF-8 byte slicing without panics or allocation |
| `python-multi-file-stream-state` | Python | Cross-module JSON state parsing and numeric stream clamping |

### Multi-file context and patch contract

The fixture workspace is copied into a temporary model-visible repository. The runner collects supported source/configuration files recursively, applies bounded file-count, per-file, total-byte, and prompt-byte limits, and presents them together as explicitly labeled files. Connected modules therefore appear in the same prompt; hidden evaluation code never does.

The model must return one strict `SYNTRA_AGENT_PATCH_V1` JSON object containing all required edits. Each path may appear only once, every path must be a safe relative path, and the production `validatePatch` / `writeCandidateChanges` implementation remains the authority for validating and applying the patch. The resulting changed-path set must exactly equal the manifest's expected paths. A multi-file task is not partially credited when one sibling file is omitted or an unexpected file is touched.

Before generation, the target test must fail with a recognized assertion/diagnostic and the independent regression test must pass. Invalid baselines are excluded from model-performance denominators. After patch application, the changed files receive language-specific validation and the separate hidden target/regression executions. Python uses AST parsing and Python tests; TypeScript uses the repository's locally installed TypeScript compiler plus runtime assertions; Go uses the Go compiler/test runner; Rust uses offline Cargo checking and Cargo integration tests.

### Language environments and offline execution

`tests/benchmark_engine/environments.py` defines the runtime matrix:

| Language | Default local image | Validation |
| --- | --- | --- |
| Python | `python:3.12-slim` | AST parsing and hidden Python tests |
| JavaScript | `node:20-slim` | `node --check` and hidden `.mjs` tests for the v1.3.0 multi-file event-normalizer task |
| TypeScript | `node:20-slim` | Local TypeScript compiler, then emitted JavaScript tests |
| Go | `golang:1.22-alpine` | `go test ./...` and hidden Go test programs |
| Rust | `rust:1.78-slim` | `cargo check --offline --locked` and Cargo integration tests |

The benchmark never pulls an image. Before inference, each selected local image tag is resolved to Docker's full immutable `sha256:<64 hex>` image ID. Evaluation containers run by that exact ID rather than by the mutable tag, so retagging after preflight cannot silently change the runtime artifact. The report retains both the requested tag and the resolved image ID; a missing or malformed ID fails preflight, and an image removed after preflight fails execution rather than falling back to another tag. Optional image-tag overrides are available as `--python-image`, `--node-image`, `--go-image`, and `--rust-image` (the legacy `--docker-image` flag remains an alias for the Python image). TypeScript evaluation mounts the repository-installed `node_modules/typescript` package read-only; it does not install a compiler or download packages in the container.

Each evaluation container uses `--pull=never`, `--network none`, a read-only root filesystem, read-only source/evaluation mounts, a non-root UID, dropped Linux capabilities, `no-new-privileges`, bounded memory/CPU/PIDs, and temporary writable build storage. Go proxy and checksum services are disabled. Cargo is explicitly offline, locked, and uses a disposable target directory. No language runner may download dependencies during evaluation. Cargo tasks with external dependencies will remain unsupported until those dependencies can be supplied and verified locally.

Docker still trusts the Docker daemon and host kernel; it is not a virtual-machine or hardware-isolation boundary. Do not expose an untrusted Docker daemon socket to the benchmark process.

For a trusted development environment only, host execution can be explicitly enabled:

```bash
npm run benchmark:model -- --isolation host --allow-host-execution
```

Host execution is not sandboxed and runs generated code with the current operating-system user's permissions. The command requires both flags so this risk cannot be selected accidentally. Host-mode unit tests validate fixture baselines; a live-model benchmark should use the default Docker isolation.

### Context-window accounting

The native runtime's `tokenize_text` and `stream_generate` APIs are called directly against the selected checkpoint. The runner records each prompt's exact native-tokenizer count, checkpoint context window, and context utilization. A prompt that exceeds the checkpoint context is rejected before generation; the report retains the measured token count, context size, utilization, and overflow-token count. A stream that reports truncation is invalid. The benchmark does not silently shrink, summarize, or drop sibling files to force a task into the window. This makes context pressure and explicit context rejections measurable, but the 13 fixtures are not a statistically designed context-saturation study; additional large-repository profiles are still needed before drawing conclusions about long-context capability.

### Direct-model measurements

The machine-readable output is written to:

```text
data/benchmarks/latest_run.json
```

The file is intentionally Git-ignored. It records:
- The checkpoint SHA-256, native checkpoint identity, training step, training mode, and selected device.
- Training and validation loss plus the bounded loss-history points actually saved in checkpoint metadata.
- Per-task language, image identity/tag, baseline validity, patch validity, compiler/syntax status, target completion, and regression prevention.
- Prompt/output tokens, context-window size/utilization, generation tokens per second, and separate generation, patch, syntax, target-test, regression-test, and total timings.
- Per-language and cross-suite target-completion, patch-correctness, regression-prevention, and fully verified counts/rates with their denominators; context measured/rejected/near-limit counts and utilization statistics; and latency aggregates.

The `OVERFITTING_LOSS_INTELLIGENCE_DIVERGENCE` warning triggers only when the saved training loss is at or below the configured threshold (default 1.0), at least three baseline-valid tasks completed model generation within the checkpoint context, and zero of those generation attempts passed their target tests. Context-rejected tasks remain in the overall end-to-end completion denominator but are excluded from this training/generalization heuristic. Configure the threshold with `--low-loss-threshold`. This is a warning heuristic, not a statistical diagnosis or proof that overfitting caused poor coding performance. Training-loss history is only recorded from checkpoints produced after history tracking was added; older checkpoints are reported as history-unavailable rather than backfilled.

## Comparability and limitations

Do not combine raw-model results with full-agent results. The direct raw-model suite now has 13 small held-out fixtures across five languages, but it is a controlled engineering regression matrix—not a representative sample of unfamiliar repositories or an industrial coding benchmark. JavaScript event normalization, Go concurrency, Rust memory safety, TypeScript type-system coverage, and multi-file Python behavior are each represented by one narrow task; that is not broad language competence.

GitHub CI installs the required Go/Rust toolchains to validate the fixture baselines and the runner implementation. Actual benchmark execution remains offline: it uses pre-provisioned local Docker images and does not pull images or packages. The benchmark does not fabricate a score when a checkpoint, local image, language runtime, or valid fixture baseline is unavailable.

For meaningful comparisons, retain the same suite version and task IDs, run the same checkpoint on the same hardware and device, and keep system load as similar as practical. Compare exact checkpoint identities and measured outputs. Do not infer skill improvement from training loss alone. Record a real baseline run before and after a training or agent change.

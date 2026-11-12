# SYNTRA coding benchmarks

SYNTRA maintains separate benchmark tracks. They answer different questions and their scores must not be combined. Each report records the actual engine and model identity. Native and hybrid reports are not interchangeable. No public performance score is claimed in this repository until a complete, real run report is committed or attached as a release artifact.

## 1. Full coding-agent benchmark

The existing Node.js suite measures SYNTRA's repository-aware coding workflow: workspace discovery, read-only source retrieval, planning, structured patch generation, second-pass review, candidate verification, and proposal staging.

Run after starting SYNTRA and verifying the selected native checkpoint or explicitly configured hybrid endpoint:

```bash
npm run benchmark -- --list
npm run benchmark
npm run benchmark -- --task utility-clamp
npm run benchmark:reports -- history
```

The versioned manifest is `benchmarks/suite.v1.json` (version 1.3.0) and currently contains 11 deterministic tasks. These are controlled regression fixtures, not a substitute for broad third-party coding benchmarks:

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

## Checkpoint training and evaluation orchestration

The repository-level Milestone 4 runner is scripts/evaluate_epochs.py, exposed as npm run brain:milestone4. It composes the existing standalone brain.train pretrainer, the repository curriculum generator, syntra_ai.instruction.audit_instruction_dataset, the checkpoint-aware masked SFT trainer, and this raw-model benchmark. It does not introduce another model provider or a second training implementation.

Run from the repository root:

```powershell
npm run brain:milestone4 -- --device auto
```

The local profile is configs/training/phase4_milestone4.json. Its initial experiment settings are:

| Setting | Initial value |
|---|---:|
| Pretraining learning rate | 3e-4 |
| Cosine floor | 1e-5 |
| Warmup | 100 optimizer steps |
| Cosine decay horizon | 5,000 optimizer steps |
| Gradient clipping | global norm 1.0, with non-finite norm rejection |
| Pretraining context / batch | 512 tokens / 2 |
| Pretraining steps | 5,000 additional steps |
| SFT learning rate / batch | 1e-4 / 2 |
| SFT steps | 1,000 |
| Evaluation interval | 500 saved optimizer steps |

These are conservative starting values for a 4-layer, 4-head, width-128 model, not empirically optimal settings. Device auto selects CUDA, then Apple MPS, then CPU when available. Select --device cpu, cuda, or mps explicitly to compare a supported local backend. Use actual throughput, memory and benchmark measurements when changing the profile.

The orchestration passes explicit exclusion roots for the repository-managed data/benchmarks, data/curricula, and data/checkpoints output directories to both corpus readers. Historical benchmark JSON, run summaries and earlier generated curriculum artifacts therefore cannot become source-pretraining or new curriculum examples. The raw trainer includes these exclusions in the resolved dataset contents/fingerprint, so exact-resume validation remains bound to the actual included files.
 
The source-pretraining checkpoint is staged at data/checkpoints/staging/syntra-brain.pt; the runner refuses to use the application's default production checkpoint as its write target. Existing staging checkpoints require either exact resume with matching dataset/settings or an explicit overwrite that first saves a backup. The SFT stage reads the synthesized JSONL with the checkpoint's actual tokenizer, verifies a prompt/source-group-separated validation split, and uses the existing -100 prompt-label masking path. Only response targets and EOS contribute to supervised loss.

At each configured saved-step boundary, the runner copies the full checkpoint, JSON metadata sidecar, and native inference companion when present. It verifies each artifact's SHA-256, preserves the source modification time for checkpoint-selection behavior, and makes the staged checkpoint artifacts read-only. It then launches npm run benchmark:model as a separate process with an explicit snapshot checkpoint and unique report path. One evaluation runs at a time to limit training/evaluation contention. Docker image preflight is completed for all benchmark languages before training starts; benchmark containers use the existing offline/no-network isolation and never pull images. Host mode is not sandboxed and requires --isolation host --allow-host-execution.

Each run writes a run manifest, immutable checkpoint snapshots, separate benchmark JSON files, evaluation logs, and summary.md / summary.csv under data/benchmarks/phase4-milestone4/<run-id>/. A benchmark result is accepted only if the native-model report identifies the exact staged checkpoint SHA-256. The summary correlates the saved training/validation losses with actual task-completion and regression-prevention rates. Missing reports, checkpoint mismatches, failed runs and absent metrics stay unavailable or failed; they are never converted into zeroes.

Useful options:

```powershell
# Fine-tune an existing staging checkpoint; pretraining is skipped
npm run brain:milestone4 -- --skip-pretraining

# Run source pretraining without benchmark subprocesses
npm run brain:milestone4 -- --skip-benchmark

# Override the step budgets or evaluation interval
npm run brain:milestone4 -- --max-pretrain-steps 1000 --sft-steps 250 --evaluation-every 250
```

The orchestration code does not itself install packages, download checkpoints, or call cloud inference services. Training and curriculum construction execute locally; Docker evaluation containers have networking disabled. This does not change the host operating system's global network configuration. Install the repository's locked dependencies and pre-provision the language images before starting a fully offline benchmark run.

A checked-in profile and passing tests are not model-performance measurements. Establish the first real checkpoint baseline on the target machine before evaluating whether any subsequent training change improves coding capability.

## Phase 5 local hardware telemetry and checkpoint promotion

The local orchestration runner captures training-side measurements at the same saved checkpoint boundaries used for raw native-model evaluation. The evaluation result JSON, Markdown trend report, and CSV table retain:

- Supervised target tokens per second, calculated over the stage wall clock including validation and checkpoint I/O (not a kernel-only throughput estimate).
- Last and mean optimizer-step latency in milliseconds, measured around the accumulated forward/backward/optimizer update.
- Process peak resident set size in bytes. This is the operating-system process-lifetime high-water; because pretraining and SFT run in one orchestration process, the SFT value includes an earlier pretraining peak and is not stage-isolated.
- Peak device memory in bytes where the backend exposes it, plus the metric source: CUDA allocator peak reset at the start of each stage, maximum sampled MPS driver/allocator memory while micro-batch activations are live, or CPU-not-applicable.
- Gradient accumulation factor and the trainer's estimated pre-allocation working-set size.

CPU RSS uses the operating system's process high-water metric. CUDA memory uses PyTorch's peak allocated allocator counter, reset before the training loop. MPS memory is a sampled high-water because the public PyTorch API does not expose an equivalent resettable peak allocator counter. A CPU memory-device value or unavailable platform reading stays blank/unavailable; it is not written as zero. The throughput basis is supervised target tokens, not raw corpus bytes.

### Hardware profiles

`configs/training/phase4_smoke.json` performs 50 source-pretraining optimizer updates and 20 response/EOS-masked SFT updates, then sends each staged snapshot through five representative offline Docker tasks covering all supported languages. Run it with:

```powershell
npm run brain:smoke
```

The profile is intentionally small and CPU-selected for predictable portability. It runs one actual task per supported language (Python, JavaScript, TypeScript, Go, Rust) while preflighting every local language image. It remains an integration smoke test, not evidence of model quality or a timing guarantee; Docker image availability and task execution still determine total duration. Its partial five-task report is rejected by the promotion utility, which requires the full current 13-task suite.

`configs/training/local_cpu_fast.json` targets low-memory CPU machines with a 4-layer/width-64 model, 256-token context, physical batch 1, and gradient accumulation. `configs/training/local_gpu_scale.json` uses 8 layers, 8 heads, width 256 and 1024-token context for suitable dedicated GPU or high-memory unified-memory systems. Both keep micro-batches distinct from optimizer steps. Accumulation normalizes gradients by actual supervised-token counts and clips once per optimizer update. Select the backend explicitly to avoid silently using CPU on a machine where GPU acceleration was intended.

```powershell
npm run brain:milestone4 -- --config configs/training/local_cpu_fast.json --device cpu
npm run brain:milestone4 -- --config configs/training/local_gpu_scale.json --device cuda
npm run brain:milestone4 -- --config configs/training/local_gpu_scale.json --device mps
```

### Verified checkpoint promotion and rollback

A benchmark report is selected explicitly. The promotion tool authenticates the report's exact staged checkpoint by SHA-256, checks the adjacent read-only snapshot manifest, requires the entire current 13-task suite with valid hidden baselines, and loads the candidate through SYNTRA's actual runtime before it can replace the active checkpoint.

```powershell
npm run brain:promote -- --report "data/benchmarks/<profile>/<run-id>/evaluations/<checkpoint-report>.json" --dry-run
npm run brain:promote -- --report "data/benchmarks/<profile>/<run-id>/evaluations/<checkpoint-report>.json" --confirm-sha256 "<exact-sha256-from-dry-run>"
npm run brain:promote -- --rollback "<archive-id-returned-by-promotion>"
```

The active checkpoint target is resolved using the same `SYNTRA_CHECKPOINT` override and default/fallback resolver as the application. The previous active checkpoint, JSON metadata, and inference companion are archived under `data/checkpoints/archive/<timestamp-id>/` before replacement. Rollback validates those hashes, restores the previous full checkpoint, and regenerates the derived inference companion so its embedded source modification time matches the restored checkpoint. Promotion is explicit and never triggered automatically by training. All file operations are local; Docker benchmark execution remains `--pull=never` and `--network none`.

## Comparability and limitations

Do not combine raw-model results with full-agent results. The direct raw-model suite now has 13 small held-out fixtures across five languages, but it is a controlled engineering regression matrix—not a representative sample of unfamiliar repositories or an industrial coding benchmark. JavaScript event normalization, Go concurrency, Rust memory safety, TypeScript type-system coverage, and multi-file Python behavior are each represented by one narrow task; that is not broad language competence.

GitHub CI installs the required Go/Rust toolchains to validate the fixture baselines and the runner implementation. Actual benchmark execution remains offline: it uses pre-provisioned local Docker images and does not pull images or packages. The benchmark does not fabricate a score when a checkpoint, local image, language runtime, or valid fixture baseline is unavailable.

For meaningful comparisons, retain the same suite version and task IDs, run the same checkpoint on the same hardware and device, and keep system load as similar as practical. Compare exact checkpoint identities and measured outputs. Do not infer skill improvement from training loss alone. Record a real baseline run before and after a training or agent change.

# SYNTRA Product Roadmap

SYNTRA is intended to become a local-first software engineering assistant with the workflow depth of tools such as Claude Code, while keeping the model, project files, and execution environment under the owner's control.

This roadmap separates shipped behavior from planned capabilities. A checked item means code exists in the repository; it does not imply that every use case is production-ready.

## Product principles

- **One self-owned AI engine.** Project source, prompts, training data, checkpoints, inference, review, and embeddings stay in the SYNTRA-controlled local runtime. Alternate model providers and hosted inference are outside the project scope.
- **Real tools, not simulated tools.** A capability is only advertised after it has a working implementation, bounded permissions, observable results, and tests.
- **Reviewable changes.** File modifications are staged, verified where possible, previewed, and protected by source hashes and backups.
- **Explicit execution boundaries.** Reading a file, modifying a file, and executing a command are different permission levels. A coding prompt must not silently become arbitrary shell execution.
- **Honest model reporting.** A locally trained model is not described as a frontier model. Model quality is measured with task-level evaluations, not branding or training-step counts.
- **Small vertical slices.** Each phase must build, pass its relevant tests, and leave the project runnable before the next phase begins.

## Current baseline

The repository already contains:

- A minimal React chat interface with bounded, local multi-conversation history, automatic titles, search, rename/delete/resume, and migration from the original single-thread store.
- A loopback-only Node service that routes chat, network diagnostics, workspace operations, and coding requests.
- SYNTRA's native PyTorch Transformer is the only model used for chat, code generation, review, tokenization, and embeddings.
- Workspace inspection, text search, file reading, a local import graph, relevance-ranked source context, and a semantic index built from native SYNTRA checkpoint embeddings.
- Structured exact-anchor patches with path and secret-file validation.
- Candidate verification in a temporary project copy, bounded repair attempts, and hash-checked proposal application with backups and an audit log.
- Local ping, DNS, and single-port TCP diagnostics.

Known limitations are deliberate and must remain visible: there is no general-purpose terminal tool in the chat, no multi-agent execution system, and no evidence that the from-scratch native model performs at a Claude-class level. Conversation history and the selected workspace path are browser-local to one device; workspace proposal records are stored in each project's `.syntra` directory and can be recovered by the local service after a restart.

## Delivery phases

### Phase 1 — Trustworthy change review

**Objective:** Make every coding result inspectable before a user accepts it.

- [x] Return a per-file diff preview from the exact before/after source.
- [x] Bound large previews and report when content was omitted.
- [x] Show the preview beside the verification result in chat.
- [x] Verify the full CI matrix on the review branch before merge.

**Acceptance:** A coding task reports each changed path, the reason, a readable preview, and the actual check status. A pending proposal clearly requires explicit approval. Source hashes are rechecked before application.

### Phase 2 — Workspace and session experience

**Objective:** Make the chat operate as a persistent project workspace without turning it into a noisy administration dashboard.

- [x] Add a visible active-workspace indicator and a predictable project-switch flow.
- [x] Add multiple named conversations, local session persistence, rename/delete, search, and resume.
- [x] Keep session state separate from individual chat messages and migrate the legacy single-thread history.
- [x] Restore pending proposals after service restarts and let users inspect, apply, or discard them.
- [x] Preserve clear empty, loading, offline-model, and failed-request states while introducing the session experience.

**Acceptance:** A user can reopen SYNTRA, return to a named conversation, confirm or switch the selected workspace, and recover staged proposals with hash validation after a local service restart. Pending/stale queues remain visible and must be applied or discarded before another coding task starts.

### Phase 3 — A real, permissioned tool runtime

**Objective:** Give the agent reliable tools instead of asking a model to imitate tools in prose.

Initial tools:
- [x] Search project content and retrieve relevant file paths and matching lines.
- [x] Read files using bounded line ranges and path/secret protections.
- [x] Inspect project structure, manifests, scripts, and read-only Git status/diff summaries.
- [x] Stage file creation and edits as hash-bound reviewable proposals; deletion is not supported yet.
- [x] Run project checks through an approval-gated tool inside an isolated candidate copy.
- [x] Validate tool inputs, declare permission levels, serialize per-workspace calls, and record bounded audit events.

Later execution tools:
- A terminal runner with working-directory restrictions, command policy, timeout and process-tree termination, output limits, and explicit approval for commands that can mutate state or access the network.
- Git operations with preview-first behavior and explicit confirmation for commits, branch changes, and pushes.

Every accepted tool invocation records bounded input/output summaries, result, duration, run ID, approval and permission decisions. Raw file contents are deliberately excluded from tool audit logs. Repository content is untrusted input, not an instruction source.

**Acceptance:** Tool actions are structured and testable. The UI identifies the tools used in coding-task summaries. Project checks, source writes, and destructive proposal actions are approval-gated. No command is executed merely because repository content or a model response asked for it. A general-purpose terminal runner and model-selected execution or source-modification tools remain future work. Phase 4 includes a limited model-selected read-only retrieval step using the existing search and read tools.

### Phase 4 — Multi-step agent orchestration

**Objective:** Move from one patch request to a controlled plan–act–observe–repair workflow.

- Keep an explicit task plan with completed, active, blocked, and pending steps.
- Select tools based on observed repository state and previous tool results.
- Re-inspect files after each meaningful change.
- Verify after each logical implementation slice instead of postponing all checks.
- Support cancellation, resumable checkpoints, and bounded retries.
- Record model, context budget, selected files, tool calls, patch hashes, and verification results for each run.

**Delivered durability slice (partial Phase 4 implementation):**

- [x] Persist bounded per-workspace run records with task goal, plan, context hashes, attempts, tool-call summaries, verification state, proposal IDs, and event history.
- [x] Record local model identity and context/output budgets for each run.
- [x] Recover unfinished runs at local-service startup and when switching workspaces; keep ordinary run-history reads non-mutating.
- [x] Stop active local inference and supported candidate-check processes, while protecting an in-progress transactional apply.
- [x] Resume a validated patch only after rechecking its original file hashes and exact anchors; otherwise restart discovery/generation.
- [x] Show recent runs in the workspace panel and refresh their state when the panel is reopened.
- [x] Collect bounded, read-only Git status and diff-summary observations before patch generation; pass them to the model as untrusted context.
- [x] Discover executable build/test/lint/type-check runners from the actual candidate workspace instead of inferring runnable checks from manifest presence alone.
- [x] Re-read every changed candidate file after project checks and reject a verification pass if a check script changed or removed a proposed source file.
- [x] Persist bounded repository, check-discovery, and candidate-integrity observations with each agent run; surface the evidence count in the workspace panel.
- [x] Cover supported-runner discovery, evidence persistence, and post-check candidate mutation with regression tests.
- [x] Let the local model select up to four bounded `workspace.search`/`workspace.read` calls for task-specific context; validate the tool allowlist, query/line limits, and safe file paths before audited execution.
- [x] Constrain native-model tool-plan output with JSON schema and fall back to deterministic context ranking when the plan is invalid or exceeds the context budget.
- [x] Add bounded iterative repository investigation: the model reviews prior search/symbol/read results before selecting targeted follow-up reads, with five planning rounds and twelve unique read-only calls maximum. File reads are capped at 400 lines.
- [x] Persist each retrieval round as an auditable observation and stop early when the model reports that the evidence is sufficient.
- [x] Add an independent pre-verification patch reviewer that compares candidate changes with the user request, slice scope, and retrieved repository evidence.
- [x] Feed concrete blocking review findings into a bounded patch-correction attempt; record advisory findings without blocking.
- [x] Constrain reviewer output to a strict JSON schema and validate every referenced path against the inspected workspace.
- [x] Keep the reviewer read-only: it cannot call tools, execute commands, or modify candidate/workspace files.
- [x] Cover reject-correct-accept behavior, malformed review decisions, workspace-path validation, and reviewer schema output with tests.
- [x] Cover allowed retrieval plans, disallowed execution tools, path/range violations, evidence-driven follow-up selection, early stopping, and end-to-end retrieval-to-patch context flow with tests.
- [x] Let the selected model propose up to eight ordered implementation slices with explicit acceptance criteria; derive a smaller maximum from task and workspace complexity, validate every slice and fall back to one slice if planning fails.
- [x] Reuse one cumulative temporary candidate across slices, re-inspect and rebuild context after each verified slice, and re-inspect before bounded repair attempts after failed verification.
- [x] Aggregate multi-slice changes against original workspace contents and source hashes; stage only the final cumulative proposal set.
- [x] Mark a task unverified when any planned slice is incomplete. Do not apply a partial multi-slice result automatically.
- [x] Persist each verified multi-slice candidate boundary in a separate bounded workspace-local checkpoint; validate run/workspace identity, content hashes, and original-file hashes on resume.
- [x] Reconstruct the temporary candidate from a verified checkpoint after service restart, re-inspect it, and continue at the next pending slice; regenerate an in-flight or final-not-yet-staged slice. Persist checkpoints only when at least one slice remains pending.
- [x] Remove candidate checkpoints after final proposal staging and report checkpoint persistence failures rather than claiming restartability.
- [x] Cover candidate checkpoint bounds, invalid paths and hashes, symlink protection, interrupted-run continuation, and original-source preservation with tests.
- [x] Cover two-slice verification, candidate-context refresh, cumulative proposal hashes, and incomplete-slice failure handling with tests.
- [x] Cover durable storage, lifecycle transitions, changed-source rejection, and actual service restart recovery with tests.

Still outstanding in this phase: model-selected execution/source-modification tool choice, serialization of arbitrary intermediate model/tool operations within a slice, and optional parallel execution of independent read-only tasks. A bounded candidate snapshot is now persisted before project checks; after restart SYNTRA validates original hashes and re-runs checks before resuming from that candidate. The current workflow remains bounded and permissioned; model-selected source writes and arbitrary shell execution are not claimed as delivered.

**Acceptance:** A complex task can complete multiple verified slices, resume after interruption, and explain which evidence supports its final status. Retries are bounded and cannot bypass safety checks.

### Phase 5 — Project context and engineering memory

**Objective:** Improve context quality without silently collecting sensitive data.

**Delivered initial Phase 5 slice (partial):**

- [x] Build a workspace-local incremental declaration index for JavaScript/TypeScript, Python, Go, and Rust.
- [x] Persist bounded symbol metadata and content hashes only; do not store source-file bodies.
- [x] Reuse parsed declarations for unchanged file hashes, re-parse changed files, and prune deleted files.
- [x] Validate indexed source hashes before returning symbol-search results; reject stale entries and symbolic-link index files.
- [x] Add audited index status, refresh, and symbol-search tools, and allow the agent to select symbol search as a bounded read-only retrieval operation.

- [x] Expand the incremental index with bounded local import edges, documentation references, and heuristic test ownership.
- [x] Add bounded symbol search to model-selected read-only retrieval so the agent can retrieve declarations instead of relying only on full-text matches.
- [x] Test JS/TS and Python import resolution, Go module-local package imports, Rust module declarations, documentation link resolution, test-owner inference, and hash validation of graph relationships.
- [x] Store explicitly user-authored project notes locally and bind each store to its resolved workspace root.
- [x] Provide controls to list, search, add, edit, and delete saved notes; note changes and deletion require explicit approval.
- [x] Keep note contents out of tool audit events and limit agent retrieval to bounded, read-only search.
- [x] Enforce note-count, file-size, title, body, and tag limits; reject known credential patterns and refuse symlink or invalid stores.
- [x] Add workspace-specific relative glob exclusions for discovery, full-text search, symbol indexing, direct reads, and proposal access; store settings locally with optimistic-concurrency protection and reject invalid or symlinked configuration.
- [x] Add an on-demand, audited workspace secret scanner for common private-key and credential patterns. It respects path exclusions and scan limits, and returns only finding metadata; matched values and source lines are never returned or written to audit logs.
- [x] Add a local model-training smoke profile that runs bounded source pretraining and masked SFT, then executes representative Python, JavaScript, TypeScript, Go, and Rust tasks through the existing offline Docker benchmark path.
- [x] Capture measured supervised tokens/second, optimizer-step latency, process peak RSS, CUDA allocator peak or sampled MPS high-water, estimated working-set size, and gradient-accumulation factor in checkpoint-matched Markdown/CSV evaluation trends.
- [x] Add low-memory CPU and 1024-token GPU architecture profiles, with true optimizer-step gradient accumulation that keeps only one micro-batch of activations on the selected device at a time.
- [x] Add explicit checkpoint promotion and rollback. Promotion verifies complete current-suite benchmark evidence, hidden baseline validity, immutable snapshot manifest, SHA-256 identity, and runtime loadability; it archives the previous active checkpoint and reconstructs its inference companion on rollback.
- [x] Expand provider-specific credential and private-key pattern coverage; add explicit approval-required, hash-bound, detector-scoped secret allowlists with reason, expiry, atomic local storage, fail-closed validation, revocation, and audit redaction. Pattern scanning remains best-effort and is not a substitute for a secret manager.

**Acceptance:** Context selection is explainable, reproducible, workspace-scoped, and covered by tests for stale-index invalidation, explicit memory approval, audit redaction, workspace isolation, and best-effort secret detection.

### Phase 6 — Model quality, evaluation, and release hardening

**Objective:** Measure engineering usefulness instead of assuming it.

**Delivered evaluation slice (initial):**

- [x] Add a versioned local coding benchmark suite with eleven deterministic fixtures spanning JavaScript utilities, async/API behavior, multi-file changes, Python unittest, Go test and Rust Cargo test; non-JavaScript tasks are skipped when their required runtime is unavailable.
- [x] Run the real coding agent against disposable fixture copies using the selected local model; do not apply proposals to the active workspace.
- [x] Require the exact expected changed-file set and passing candidate verification for task completion; extra test/configuration edits invalidate the patch score.
- [x] Record native engine/checkpoint identity, task latency, prompt/output tokens, patch validity, verification outcomes, task completion, and benchmark-process peak RSS in JSON reports.
- [x] Add regression coverage for suite validation, scoring, aggregation, fixture isolation, baseline checks, and report output.

**Delivered reporting slice:**

- [x] Preserve timestamped machine-readable run reports under a Git-ignored local history directory.
- [x] Add a report-history command with bounded listing and corrupt/symlink report rejection.
- [x] Compare runs only when suite version and exact task selection match; report task-level and aggregate quality/performance changes without a composite score.
- [x] Capture CPU, memory, runtime, and architecture metadata and mark latency/RSS comparisons incomparable when environments differ.
- [x] Test report validation, improvement/regression detection, task-set matching, and environment comparability.

Remaining measurement limitations: peak RSS covers only the benchmark Node.js process, not the separate native AI service or GPU memory. Native prompt counts use SYNTRA's tokenizer. Hardware metadata cannot prove identical background load or power/thermal conditions. The eleven-task deterministic suite now includes Python, Go and Rust tasks, but the non-JavaScript fixtures are small language-smoke tasks, not broad framework coverage. The suite is not evidence of parity with frontier engineering agents until real model runs and a larger, versioned benchmark corpus have been measured.

- [x] Create a versioned coding benchmark from representative local tasks and regression fixtures.
- [x] Record patch validity, candidate-test verification, task completion, end-to-end latency, prompt/output tokens, and benchmark-process RSS with each model run.
- [x] Add native checkpoint and capability metadata that distinguishes locally measured attributes from unmeasured engineering task performance. No model is labeled frontier-equivalent.
- [x] Add bounded native-generation cancellation and streaming safeguards; classify context-window and model-memory exhaustion; reject incomplete streams and never replay partial streamed output.
- Review local-service origin checks, filesystem boundaries, command permissions, dependency handling, and Windows launch behavior.
- [x] Commit `package-lock.json`, enforce clean `npm ci` installs in CI and the Windows launcher, and configure weekly Dependabot npm/pip update pull requests.
- [x] Add a universal SHA-256 hash-locked Python requirements file with CPU-only PyTorch markers; enforce lock freshness and hash installs in Linux/Windows CI and the native-runtime installer.
- [x] Reject case-variant access to the reserved `.syntra` state directory and cross-platform Windows path aliases, including NTFS alternate data streams, reserved device names, and trailing-dot/space names.
- [x] Remove alternate model selection and external-provider integrations; the Windows launcher trains, validates, and starts only SYNTRA's native checkpoint.
- [x] Use native checkpoint embeddings for semantic source retrieval and user-authored project memory; persist only vectors, path/line metadata and source hashes, then revalidate hashes before returning excerpts.
- [x] Expand orchestration to a complexity-derived maximum of eight slices, six attempts per slice and a 24-attempt total ceiling. Persist the global attempt counter and execution budget across local-service restarts.
- [x] Harden Docker check containers with init, private IPC, bounded swap and an open-file limit; retain no-network, read-only root, dropped capabilities, no-new-privileges and candidate-only write mount.
- [x] Persist a bounded, hash-validated in-flight candidate snapshot before project checks. On restart, restore it only after validating workspace identity and original-file hashes, re-run discovered checks and candidate-integrity verification, then continue or repair from the restored candidate.
- [x] Add checkpoint-aware context sizing with output caps and safety reserves; derive exact prompt counts from SYNTRA's saved tokenizer.
- [x] Implement report-derived per-language comparison rows with exact hidden-baseline-valid denominators, completion/patch/regression rates, hardware-gated performance, and explicit small-sample/zero-valid warnings.
- [ ] Populate and assess the cross-language capability matrix using repeatable live-checkpoint runs across TypeScript, Python, Go and Rust fixtures; implementation and fixture baselines are not a substitute for measured model outcomes.
- [x] Expand the raw native-model suite from its initial Python-only baseline to version 1.4.0 with 14 independently validated held-out fixtures: nine Python tasks, two JavaScript tasks, and one each for TypeScript, Go and Rust.
- [x] Add coordinated multi-file tasks in Python and JavaScript; present connected modules together and require one exact, non-duplicated changed-path set.
- [x] Add language-specific offline runtime selection, compiler/test validation, local-image-only Docker preflight, and per-language score breakdowns without combining the direct-model track with the full-agent suite.
- [x] Record exact prompt-token count and context-window utilization per task; retain overflow-token evidence for context-rejected prompts and reject context overflow or truncated generation streams rather than silently dropping source context.
- [ ] Establish a repeatable raw-checkpoint benchmark baseline on target hardware before comparing training/curriculum changes; do not use training loss as a substitute for task-level outcomes.
- [x] Add bounded, validated training-loss history to native source pretraining and instruction fine-tuning checkpoints, including inference sidecars and model status.
- [x] Add a direct native-model benchmark track that calls the checkpoint streaming runtime, hides target/regression tests from model context, reuses SYNTRA's production patch validator, and separates patch validity, target completion, and regression prevention.
- [x] Correlate actual saved checkpoint loss metadata/history with raw-model task outcomes and emit a clearly qualified loss/intelligence divergence warning.
- [x] Add a local Milestone 4 runner that executes bounded source pretraining, audits AST-derived curriculum JSONL with the checkpoint tokenizer, and then performs response/EOS-only masked SFT.
- [x] Add the v2 resumable warmup/cosine scheduler with an explicit 1e-5 floor, exact optimizer-step blocks, non-finite gradient rejection, and a persisted 1.0 gradient-clipping policy while retaining legacy v1 resume behavior.
- [x] Prevent synthesized instruction JSONL from contaminating the source-pretraining corpus; record exact dataset inclusion/exclusion in the checkpoint fingerprint.
- [x] Stage read-only SHA-256-verified checkpoint/sidecar/inference snapshots and accept benchmark metrics only when the raw-model report confirms the exact staged checkpoint hash.
- [x] Add a single-checkpoint baseline-only run mode and local Markdown/CSV trend summaries; failed or absent benchmark measurements remain unavailable rather than being imputed as zero.
**Phase 5 local model operations and hardware baseline extension:**

- [x] Add a CPU smoke profile (50 pretraining optimizer updates and 20 masked SFT updates) that exercises the real checkpoint snapshot and complete offline Docker evaluation handoff.
- [x] Add low-memory CPU and scaled 8-layer/8-head, width-256, 1024-token GPU training profiles with true gradient accumulation at optimizer-step boundaries.
- [x] Capture measured supervised tokens/second, optimizer-step latency, process peak RSS, and available CUDA/MPS allocator high-water telemetry in checkpoint-comparison Markdown/CSV and per-run JSON.
- [x] Add benchmark-report-bound checkpoint promotion: require the complete current 21-task suite, valid hidden baselines, exact staged snapshot SHA-256, and successful native-runtime checkpoint loading.
- [x] Archive the previous active checkpoint, metadata and inference companion under a timestamped ID; provide hash-verified rollback and rebuild the derived inference companion against the restored checkpoint timestamp.
- [x] Keep promotion explicit, local and offline; training never auto-promotes or modifies loopback service/network policy.

- [x] Publish a reproducible release checklist and upgrade notes covering CI/CodeQL, the native runtime, isolation, native semantic retrieval, benchmark history and release gates.

**Acceptance:** Release claims are backed by repeatable measurements. The installation path, test results, supported hardware, and model limitations are documented.

### Phase 19 — End-to-end chat cancellation integrity

**Objective:** Make the local chat cancellation control stop the work it claims to stop, across ordinary model responses, workspace shortcuts, proposal operations, and local network diagnostics.

- [x] Propagate the chat AbortSignal into normal-chat tokenization and streamed native inference requests.
- [x] Propagate cancellation into /secrets, proposal listing, /apply, /discard, /status, workspace switching, and local network checks.
- [x] Destroy active TCP probe sockets and terminate ping child processes on cancellation; stop waiting on DNS results promptly.
- [x] Make cancellation terminal for a chat job so a late response cannot overwrite its cancelled status.
- [x] Keep cancellation scoped to the workspace captured by the job when persisting agent-run state.
- [x] Add HTTP integration coverage proving /cancel closes an active native inference response and leaves the chat job cancelled.
- [x] Add a network-diagnostic regression proving TCP cancellation destroys the underlying socket.

**Acceptance:** A cancellation request stops cancellable in-flight work, prevents late completion from replacing the terminal cancelled state, and leaves workspace/proposal operations under their existing transactional and explicit-approval rules. Cancellation is not reported as successful application or completed work.

### Phase 28 — Native task-aware skill retrieval

**Objective:** Make SYNTRA's existing local engineering skill library operational in the native coding-agent workflow without introducing an alternate inference provider.

- [x] Discover bundled SYNTRA skills and optional skills under the active workspace.
- [x] Select the project guardrails and verification guidance when available, then rank task-specific playbooks deterministically.
- [x] Reject symbolic-link skill folders/files, hard-linked skill documents, path escapes, unreadable entries, and oversized skill documents.
- [x] Bound selected content and fit it with the native checkpoint tokenizer; compact or omit optional guidance before reducing the user's task or patch contract.
- [x] Treat all skill text as untrusted reference context that cannot change the tool allowlist, permission boundary, or proposal workflow.
- [x] Record selected skill identities, excerpt hashes, and retrieval/compaction outcomes with agent-run evidence.
- [x] Add loader, filesystem-boundary, deterministic-ranking, context-compaction, and native-agent prompt integration tests.

**Acceptance:** A task-relevant local skill reaches the actual native patch-generation prompt when the checkpoint context budget permits. Missing or unsafe skill libraries do not block coding tasks, and skill content never authorizes additional operations.

**Limitation:** Skill retrieval supplies procedural guidance only; it does not provide pretrained weights or prove model-quality gains. Model capability must continue to be measured using real checkpoint benchmark runs.

### Phase 29 — Native-only inference boundary

**Objective:** Enforce the project's self-owned PyTorch model architecture consistently across runtime code, startup, benchmarks, tests, and documentation.

- [x] Remove the optional external inference bridge and endpoint configuration.
- [x] Remove provider-specific branches from the workspace server, coding agent, and benchmark runner.
- [x] Require an actual trained native checkpoint for repository coding tasks and benchmark execution; report missing checkpoints explicitly.
- [x] Always prepare and validate the local Python/PyTorch environment and checkpoint during the Windows one-click startup.
- [x] Restore native semantic retrieval to model-selected read-only context planning without alternate-provider exclusions.
- [x] Replace alternate-provider startup tests and CI assertions with native-only regression checks.
- [x] Align the README, model-capability guide, release checklist, benchmark guide, changelog, and architecture diagram with the implemented runtime boundary.

**Acceptance:** All model operations use the local SYNTRA checkpoint. The runtime, launcher, and benchmark paths do not contact third-party inference endpoints, external embedding services, or model registries. A missing or invalid checkpoint produces an explicit failure rather than an alternate-model fallback.

## Definition of done for every phase

A phase is not complete merely because the interface looks finished. It must include:

1. An implemented end-to-end user flow.
2. Unit and integration coverage for normal and failure cases.
3. A successful production build and relevant CI jobs.
4. Security review of filesystem, process, network, and model-input boundaries affected by the change.
5. Updated user documentation that describes actual behavior and remaining limitations.



### Phase 17 — Content-grouped raw dataset holdouts

- [x] Group exact duplicate raw source contents before deterministic train/validation assignment.
- [x] Fail closed when a required holdout has fewer than two independent source contents.
- [x] Add regressions for duplicate files across paths and duplicate-only corpora.

**Acceptance:** Exact copies of a source file cannot cross the raw training/validation boundary. This is exact-content deduplication, not semantic near-duplicate detection.


### Phase 30 — Document-isolated native source pretraining and evaluation

**Objective:** Make source-pretraining validation evidence independent at the source-document level instead of cutting a concatenated token stream.

- [x] Assign complete source documents to training or validation using deterministic, size-aware 80/20 partitioning.
- [x] Group byte-identical source contents before partitioning and count unique documents against the source token budget only once.
- [x] Learn tokenizer BPE merges exclusively from training documents; encode held-out documents with that frozen tokenizer.
- [x] Preserve complete documents in corpus loading and never truncate a source file to fill the remaining token budget.
- [x] Require at least two independent documents and enough tokens for both context windows; fail explicitly instead of reverting to token-offset validation.
- [x] Bind the validation strategy to the checkpoint training-data fingerprint and expose split counts in checkpoint metadata and model status.
- [x] Apply the same document-isolated split to fresh corpus evaluation and report the validation provenance.
- [x] Add deterministic traversal, duplicate-content, holdout, resume-compatibility, and real training/checkpoint regression tests.
- [ ] Run a fresh target-hardware raw-checkpoint baseline and compare actual coding-task outcomes before claiming a model-quality improvement.

**Acceptance:** No exact source document or byte-identical copy crosses the training/validation boundary. The tokenizer is fitted only on training content. The split is reproducible, bounded by the configured source budget, independently tested, and explicitly reported. Historical token-offset checkpoints cannot silently resume under the new validation strategy.


### Phase 31 — Resumable cosine scheduling in the application trainer

**Objective:** Align the production `syntra_ai` source-pretraining path with the tested global warmup/cosine scheduler used by the standalone native trainer, without changing the optimizer trajectory of existing checkpoints.

- [x] Replace the active trainer’s warmup-then-constant learning rate for new checkpoints with a global-step warmup/cosine schedule, 100 warmup steps, a 10,000-step decay horizon, and an explicit learning-rate floor.
- [x] Persist scheduler identity and exact schedule/gradient-clipping settings in full and inference checkpoints and expose them through model status.
- [x] Preserve the pre-existing linear-warmup/constant schedule when resuming checkpoints that predate scheduler metadata.
- [x] Reject non-finite gradient norms before optimizer updates and record actual pre-clipping gradient norm telemetry.
- [x] Test scheduler boundaries, invalid settings, checkpoint persistence, global-step resume continuity, and legacy checkpoint compatibility.
- [ ] Measure schedule variants on the same raw native-model benchmark suite and target hardware before changing the fixed defaults or claiming a quality improvement.

**Acceptance:** A training run resumed across process boundaries computes the same learning rate as one uninterrupted global-step trajectory. Old checkpoints retain their original schedule; new checkpoints carry enough metadata to reject incompatible scheduler changes. Non-finite gradients cannot silently enter the optimizer.


### Phase 32 — Stage-scoped instruction fine-tuning schedule

**Objective:** Replace the active application SFT path's constant learning rate with a bounded warmup/cosine schedule while making the optimizer lifecycle explicit.

- [x] Derive a deterministic per-stage warmup from the requested optimizer-step budget (10%, capped at 100 steps) and decay to the configured floor at the stage boundary.
- [x] Use the same validated native scheduler implementation as source pretraining without coupling SFT steps to the source-pretraining global step.
- [x] Persist the exact SFT schedule, stage step, current learning rate, clipping threshold, and measured pre-clipping gradient norm in full/inference checkpoints and model status.
- [x] Keep new SFT stages distinct while making each stage's AdamW optimizer state resumable across an interrupted invocation.
- [x] Add schedule-boundary and invalid-configuration tests plus a real SFT checkpoint/status integration assertion.
- [x] Implement explicit dataset-bound SFT optimizer/RNG resume with an operator-visible resume contract and mismatch rejection.

**Acceptance:** The SFT learning rate follows the requested stage's optimizer-step budget, prompt-mask/response-only loss and gradient accumulation remain unchanged, and checkpoint/status records distinguish stage-scoped scheduling from resumable source pretraining.


### Phase 33 — Resumable supervised fine-tuning checkpoints

**Objective:** Make an interrupted instruction fine-tuning stage recoverable without silently resetting AdamW, scheduler position, or random sampling state.

- [x] Add an explicit `resume` argument to the native trainer, CLI, and local API; preserve fresh-optimizer behavior when starting a new stage.
- [x] Load the full training checkpoint through the bounded snapshot loader for resume; never use the inference-only companion as a source of optimizer state.
- [x] Bind resume to the exact instruction-dataset SHA-256 and byte count, original target step count, learning rate, batch size, gradient accumulation, weight decay, schedule, device type, and CUDA device count.
- [x] Restore AdamW and CPU/CUDA RNG state, keep the original stage-relative cosine schedule, and reject missing/legacy/incompatible state or completed stages.
- [x] Atomically persist model, optimizer, scheduler position, dataset identity, and RNG state after every optimizer update; refresh inference snapshots at validation points.
- [x] Add CPU interrupted-versus-uninterrupted weight/optimizer equivalence coverage, mismatch rejection tests, and API/CLI contract tests.
- [x] Document the total-target `steps` semantics, CPU/CUDA resume scope, and MPS limitation.

**Acceptance:** An interrupted CPU SFT stage resumes from its last committed optimizer update and reaches the same model/optimizer state as an uninterrupted run. Changed data/settings, incomplete state, unsupported devices, or a completed stage must fail explicitly instead of silently starting over.


### Phase 34 — Recover authoritative native training progress in model status

**Objective:** Prevent a process interruption between the full-checkpoint commit and the JSON progress write from leaving a stale optimizer step silently visible.

- [x] Bind source-pretraining and SFT JSON progress records to the full checkpoint's device/inode/size/mtime/ctime identity.
- [x] Detect checkpoint/metadata identity mismatches and attempt a one-time migration for legacy metadata without an identity record.
- [x] Reconcile stale source-pretraining or resumable-SFT metadata from the bounded, same-bytes checkpoint snapshot and verify the checkpoint identity again after metadata repair.
- [x] Atomically repair allowlisted progress fields from the full checkpoint, including global step, scheduler settings, SFT stage position, loss, and training history.
- [x] Report an explicit stale warning when reconciliation fails, the checkpoint is malformed, or the checkpoint changes during reconciliation.
- [x] Add regressions for successful repair, idempotent subsequent reads, and fail-closed stale reporting.

**Acceptance:** A checkpoint committed ahead of its JSON sidecar is never silently reported as the older step. Status reconciles only from a stable, verified native source-pretraining or resumable-SFT checkpoint; otherwise it exposes the stale/unknown identity state honestly.


### Phase 35 — Expand secret detection with reviewed, hash-bound exceptions

**Objective:** Improve common credential coverage without silently suppressing findings or weakening the existing workspace boundary.

- [x] Add provider-specific token patterns and encrypted/PGP private-key header detection while retaining generic credential-assignment and connection-string patterns.
- [x] Keep findings metadata-only: relative path, line, detector ID, severity, and exact file SHA-256; never return matching values or source lines.
- [x] Add a versioned local allowlist scoped to one workspace-relative file, exact SHA-256, and explicitly selected detector IDs.
- [x] Require a reviewer reason, a strict expiry date with a maximum 366-day lifetime, and an explicit approval-required tool call for every allowlist write or revocation.
- [x] Re-read and hash the source file immediately before approval is persisted; any file change invalidates the exception.
- [x] Store allowlist updates atomically under the workspace's validated .syntra state directory, reject symlink/malformed state, and write restrictive POSIX permissions.
- [x] Record audit metadata without the source path, reason, matched value, or source line; preserve fail-closed behavior when audit start or allowlist validation fails.
- [x] Test additional provider patterns, explicit approval and revocation, file-hash invalidation, mismatch rejection, malformed/symlinked state, exclusion behavior, and audit redaction.

**Acceptance:** Only a user-reviewed detector exception for one exact file hash can suppress that detector in that file. An edited file, expired exception, malformed allowlist, or missing approval must leave the detection active.


### Phase 36 — Raw benchmark hardware provenance and controlled report comparison

**Objective:** Make raw native-model timing evidence traceable to the actual host/backend, and prevent incomplete or mismatched execution environments from producing controlled-performance claims.

- [x] Resolve the actual inference device once before a raw benchmark and use the same explicit resolved backend/device index for all tasks.
- [x] Record measured host OS/release/architecture, exposed CPU model, logical cores, installed physical memory, Python identity, PyTorch build, selected device and accelerator properties where the active runtime exposes them.
- [x] Preserve unknown CPU/GPU/driver/memory properties as null with explicit limitations; measure NVIDIA/AMD driver identity from local OS tools when available and keep CUDA/MPS identities incomplete when the selected backend cannot be fully identified.
- [x] Add strict raw-report validation for checkpoint and evaluation-protocol identities, task outcomes, summary denominators, and hardware fingerprint integrity.
- [x] Add a raw-report comparison CLI that requires matching suite/protocol/task sets and hidden-baseline-valid denominators.
- [x] Separate task quality from hardware-dependent performance, marking latency/throughput incomparable when identity is incomplete/different and marking regression/throughput rates incomparable when measured task subsets differ.
- [x] Add CPU/CUDA/MPS metadata tests, missing-measurement tests, report-integrity tests, denominator/protocol mismatch tests, and CLI regression coverage.

**Acceptance:** Every new raw benchmark report identifies the actual selected backend and records only measured system/device properties. A comparison cannot claim a controlled latency/throughput delta unless both complete execution identities match. This phase implements provenance and comparison tooling only; it does not satisfy the separate target-hardware baseline or empirical language-capability roadmap items.

### Phase 37 — Per-language raw benchmark outcome matrix

**Objective:** Make cross-language raw-model comparisons interpretable without changing the task set, hiding invalid fixture baselines, or presenting small samples as general capability evidence.

- [x] Derive language rows directly from task-level baseline/candidate records using each language's own hidden-baseline-valid task denominator.
- [x] Report selected fixtures, valid fixtures, excluded invalid baselines, completion/patch outcomes, and separately denominated regression-prevention measurements.
- [x] Mark per-language latency and generation throughput incomparable when hardware identity or the corresponding measured-task denominator differs.
- [x] Preserve null/unavailable rates for languages with no valid fixtures and attach a visible warning to languages with fewer than three valid fixtures.
- [x] Expose the matrix in comparison JSON and the comparison CLI without running inference or fabricating scores.
- [x] Test multi-language reports, invalid-baseline exclusion, language-specific regression-denominator mismatch, hardware gating, small sample warnings, and CLI output.

**Acceptance:** Every language row can be traced to exact task IDs and task-level outcomes. Invalid baselines never enter rates, unmeasured metrics remain unavailable/incomparable, and the output explicitly states that fixture-level results do not prove general language competence.

**Outstanding evidence:** The separate empirical capability-matrix item remains open until actual live-checkpoint reports have been produced on the intended hardware.

### Phase 38 — Expand held-out polyglot benchmark fixture coverage

**Objective:** Improve the balance of the raw-model evaluation matrix so each non-Python language has multiple independent task categories while preserving real failing baselines and independent regression tests.

- [x] Add TypeScript generic result mapping with typed error preservation and own-property picking with inherited-property exclusion.
- [x] Add Go saturating retry backoff and deterministic case-insensitive HTTP header lookup.
- [x] Add Rust overflow-safe borrowed windows and strict ASCII unsigned parsing.
- [x] Add JavaScript prototype-safe grouping for special property names.
- [x] Require each new fixture to have a failing target test and a passing independent regression test.
- [x] Keep hidden evaluation files outside the model-visible workspace and bind all new fixtures/tests into the content-addressed evaluation protocol.
- [x] Upgrade the raw-model suite to 1.5.0 with 21 tasks: nine Python and three each for JavaScript, TypeScript, Go, and Rust.
- [x] Update suite-count, capability, baseline, and documentation coverage without generating any model score.

**Acceptance:** All 21 fixtures have a recognized expected target failure and a passing regression baseline in the actual language runtime. The expanded suite remains offline, dependency-free, and content-addressed. The empirical capability matrix remains unpopulated until a genuine local checkpoint completes a full run on target hardware.



### Phase 39 — Strict supervised-instruction JSONL validation

**Objective:** Remove permissive JSON interpretations from the local instruction-data boundary without changing accepted prompt/completion semantics or the SFT objective.

- [x] Reject duplicate JSON object member names at every depth, including optional metadata, rather than silently applying last-value-wins parsing.
- [x] Reject non-standard NaN/Infinity constants and decimal values that overflow to non-finite Python floats.
- [x] Enforce a quote/escape-aware 128-level JSON nesting limit and convert parser recursion into a bounded, line-scoped dataset validation error.
- [x] Preserve the existing approved-root/no-follow dataset read, exact-byte SHA-256 provenance, file/record/field/token limits, prompt-context checks, and optional split/source-path metadata.
- [x] Add regressions for duplicate required/nested keys, non-finite values, float overflow, deeply nested JSON, valid metadata, and read-only audit failure reporting.

**Acceptance:** Ambiguous or non-standard instruction records fail before tokenization/training with their source line; valid existing JSONL records retain the same parsing, tokenization, split, and training behavior.


### Phase 40 — Native training API request contracts and resume safety

**Objective:** Ensure invalid training requests fail synchronously at the local API boundary and explicit source-pretraining resume requests can never silently become fresh training runs.

- [x] Validate source-training and SFT request keys, integer ranges/types, finite learning rates, seeds, model configuration fields, and supported/available local devices before creating a worker.
- [x] Resolve approved corpus/dataset paths before device checks and preserve the existing filesystem boundary.
- [x] Require an existing checkpoint for source-pretraining resume in both the API and core trainer; keep fresh training available only when resume is false.
- [x] Add HTTP-level and direct API regressions proving invalid requests return an error without creating a training job, plus core trainer regressions for invalid inputs and missing checkpoints.

**Acceptance:** Invalid settings are rejected before asynchronous job creation; valid requests retain their existing defaults and are passed to the worker without coercive parsing; resume without a checkpoint fails explicitly and creates no checkpoint or training state.


### Phase 41 — Shared CUDA device-index preflight

**Objective:** Make explicit CUDA device selection fail at the shared device boundary before model allocation, consistently across API, CLI, training, and inference callers.

- [x] Validate `cuda:N` indices against the local PyTorch CUDA device count in `choose_device()`.
- [x] Preserve unindexed `cuda` default-device behavior and existing CPU/MPS selection.
- [x] Add regressions for valid indexed selection, out-of-range indices, and CUDA-unavailable behavior.

**Acceptance:** A valid explicit CUDA index resolves to the requested device; an out-of-range index fails with a clear `ValueError` before model allocation; unindexed `cuda` remains unchanged.


### Phase 42 — Strict native HTTP JSON request decoding

**Objective:** Prevent ambiguous or non-standard JSON request bodies from reaching native API route handlers.

- [x] Reject duplicate object member names recursively at every nesting level.
- [x] Reject JavaScript-style NaN/Infinity constants and decimal values that overflow to non-finite floats.
- [x] Enforce a quote/escape-aware 64-level structural nesting limit and normalize parser recursion into a bounded client validation error.
- [x] Add live HTTP regressions proving invalid request bodies return HTTP 400 before endpoint dispatch.

**Acceptance:** Each invalid request is rejected consistently with HTTP 400; valid JSON objects preserve existing endpoint behavior and accepted field semantics.


### Phase 43 — Strict native generation request contracts

**Objective:** Make streaming and non-streaming native generation enforce the same parameter contract before loading a checkpoint or allocating an inference model.

- [x] Validate strict integer types/ranges for output token count and top-k; reject booleans and fractional/string values.
- [x] Validate finite temperature, top-p, and repetition-penalty ranges without accepting booleans or overflowing numbers.
- [x] Validate local device selection before checkpoint loading and share the same validator between streaming and non-streaming generation.
- [x] Remove HTTP `int()`/`float()`/`str()` coercion for generation parameters and require an actual boolean stream flag.
- [x] Add direct runtime and live HTTP regressions proving invalid controls fail before a checkpoint is loaded.

**Acceptance:** Both generation modes accept the same parameter types and ranges; invalid inputs return a client validation error before model loading, while existing valid defaults and output behavior remain unchanged.


### Phase 44 — Remaining native endpoint parameter validation

**Objective:** Keep embedding, fresh evaluation, and repository curriculum request values strict from the HTTP boundary through the underlying runtime.

- [x] Validate embedding/evaluation device names and local availability before checkpoint identity checks or model loading.
- [x] Require strict integer types and supported ranges for evaluation batches and token budgets.
- [x] Remove HTTP coercion of device, evaluation, and curriculum numeric parameters.
- [x] Preserve the curriculum engine's existing `max_examples` integer bounds and add live HTTP/direct-runtime regressions.

**Acceptance:** Invalid endpoint values fail with HTTP 400 before checkpoint access or curriculum scanning; valid defaults and string/list embedding input behavior remain unchanged.


### Phase 45 — Single-flight resident checkpoint loading

**Objective:** Prevent duplicate resident Transformer construction under concurrent first requests and ensure a checkpoint replacement during model construction cannot publish stale cached state.

- [x] Add per-checkpoint load synchronization so concurrent requests share one checkpoint read and model construction.
- [x] Re-check resident cache after waiting, preserving the existing four-entry cache limit and per-device cache signature.
- [x] Recompute source/companion checkpoint identity after model construction and reject cache publication if any identity changed.
- [x] Reference-count per-path locks so idle lock entries are removed without racing queued waiters.
- [x] Add concurrent-load and atomic replacement-during-load regressions.

**Acceptance:** Concurrent requests for the same checkpoint/device construct one model; different checkpoint paths can still load independently; a checkpoint replaced during load never enters the resident cache.


### Phase 46 — Bounded native inference admission control

**Objective:** Prevent unbounded concurrent HTTP inference requests from multiplying activation/KV memory use on a local CPU/GPU while preserving responsive health/status and lightweight tokenization endpoints.

- [x] Add a bounded inference semaphore with a safe default of one in-flight model operation and a strict configurable range of 1–4.
- [x] Apply admission to generation, embeddings, and fresh evaluation; retain the slot throughout streamed generation.
- [x] Reject excess work immediately with HTTP 503 and `Retry-After: 1`, rather than queueing an unbounded set of request threads.
- [x] Release slots on success, validation failure, and stream completion; keep health/status/tokenize endpoints outside the inference gate.
- [x] Add live HTTP tests for busy responses, release-on-error, streaming lifetime, and invalid capacity settings.

**Acceptance:** Active heavy inference operations never exceed the configured limit; rejected requests receive a retryable response; each acquired slot is released exactly once on every exit path.



### Phase 47 — Coordinate native training and inference admission

**Objective:** Prevent local source pretraining or supervised fine-tuning from consuming CPU/GPU memory concurrently with HTTP inference in the same native service.

- [x] Synchronize training admission with the active inference counter so a training job starts only after all admitted inference requests finish.
- [x] Reject new inference requests immediately while a training/fine-tuning worker is alive, without queueing additional model work.
- [x] Return retryable HTTP 503 with `Retry-After: 1` when training cannot start due to active inference; preserve the existing inference-busy response for rejected model requests.
- [x] Keep health, model status, training status, and tokenization available while heavy compute is excluded.
- [x] Roll back worker state when thread startup fails and prevent inference-capacity reconfiguration while compute is active.
- [x] Add regressions for both directions of exclusion and failed worker startup.

**Acceptance:** Within one native API process, admitted inference and active training/fine-tuning never overlap; admission decisions and active-request release are serialized, rejected work starts no model operation, and every failure path leaves the service able to accept later work.



### Phase 48 — Cross-process native compute exclusion

**Objective:** Extend the existing in-process training/inference admission contract to independently launched SYNTRA CLI and API processes running under the same operating-system user.

- [x] Add a kernel-managed, non-blocking per-user process lock with POSIX file locking and Windows byte-range locking; rely on the operating system to release ownership after process termination instead of trusting stale PID/marker files.
- [x] Share one process reservation across simultaneous inference calls in the same process, preserving the configured in-process inference capacity while excluding training/fine-tuning.
- [x] Guard direct native source training, supervised fine-tuning, embeddings, generation, streaming generation, and fresh evaluation entry points.
- [x] Reserve training/fine-tuning ownership synchronously before starting an API worker; release the reservation on worker-start failure and on every worker exit.
- [x] Keep streaming ownership active until the generator is exhausted or closed; keep health, status, tokenization, and repository curriculum routes outside the heavy model-compute reservation.
- [x] Add subprocess regressions for cross-process inference/training exclusion and lock release, plus same-process concurrency, generator-close, worker-lease transfer, and private lock-file checks.

**Acceptance:** Under one OS user and the same temporary-directory configuration, a SYNTRA process cannot begin native model compute while another process owns the compute reservation; same-process inference concurrency remains governed by SYNTRA_MAX_CONCURRENT_INFERENCE; API conflicts fail immediately with HTTP 503 and Retry-After: 1; process termination releases the OS lock without stale-lock cleanup.

**Boundary:** The process reservation is per OS user and per host temporary-directory configuration. It does not coordinate different OS users, separate hosts, or processes intentionally configured with different temporary directories, and it does not claim to measure or guarantee a specific GPU/CPU memory ceiling.

### Phase 49 — Autograd-free streamed native inference

**Objective:** Ensure token-by-token native inference does not retain training graphs across generated tokens or while a client consumes a stream.

- [x] Execute the initial and incremental Transformer forward passes in explicit no-gradient contexts inside the streaming generator.
- [x] Keep the existing generator lifetime and compute-lease semantics unchanged; streaming still owns its process reservation until exhaustion or close.
- [x] Add a regression that runs streaming under an explicitly gradient-enabled caller context and asserts both forward passes observe gradients disabled.
- [ ] Measure peak memory on the same trained checkpoint and target hardware before publishing any memory-reduction figure.

**Acceptance:** Streaming generation remains functionally identical while every Transformer forward pass runs with autograd disabled, independent of the caller's ambient gradient mode; no synthetic memory measurements or model-quality claims are introduced.


### Phase 50 — Strict tokenizer-training inputs and document isolation

**Objective:** Make native BPE vocabulary learning reject malformed configuration and avoid creating merge candidates from artificial concatenations of separate source documents.

- [x] Require strict integer types and supported ranges for vocabulary size, training-byte budget, and minimum pair frequency; reject booleans, fractional values, and numeric strings.
- [x] Require an iterable of UTF-8-encodable text documents and report scalar text, non-text entries, and invalid Unicode explicitly.
- [x] Apply the aggregate byte sample budget while preserving sampled document boundaries, so merge statistics never cross separate source files.
- [x] Add focused regressions for parameter validation, malformed document input, empty iterables, and boundary-crossing merge prevention.

**Acceptance:** Valid tokenizer training remains deterministic and bounded by the configured byte sample budget; invalid input fails with a clear ValueError; every learned merge is counted within an individual sampled document, never across adjacent documents. This is a tokenizer correctness/input-contract improvement, not a model-quality claim.

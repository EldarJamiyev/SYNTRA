# SYNTRA Product Roadmap

SYNTRA is intended to become a local-first software engineering assistant with the workflow depth of tools such as Claude Code, while keeping the model, project files, and execution environment under the owner's control.

This roadmap separates shipped behavior from planned capabilities. A checked item means code exists in the repository; it does not imply that every use case is production-ready.

## Product principles

- **Local-first by default.** Project source and prompts stay on the local machine unless the owner explicitly selects a hosted provider; unpaid cloud tiers carry separate data-use risks. Local inference uses Ollama or SYNTRA's native runtime.
- **Real tools, not simulated tools.** A capability is only advertised after it has a working implementation, bounded permissions, observable results, and tests.
- **Reviewable changes.** File modifications are staged, verified where possible, previewed, and protected by source hashes and backups.
- **Explicit execution boundaries.** Reading a file, modifying a file, and executing a command are different permission levels. A coding prompt must not silently become arbitrary shell execution.
- **Honest model reporting.** A locally trained model is not described as a frontier model. Model quality is measured with task-level evaluations, not branding or training-step counts.
- **Small vertical slices.** Each phase must build, pass its relevant tests, and leave the project runnable before the next phase begins.

## Current baseline

The repository already contains:

- A minimal React chat interface with bounded, local multi-conversation history, automatic titles, search, rename/delete/resume, and migration from the original single-thread store.
- A loopback-only Node service that routes chat, network diagnostics, workspace operations, and coding requests.
- Local model routing to installed Ollama models, with a native PyTorch research runtime as fallback and an explicitly opt-in Gemini API provider.
- Workspace inspection, text search, file reading, a local import graph, relevance-ranked source context, and an optional local Ollama embedding index for semantic retrieval.
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
- [x] Constrain Ollama tool-plan output with JSON schema and fall back to deterministic context ranking when the plan is invalid or exceeds the context budget.
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

Still outstanding in this phase: model-selected execution/source-modification tool choice, durable resume of a partially verified slice (the recovery point remains the last verified slice boundary), and optional parallel execution of independent read-only tasks. The current workflow remains bounded and deterministic; these items are not claimed as delivered.

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
- [ ] Expand provider-specific coverage and add optional allowlists with safe, explicit review. Pattern scanning remains best-effort and is not a substitute for a secret manager.

**Acceptance:** Context selection is explainable, reproducible, workspace-scoped, and covered by tests for stale-index invalidation, explicit memory approval, audit redaction, workspace isolation, and best-effort secret detection.

### Phase 6 — Model quality, evaluation, and release hardening

**Objective:** Measure engineering usefulness instead of assuming it.

**Delivered evaluation slice (initial):**

- [x] Add a versioned local coding benchmark suite with three intentionally failing Node.js regression fixtures.
- [x] Run the real coding agent against disposable fixture copies using the selected local model; do not apply proposals to the active workspace.
- [x] Require the exact expected changed-file set and passing candidate verification for task completion; extra test/configuration edits invalidate the patch score.
- [x] Record provider/model identity, task latency, prompt/output tokens, patch validity, verification outcomes, task completion, and benchmark-process peak RSS in JSON reports.
- [x] Add regression coverage for suite validation, scoring, aggregation, fixture isolation, baseline checks, and report output.

**Delivered reporting slice:**

- [x] Preserve timestamped machine-readable run reports under a Git-ignored local history directory.
- [x] Add a report-history command with bounded listing and corrupt/symlink report rejection.
- [x] Compare runs only when suite version and exact task selection match; report task-level and aggregate quality/performance changes without a composite score.
- [x] Capture CPU, memory, runtime, and architecture metadata and mark latency/RSS comparisons incomparable when environments differ.
- [x] Test report validation, improvement/regression detection, task-set matching, and environment comparability.

Remaining measurement limitations: peak RSS covers only the benchmark Node.js process, not Ollama or GPU memory. Prompt token counts may be unavailable from the provider. Hardware metadata cannot prove identical background load or power/thermal conditions. The current eight-task deterministic suite broadens JavaScript task coverage but does not yet provide equivalent Python, Rust, Go or framework coverage and cannot prove parity with frontier engineering agents.

- [x] Create a versioned coding benchmark from representative local tasks and regression fixtures.
- [x] Record patch validity, candidate-test verification, task completion, end-to-end latency, prompt/output tokens, and benchmark-process RSS with each model run.
- Add provider selection and clear model capability metadata.
- [x] Add bounded Ollama retries for transient transport/server failures; classify context-window and model-memory exhaustion; reject incomplete streams and never replay partial streamed output.
- Review local-service origin checks, filesystem boundaries, command permissions, dependency handling, and Windows launch behavior.
- [x] Add an explicitly opt-in Gemini API provider with server-side API-key handling, request cancellation, provider metadata, and a separate consent gate before repository context is sent to the cloud.
- [x] Add optional local semantic retrieval using Ollama embeddings; persist only vectors, path/line metadata and source hashes; revalidate hashes before returning excerpts.
- [x] Expand orchestration to a complexity-derived maximum of eight slices, six attempts per slice and a 24-attempt total ceiling. Persist the global attempt counter and execution budget across local-service restarts.
- [x] Harden Docker check containers with init, private IPC, bounded swap and an open-file limit; retain no-network, read-only root, dropped capabilities, no-new-privileges and candidate-only write mount.
- [x] Persist a bounded, hash-validated in-flight candidate snapshot before project checks. On restart, restore it only after validating workspace identity and original-file hashes, re-run discovered checks and candidate-integrity verification, then continue or repair from the restored candidate.
- [ ] Add provider-calibrated context sizing and a cross-language empirical model capability matrix. Hosted-provider token counts remain estimates.
- Publish a reproducible release checklist and migration notes.

**Acceptance:** Release claims are backed by repeatable measurements. The installation path, test results, supported hardware, and model limitations are documented.

## Definition of done for every phase

A phase is not complete merely because the interface looks finished. It must include:

1. An implemented end-to-end user flow.
2. Unit and integration coverage for normal and failure cases.
3. A successful production build and relevant CI jobs.
4. Security review of filesystem, process, network, and model-input boundaries affected by the change.
5. Updated user documentation that describes actual behavior and remaining limitations.


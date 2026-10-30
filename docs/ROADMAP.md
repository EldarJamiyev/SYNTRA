# SYNTRA Product Roadmap

SYNTRA is intended to become a local-first software engineering assistant with the workflow depth of tools such as Claude Code, while keeping the model, project files, and execution environment under the owner's control.

This roadmap separates shipped behavior from planned capabilities. A checked item means code exists in the repository; it does not imply that every use case is production-ready.

## Product principles

- **Local-first by default.** Project source and prompts stay on the local machine. Model inference uses a locally installed Ollama model or SYNTRA's native runtime.
- **Real tools, not simulated tools.** A capability is only advertised after it has a working implementation, bounded permissions, observable results, and tests.
- **Reviewable changes.** File modifications are staged, verified where possible, previewed, and protected by source hashes and backups.
- **Explicit execution boundaries.** Reading a file, modifying a file, and executing a command are different permission levels. A coding prompt must not silently become arbitrary shell execution.
- **Honest model reporting.** A locally trained model is not described as a frontier model. Model quality is measured with task-level evaluations, not branding or training-step counts.
- **Small vertical slices.** Each phase must build, pass its relevant tests, and leave the project runnable before the next phase begins.

## Current baseline

The repository already contains:

- A minimal React chat interface with bounded, local multi-conversation history, automatic titles, search, rename/delete/resume, and migration from the original single-thread store.
- A loopback-only Node service that routes chat, network diagnostics, workspace operations, and coding requests.
- Local model routing to installed Ollama models, with a native PyTorch research runtime as fallback.
- Workspace inspection, text search, file reading, a local import graph, and relevance-ranked source context.
- Structured exact-anchor patches with path and secret-file validation.
- Candidate verification in a temporary project copy, bounded repair attempts, and hash-checked proposal application with backups and an audit log.
- Local ping, DNS, and single-port TCP diagnostics.

Known limitations are deliberate and must remain visible: there is no general-purpose terminal tool in the chat, no multi-agent execution system, and no evidence that the from-scratch native model performs at a Claude-class level. Sessions remain browser-local to one device; active-workspace labeling and restoring pending proposals after a service restart are not implemented yet.

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

- [ ] Add a visible active-workspace indicator and a predictable project-switch flow.
- [x] Add multiple named conversations, local session persistence, rename/delete, search, and resume.
- [x] Keep session state separate from individual chat messages and migrate the legacy single-thread history.
- [ ] Restore pending proposals after service restarts and let users inspect or discard them.
- [x] Preserve clear empty, loading, offline-model, and failed-request states while introducing the session experience.

**Acceptance:** A user can reopen SYNTRA, return to a named conversation, confirm the selected workspace, and recover staged proposals without guessing which project they belong to.

### Phase 3 — A real, permissioned tool runtime

**Objective:** Give the agent reliable tools instead of asking a model to imitate tools in prose.

Initial tools:
- Search files and symbols in the selected workspace.
- Read files and bounded line ranges.
- Inspect project structure, manifests, scripts, and Git status/diff.
- Propose file creation, edits, and deletion with reviewable patches.
- Run explicitly selected project checks in an isolated working copy.

Later execution tools:
- A terminal runner with working-directory restrictions, command policy, timeout and process-tree termination, output limits, and explicit approval for commands that can mutate state or access the network.
- Git operations with preview-first behavior and explicit confirmation for commits, branch changes, and pushes.

Every tool call must record its inputs, result, duration, and permission decision. Repository content is untrusted input, not an instruction source.

**Acceptance:** Tool actions are structured and testable. The UI identifies what SYNTRA is doing and why. No command is executed merely because source code or a model response asked for it.

### Phase 4 — Multi-step agent orchestration

**Objective:** Move from one patch request to a controlled plan–act–observe–repair workflow.

- Keep an explicit task plan with completed, active, blocked, and pending steps.
- Select tools based on observed repository state and previous tool results.
- Re-inspect files after each meaningful change.
- Verify after each logical implementation slice instead of postponing all checks.
- Support cancellation, resumable checkpoints, and bounded retries.
- Record model, context budget, selected files, tool calls, patch hashes, and verification results for each run.

**Acceptance:** A complex task can complete multiple verified slices, resume after interruption, and explain which evidence supports its final status. Retries are bounded and cannot bypass safety checks.

### Phase 5 — Project context and engineering memory

**Objective:** Improve context quality without silently collecting sensitive data.

- Add an incremental workspace index for symbols, imports, documentation, and test ownership.
- Prefer targeted retrieval over sending entire repositories to a model.
- Maintain task summaries and user-approved project notes.
- Keep memory local and workspace-scoped, with controls to inspect, edit, and delete it.
- Exclude credentials, private keys, ignored build artifacts, and user-configured paths.

**Acceptance:** Context selection is explainable, reproducible, workspace-scoped, and covered by tests for secret exclusion and stale-index invalidation.

### Phase 6 — Model quality, evaluation, and release hardening

**Objective:** Measure engineering usefulness instead of assuming it.

- Create a versioned coding benchmark from representative local tasks and regression fixtures.
- Measure patch validity, build/test success, task completion, latency, token usage, and memory consumption by model.
- Add provider selection and clear model capability metadata.
- Add failure recovery for provider disconnects, context exhaustion, and interrupted generations.
- Review local-service origin checks, filesystem boundaries, command permissions, dependency handling, and Windows launch behavior.
- Publish a reproducible release checklist and migration notes.

**Acceptance:** Release claims are backed by repeatable measurements. The installation path, test results, supported hardware, and model limitations are documented.

## Definition of done for every phase

A phase is not complete merely because the interface looks finished. It must include:

1. An implemented end-to-end user flow.
2. Unit and integration coverage for normal and failure cases.
3. A successful production build and relevant CI jobs.
4. Security review of filesystem, process, network, and model-input boundaries affected by the change.
5. Updated user documentation that describes actual behavior and remaining limitations.


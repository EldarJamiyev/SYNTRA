# SYNTRA Architecture

SYNTRA uses a small local stack rather than a cloud AI service.

## Runtime components

1. **React/Vite frontend** — one active conversation view and one composer, with a compact history drawer for multiple named sessions. Session state is stored in bounded browser storage, the previous single-thread history is migrated on first load, and /clear erases only the active conversation. Search, rename, delete, resume, and storage-failure fallback are handled locally. Assistant Markdown is rendered without enabling raw HTML; links are protocol-checked before rendering.
2. **Node loopback service** — validates requests, owns chat jobs, keeps the active workspace, routes ordinary messages to chat, and coordinates repository edits.
3. **Local model backend** — if a suitable local Ollama model is installed, prompts are sent to that loopback Ollama service. Otherwise the router falls back to SYNTRA's local Python/PyTorch service. The native path is a from-scratch research model, not a pretrained frontier assistant.

The Node workspace server binds to 127.0.0.1. The frontend runs at 127.0.0.1:3000; the workspace API is proxied to 127.0.0.1:3001; the native AI service normally uses 127.0.0.1:3002; Ollama listens on 127.0.0.1:11434.

## Chat request lifecycle

1. The Node service checks whether the request is a clear software-change command. Broad verbs alone are insufficient: “write a poem” and “write an email” stay on the conversation path.
2. Before model loading, the service recognises a narrow set of network diagnostic requests (ICMP ping, local DNS lookup, one TCP port check) and runs them with bounded timeouts, validated hostnames/ports, and no shell command string.
3. For normal chat, it builds a bounded prompt from the latest message and recent conversation turns, counts tokens through the selected backend, trims excess history when required, and generates a local reply. Ollama NDJSON token chunks update the active chat job, and the UI polls those partial updates; the native research backend is currently non-streaming.
4. If the user explicitly asks to modify software, SYNTRA inspects the selected workspace, builds a local import graph (including recognised hidden project-config folders such as .github and .devcontainer), and ranks relevant source files.
5. The model is asked for a constrained JSON patch with exact search anchors and replacements.
6. The agent rejects malformed JSON, duplicate edits, ambiguous anchors, path traversal, sensitive file paths, and oversized changes.
7. Candidate edits are applied only to a temporary project copy. Available build, test, lint, and type-check scripts run with bounded timeouts. Inferred native test commands are selected from project manifests and matching test files; an empty test directory or a mismatched runner is not accepted as a passing test suite. Python checks prefer a workspace virtual environment when present, and timeouts terminate spawned process trees rather than only the top-level process. .NET project files and solutions are detected explicitly, but a .NET test command is selected only when a test project or test SDK is identified.
8. Failed checks are supplied as feedback for a bounded repair attempt. The original source is not modified during planning or verification.
9. If every discovered check passes and the patch does not modify tests or verification configuration, SYNTRA rechecks source hashes and applies the change with a backup and audit record. Dependency/build manifest or lockfile changes block automatic verification because sandbox dependencies are not reinstalled; changes to test or verification configuration require review even when checks pass. In those cases, proposals remain unapplied unless the user explicitly types /apply.

Before reporting a coding task, the agent also returns a bounded per-file diff preview generated from the exact original and candidate contents. Large changed sections are explicitly marked as truncated; the preview is for inspection, not a replacement for opening the full file. The chat response presents these diffs alongside the verification result and the explicit `/apply` instruction for proposals that were not automatically applied.

A passing check proves only that the project's discovered checks passed in the temporary copy; it does not prove the absence of defects.

## Workspace selection and proposal recovery

The active workspace is held by the local Node service. The frontend reads it from `GET /local-api/workspace/status`, restores the last selected folder from device-local browser storage, and switches the backend through the origin-checked `POST /local-api/workspace/switch` route. The selected path is validated by the same workspace resolver used by file operations. Workspace selection and conversation history are stored separately.

Unverified proposals are written under `.syntra/proposals` with the workspace root, original source hash, proposed content hash, and bounded change summary. The server reconstructs the pending queue from those files rather than relying on an in-memory list. It classifies a proposal as pending only when its current source hash still equals the recorded base hash; an already-applied proposal is omitted, and a changed source is marked stale. The `/proposals` command lists the restored queue, `/apply` rechecks hashes through the existing transactional apply path, and `/discard` removes selected queue records after explicit user intent without touching project files. A new coding task is blocked while pending or stale proposals remain, preventing a later request from silently replacing an earlier review queue.

The browser remembers the selected workspace path locally. That value is untrusted input and must pass workspace resolution before it can become active. The server remains loopback-only; switching projects does not expand its network exposure.

## Permissioned tool runtime

`local/tool-runtime.mjs` is the shared registry for local workspace operations. `GET /local-api/tools` returns the 12 tool definitions and their input schemas, permission classes, and approval requirements. `POST /local-api/tools/execute` validates calls against those definitions; unknown tools, undeclared fields, malformed IDs, out-of-range line windows, and invalid types are rejected.

The current registry contains:

- Read-only tools: `workspace.inspect`, `workspace.search`, `workspace.read`, `workspace.proposals.list`, `git.status`, and `git.diff-summary`.
- Local SYNTRA-state writes: `workspace.plan` and `workspace.proposals.stage`. Staging writes a hash-bound proposal under `.syntra`; it does not alter the project source.
- Approval-gated operations: `workspace.check` (executes project-defined checks), `workspace.proposals.apply`, `workspace.proposals.apply-many` (transactional source changes), and `workspace.proposals.discard` (irreversible deletion of proposal records).

Every accepted tool invocation receives a UUID run ID and emits started/completed/failed or approval-required records to `.syntra/actions.jsonl`. Events contain bounded input/output summaries, permission and approval decisions, duration, and errors; source contents and full file excerpts are deliberately excluded. An initial audit-write failure prevents the operation from running. A final audit-write failure is returned as an explicit warning because an already-completed operation cannot safely be represented as not having happened. Calls for one workspace are serialized to prevent overlapping proposal, check, and audit operations.

The coding agent now routes project inspection, isolated candidate checks, proposal staging, and transactional apply through this registry. Its orchestration is still deterministic in the server/agent code; the local model does not yet choose arbitrary tools dynamically. Git inspection uses fixed `execFile` arguments, disables optional index writes and external diff, and does not contact remotes. There is no general-purpose terminal tool in this phase.

## Model and context handling

Ollama discovery excludes cloud-tagged models and rejects weights files larger than 70% of physical RAM to preserve working memory for the OS and inference context/cache. Stored weight size is a heuristic, not a guarantee of runtime fit. Model detection is briefly cached between adjacent chat stages; injected/test fetch implementations bypass the cache for deterministic testing. The Ollama router uses an 8,192-token default context, configurable with SYNTRA_LOCAL_CONTEXT within the supported limits.

The native Transformer has its own tokenizer, context configuration, training loop, checkpointing, and inference implementation. Its weights start randomly initialized. Training it from scratch on a small project is useful for runtime research, but does not confer reliable general knowledge or frontier-model coding ability.

## Reliability and security boundaries

- File access is rooted in a selected absolute workspace, with path traversal and symlink path components blocked. The reserved .syntra state directory and its proposals/backups/plans subdirectories must be real directories inside that workspace, and the audit log must be a regular file; none may be symlinks to external paths.
- Likely credential files and secret paths are excluded from workspace read/edit operations.
- Agent patches are restricted to recognized text/source/configuration files and validated against exact file hashes. Multi-file apply preflights every proposal, saves backups together, rechecks each source immediately before writing, and attempts a guarded rollback if a later write fails.
- Candidate checks execute in a temporary project copy. Build/test scripts are project code and should still be reviewed when working in an untrusted repository.
- The services are intended for local use, not direct internet exposure. Automated Node tests launch an isolated loopback server on a temporary port to check request validation and origin rejection. A static hosted copy cannot access the user's local model or workspace agent by itself.

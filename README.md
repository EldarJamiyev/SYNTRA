# SYNTRA — Simple local AI chat

SYNTRA is a deliberately minimal chat interface backed by a local model and a repository-aware coding agent. It keeps the conversation and message composer in focus instead of filling the screen with model controls. Multiple named conversations are saved locally on the current device; you can start, search, resume, rename, or delete a conversation from the history drawer. Session storage is bounded, invalid data is discarded safely, and the previous single-thread history is migrated automatically. Each conversation keeps its latest 60 messages, and prompts are saved immediately so a refresh during generation does not erase what you asked. Type /clear to clear the active conversation without deleting the other sessions.

See [the staged product roadmap](docs/ROADMAP.md) for the planned path from the current local coding agent to a more complete, permissioned engineering assistant.

## Windows launch

Requirements: Node.js 20+ and Git. The test suite includes an HTTP smoke test for the local API's origin guards and request validation. Python 3.11+ is required only for the native model fallback. A compatible local model in Ollama is preferred automatically.

In your existing SYNTRA repository folder, run:

~~~powershell
git pull
~~~

Then double-click **Start-SYNTRA.bat**. Open http://127.0.0.1:3000 and keep the service windows open. On future runs, double-click the same file again.

The launcher checks the page signature, detects stale app processes on the chat port, and installs newly required frontend dependencies instead of blindly accepting an unrelated page.

## Chat and coding behavior

The chat understands natural requests such as “Could you ping example.com?” and “Can you check port 443 on example.com?”, and can also run bounded local network checks without a trained model: `ping example.com`, `DNS lookup for example.com`, and `check port 443 on example.com`. Slash forms (`/ping`, `/dns`, `/port`) are available for explicit requests. Type /help in the chat (or ask “What can you do?”) for available commands; it works even when no model is installed. Type /status to see whether the local model backend responds. Ping is limited to four packets, TCP checks to one host/port, and targets are validated before any system utility or socket is called. These checks run locally and do not use a hosted service.


Normal requests such as writing an email, explaining a concept, or brainstorming stay in chat. Explicit software tasks—such as fixing an API endpoint, updating a component, or changing the project UI—are routed to the repository agent.

Assistant replies support Markdown, GitHub-flavored tables and lists, inline code, fenced code blocks, and safe links. Local Ollama replies stream into the conversation while the model generates; the native research model displays its response when generation completes. User prompts are rendered as literal text so HTML-like snippets are not silently removed. Raw assistant HTML is not rendered, and external links open with appropriate isolation.

The top bar shows the active local project folder. Use its folder control to switch projects; the selection is saved on this device and restored when the frontend reconnects to the local service. Staged changes are persisted under the selected project's `.syntra/proposals` directory. Type `/proposals` after a service restart to reconstruct the pending queue, `/apply` only after reviewing the saved proposal and its original diff in chat, or `/discard` to remove pending/stale proposal records without editing project files. Proposals whose source hash no longer matches are marked stale and cannot be applied. SYNTRA blocks a new coding task until the existing proposal queue is resolved.

Larger coding tasks are planned as up to three ordered implementation slices, each with an acceptance criterion. SYNTRA applies each slice to one temporary candidate workspace, runs the discovered project checks, validates the changed-file hashes, then re-inspects the candidate before generating the next slice. Failed checks trigger bounded repair attempts against freshly inspected candidate context. The final proposals combine all slice changes against the original project-file hashes. If any slice is incomplete, the combined result stays unverified and is not automatically applied.

After each successfully verified non-final slice, SYNTRA can persist a bounded candidate checkpoint beside the run record as `.syntra/runs/<run-id>.candidate.json`. It contains the cumulative changed-file contents, original-file hashes, slice plan/progression, and verification summary; it is limited to 24 files, 128 KB per file, and 512 KB total. On `/resume`, SYNTRA rechecks every original-file hash, reconstructs a fresh temporary candidate, re-inspects it, and continues at the next pending slice. If interruption happens during an unverified slice—or after the final slice passes but before proposal staging—the in-flight/final slice is regenerated from the last verified checkpoint; if no verified slice exists, the task restarts from the overall goal. SYNTRA does not persist an all-slices-complete checkpoint, so a resumed task never skips final candidate checks based only on stored status. A stale or invalid checkpoint is rejected rather than applied, and candidate checkpoint persistence failure is reported rather than presented as restartable.

Coding runs are persisted in each workspace's `.syntra/runs` directory. Open the workspace control to inspect recent tasks; reopening it refreshes the run state after a local service restart. Stop cancels supported local inference and candidate checks, while transactional file application is allowed to finish once it has started. Single-slice saved patches are revalidated against their original source hashes before they can be reused. Multi-slice runs resume from the last verified slice boundary, not from an arbitrary in-flight model or tool operation. Run records retain bounded plans, context, tool calls, verification outcomes, and model/context metadata locally.

For coding tasks, SYNTRA maps the project files (including recognised CI/configuration folders such as .github and .devcontainer) and imports, asks the selected local model for a structured patch, validates exact source anchors and paths, and runs available project checks in a temporary copy. Inferred test runs are language-aware and require matching test files. Rust and Go use their native test commands; Python selects pytest or unittest from detected project configuration and prefers an existing .venv/venv interpreter; timed-out checks terminate their process tree. .NET requires a recognised test project. When several ecosystems exist without a configured test command, SYNTRA refuses to call a partial run a full pass. Changes are auto-applied only when discovered checks pass and the patch does not modify tests or verification configuration. Changes to dependency/build manifests and lockfiles are not auto-verified against stale sandbox dependencies; those proposals remain unapplied for explicit review. Test/CI/config changes also require review and explicit /apply. Check selection is based on a read-only discovery tool run against the temporary candidate workspace: only build/test/lint/type-check runners that SYNTRA can actually resolve are executed. Afterward, the agent re-reads every changed candidate file and compares its hash with the validated patch. A project script that mutates or removes a proposed source file causes verification to fail instead of producing a false pass. Bounded Git baseline, check-discovery, and candidate-integrity observations are stored in the local agent run record; recent runs show the number of evidence points.
The agent keeps backups and audit records under .syntra. Coding-task replies include bounded per-file before/after diff previews and explain when a preview is truncated, so staged changes can be reviewed before typing `/apply`. The state directory and its proposal/backup/plan subdirectories plus audit log cannot be symlinks to another location, saved proposals are checked against their content hashes before applying, and multi-file changes are preflighted and applied as a transaction with a shared backup directory.

Before generating a code patch, SYNTRA can ask its local model to choose up to four read-only context retrieval calls. The model may request `workspace.search`, `workspace.read`, `workspace.symbols.search`, or `workspace.memory.search`; file reads must target paths already present in the safe source graph and are limited to 200 lines. Memory results are drawn only from notes explicitly saved by the user for the selected workspace. They are treated as untrusted project context, not privileged instructions. Symbol results include validated local importers/imports, documentation links, and heuristic test-owner relationships. Each call is validated and audited by the tool runtime. Invalid plans fall back to deterministic file ranking. The model cannot use this retrieval step to run checks, modify files, apply proposals, execute shell commands, or access the network.

SYNTRA's local tool runtime exposes a structured catalogue at `GET /local-api/tools` and accepts validated calls at `POST /local-api/tools/execute`. It currently provides 21 tools for workspace inspection, incremental source-symbol indexing/search, user-controlled project-memory list/search/write operations, file search/read, planning, proposal review/application/discard, approved project checks, check-runner discovery, and Git status/diff summaries. The workspace context index is local to each workspace, stores metadata and hashes rather than source/documentation contents, and reuses parsed declarations and references for unchanged files. It includes bounded import edges, Markdown/MDX/reStructuredText links, and heuristic test ownership. Each call has a run ID, a permission class, bounded input/output summaries, duration, result state, and an audit record in `.syntra/actions.jsonl`; raw file contents are not copied into tool audit events. The runtime serializes calls within a workspace. Ad-hoc project-check tool calls, applying source changes, discarding proposals, and adding, editing, or deleting project-memory notes require approval. A user-requested coding task authorizes SYNTRA's built-in checks in the temporary candidate copy and the existing verified-patch apply policy. That candidate copy is a separate working directory, not a hardened OS/container sandbox: project scripts still run with the current user's operating-system permissions and may access the network. Only run coding tasks against projects whose scripts you trust. Git inspection uses fixed, non-interactive arguments and does not fetch from remotes. This is a permissioned tool foundation, not a general-purpose terminal or arbitrary command runner.

## Reproducible coding benchmark

SYNTRA includes a versioned local coding benchmark (`coding-v1`) with three intentionally failing engineering fixtures: slug normalization, bounded duration parsing, and case-insensitive HTTP header merging. Run it while the local Vite/API stack is running and a local Ollama model or trained native checkpoint is available:

```powershell
npm run dev
# In a second terminal:
npm run benchmark:coding
```

Run one task or preserve its disposable workspace for inspection:

```powershell
npm run benchmark:coding -- --task slugify-normalization
npm run benchmark:coding -- --keep-workspaces
npm run benchmark:coding -- --timeout-seconds 1200 --output .\benchmark-results\coding-v1.json
```

The runner creates a fresh temporary workspace for each task and sends the real task through SYNTRA's local chat/coding-agent API. Its pass rule requires a schema-valid patch, passing agent verification, an independent post-run test, a changed target source file, and unchanged protected fixture tests/configuration. It records the actual provider/model, full-run prompt/output token totals, task latency, repair attempts, and per-task benchmark-runner peak RSS. Prompt-token exactness is reported separately when the provider returns token counts versus requiring tokenizer fallback.

Memory figures are deliberately scoped: the runner's RSS belongs to the benchmark-controller process, not the model server. For Ollama, loaded model size and VRAM allocation are sampled from its local `/api/ps` endpoint when available; this is not total Ollama process memory. Native-model process memory is currently reported as unavailable. The report is written to `benchmarks/results/` by default and contains the real outcomes—SYNTRA does not fill in missing model results with synthetic scores.

The benchmark uses the currently selected local provider; it does not switch models automatically. To compare models, run the same benchmark version separately under each configured local model and compare the JSON reports. The three tasks are small regression fixtures, not a broad industry benchmark or evidence of frontier-model equivalence. See [the benchmark methodology](docs/CODING_BENCHMARK.md) for metric definitions and limitations.

## Local model

SYNTRA prefers an installed local Ollama model and does not call a hosted AI API. Its model selection excludes cloud-only models and rejects weights larger than 70% of physical RAM, leaving room for the OS and inference context/cache. That is a conservative file-size filter, not a guarantee that every remaining model will fit in practice. Model discovery is briefly cached between the status, token-count, and generation stages to avoid repeating the same probe. Inference speed and answer quality still depend on your hardware and the installed model.

If Ollama is unavailable, SYNTRA can use its own local PyTorch runtime. The native model starts with randomly initialized weights: training it from scratch on a small local corpus does not make it comparable to Claude or other frontier models.

## Important boundary

This app is designed to run locally at http://127.0.0.1:3000; the Node service binds to loopback and is not a hosted AI endpoint. A static GitHub Pages copy cannot execute your local model or workspace agent. Use the local launcher for real chat and code changes.

## Development-server startup

The Vite development configuration starts the Node workspace API automatically. It first checks `http://127.0.0.1:3001/local-api/health`, reuses a healthy SYNTRA API when one is already running, or starts `local/server.mjs` and waits for its health endpoint before Vite finishes starting. The API child started by Vite is stopped when the development server closes. Set `SYNTRA_AUTOSTART_LOCAL_API=0` only when managing the API process yourself.

The workspace API remains bound to loopback. Browser POST requests are restricted to the built-in local origins, exact same-origin requests forwarded by the Vite proxy, or exact additional origins listed in `SYNTRA_ALLOWED_ORIGINS` (comma-separated; no wildcard values).

**Hosted-preview boundary:** this starts the Node workspace API inside the same runtime as Vite; it does not connect that runtime to files, Ollama, or a Python model running on your personal computer. A hosted preview can use only files and model services available inside its own runtime. The native Python model remains a separate service and must be installed, started, and supplied with a valid checkpoint independently.

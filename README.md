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

For coding tasks, SYNTRA maps the project files (including recognised CI/configuration folders such as .github and .devcontainer) and imports, asks the selected local model for a structured patch, validates exact source anchors and paths, and runs available project checks in a temporary copy. Inferred test runs are language-aware and require matching test files. Rust and Go use their native test commands; Python selects pytest or unittest from detected project configuration and prefers an existing .venv/venv interpreter; timed-out checks terminate their process tree. .NET requires a recognised test project. When several ecosystems exist without a configured test command, SYNTRA refuses to call a partial run a full pass. Changes are auto-applied only when discovered checks pass and the patch does not modify tests or verification configuration. Changes to dependency/build manifests and lockfiles are not auto-verified against stale sandbox dependencies; those proposals remain unapplied for explicit review. Test/CI/config changes also require review and explicit /apply. The agent keeps backups and audit records under .syntra. Coding-task replies include bounded per-file before/after diff previews and explain when a preview is truncated, so staged changes can be reviewed before typing `/apply`. The state directory and its proposal/backup/plan subdirectories plus audit log cannot be symlinks to another location, saved proposals are checked against their content hashes before applying, and multi-file changes are preflighted and applied as a transaction with a shared backup directory.

SYNTRA's local tool runtime exposes a structured catalogue at `GET /local-api/tools` and accepts validated calls at `POST /local-api/tools/execute`. It currently provides 12 tools for workspace inspection/search/read, planning, proposal listing/staging/application/discard, approved project checks, and read-only Git status/diff summaries. Each call has a run ID, a permission class, bounded input/output summaries, duration, result state, and an audit record in `.syntra/actions.jsonl`; raw file contents are not copied into tool audit events. The runtime serializes calls within a workspace. Ad-hoc project-check tool calls, applying source changes, and discarding proposals require approval. A user-requested coding task authorizes SYNTRA's built-in checks in the temporary candidate copy and the existing verified-patch apply policy. That candidate copy is a separate working directory, not a hardened OS/container sandbox: project scripts still run with the current user's operating-system permissions and may access the network. Only run coding tasks against projects whose scripts you trust. Git inspection uses fixed, non-interactive arguments and does not fetch from remotes. This is a permissioned tool foundation, not a general-purpose terminal or arbitrary command runner.

## Local model

SYNTRA prefers an installed local Ollama model and does not call a hosted AI API. Its model selection excludes cloud-only models and rejects weights larger than 70% of physical RAM, leaving room for the OS and inference context/cache. That is a conservative file-size filter, not a guarantee that every remaining model will fit in practice. Model discovery is briefly cached between the status, token-count, and generation stages to avoid repeating the same probe. Inference speed and answer quality still depend on your hardware and the installed model.

If Ollama is unavailable, SYNTRA can use its own local PyTorch runtime. The native model starts with randomly initialized weights: training it from scratch on a small local corpus does not make it comparable to Claude or other frontier models.

## Important boundary

This app is designed to run locally at http://127.0.0.1:3000; the Node service binds to loopback and is not a hosted AI endpoint. A static GitHub Pages copy cannot execute your local model or workspace agent. Use the local launcher for real chat and code changes.

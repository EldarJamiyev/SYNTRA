<p align="center">
  <img src="docs/assets/syntra-hero.svg" alt="SYNTRA — Local intelligence. Real engineering." width="100%">
</p>

<p align="center">
  <a href="https://github.com/EldarJamiyev/SYNTRA/actions/workflows/checks.yml"><img src="https://img.shields.io/github/actions/workflow/status/EldarJamiyev/SYNTRA/checks.yml?branch=main&style=for-the-badge&label=BUILD" alt="Build status"></a>
  <img src="https://img.shields.io/badge/AI-LOCAL--FIRST-79DCCB?style=for-the-badge" alt="Local-first AI">
  <img src="https://img.shields.io/badge/Model-Ollama-91B7FF?style=for-the-badge" alt="Ollama">
  <img src="https://img.shields.io/badge/Frontend-React%2018-9B9BE8?style=for-the-badge" alt="React 18">
  <img src="https://img.shields.io/badge/Runtime-Node.js%2020%2B-C4A7FF?style=for-the-badge" alt="Node.js 20 or newer">
  <a href="https://github.com/EldarJamiyev/SYNTRA/stargazers"><img src="https://img.shields.io/github/stars/EldarJamiyev/SYNTRA?style=for-the-badge&color=E8C77A" alt="GitHub stars"></a>
</p>

<p align="center">
  <strong>A private AI workspace for understanding, changing and verifying real software projects.</strong>
  <br>
  One focused conversation. A local model. A repository-aware engineering agent.
</p>

<p align="center">
  <a href="#-experience">Experience</a> ·
  <a href="#-engineering-agent">Engineering agent</a> ·
  <a href="#-architecture">Architecture</a> ·
  <a href="#-safety--control">Safety &amp; control</a> ·
  <a href="#-get-started">Get started</a> ·
  <a href="#-benchmarks">Benchmarks</a>
</p>

---

## The idea

SYNTRA brings local AI chat and practical software engineering into one clean workspace. Ask questions naturally, explore a codebase, request a change, inspect the proposed diff, and verify it against the project before applying anything.

It is designed to keep the engineering process visible: **what the agent inspected, what it changed, which checks ran, and what still needs your approval.**

<table>
  <tr>
    <td width="25%" align="center">
      <h3>01</h3>
      <strong>Understand</strong>
      <br><sub>Conversations and project context</sub>
    </td>
    <td width="25%" align="center">
      <h3>02</h3>
      <strong>Investigate</strong>
      <br><sub>Search, symbols, memory and evidence</sub>
    </td>
    <td width="25%" align="center">
      <h3>03</h3>
      <strong>Engineer</strong>
      <br><sub>Structured patches and bounded repair</sub>
    </td>
    <td width="25%" align="center">
      <h3>04</h3>
      <strong>Verify</strong>
      <br><sub>Isolated checks and human approval</sub>
    </td>
  </tr>
</table>

## ✦ Experience

SYNTRA keeps the interface deliberately restrained. The conversation and composer stay central; project operations appear when they are relevant instead of competing for attention.

| | Capability | What it does |
|:--:|---|---|
| ◈ | **Focused AI chat** | Natural conversation, Markdown, code blocks, tables, safe links and streamed local Ollama responses. |
| ◷ | **Conversation history** | Multiple named sessions stored on the current device; search, resume, rename and delete conversations. |
| ⌘ | **Workspace awareness** | Switch the active local project and keep proposals, run history and engineering state with that workspace. |
| ⌁ | **Quick diagnostics** | Bounded ping, DNS lookup and TCP port checks, including natural-language requests and slash commands. |
| ⌗ | **Project memory** | Save and search user-authored architecture notes, decisions, conventions and testing knowledge. |
| ◉ | **Reviewable changes** | Inspect per-file diffs and hash-bound proposals before explicitly applying changes. |

### One conversation, multiple kinds of work

- Ask a normal question, draft text, explain a concept or brainstorm without starting a coding task.
- Ask for a network check such as `ping example.com`, `DNS lookup for example.com` or `check port 443 on example.com`.
- Request an explicit software change to invoke the repository-aware coding workflow.
- Use `/help`, `/status`, `/clear`, `/proposals`, `/apply`, `/discard` and `/resume` for direct control.

> Network diagnostics are local and bounded. Ping is limited to four packets; TCP checks target one validated host and port. They do not require a coding model or a hosted service.

## ◈ Engineering agent

SYNTRA is more than a chat wrapper. For explicit coding tasks, it builds a bounded view of the project, retrieves relevant evidence, asks the local model for a structured patch, and checks the result in a temporary candidate workspace.

<p align="center">
  <img src="docs/assets/syntra-agent-loop.svg" alt="SYNTRA's seven-stage engineering workflow: understand, inspect, retrieve, generate, review, verify and approve." width="100%">
</p>

### Repository understanding

- Builds a workspace-local file inventory and import/context graph.
- Uses TypeScript compiler AST analysis for JavaScript and TypeScript declarations, imports, statically imported calls and selected type references.
- Resolves supported local relationships for JavaScript/TypeScript and Python, with bounded heuristics for Go and Rust.
- Connects documentation references and heuristic test ownership to source context.
- Refreshes hashes, reuses unchanged index entries, and rejects stale source evidence.

### Evidence-led retrieval

The agent can select bounded read-only retrieval calls, then use earlier results to guide follow-up investigation.

| Retrieval path | Purpose |
|---|---|
| **Lexical search** | Find matching terms in current project files. |
| **Symbol search** | Locate declarations, imports and related source files. |
| **Semantic source search** | Retrieve relevant source chunks using local Ollama embeddings when configured. |
| **Project memory** | Search notes explicitly saved by the user. |
| **Workspace reads** | Inspect bounded line ranges from already discovered, safe project paths. |

Semantic retrieval uses overlapping source chunks, validates the indexed file hash before returning excerpts, and caches local vectors without storing source text. When local embeddings are unavailable, it falls back to lexical search.

### Structured implementation

- Generates a constrained JSON patch with exact source anchors and relative paths.
- Validates paths, file hashes and unique match locations before writing to a candidate.
- Can plan up to **three ordered implementation slices**, each with an acceptance criterion.
- Re-inspects the candidate between slices and after failed checks.
- Supports bounded repair attempts using fresh check output and repository evidence.
- Persists verified slice checkpoints so an interrupted multi-slice run can resume from the last verified boundary.

### Review and verification

A second-pass reviewer can identify concrete blocking and advisory issues. It may use a separately configured local reviewer model; without one, a second pass by the same model is **not** independent-model evaluation.

Candidate verification discovers supported build, test, lint and type-check commands and runs them in a temporary project copy. Changed candidate files are re-read after checks to detect scripts that modify or remove proposed source files.

The final changes are staged as hash-bound proposals. Review the diff first; applying changes is an explicit user action.

## ◫ Architecture

<p align="center">
  <img src="docs/assets/syntra-architecture.svg" alt="SYNTRA local architecture showing React and Vite, the loopback Node service, local model runtime, workspace intelligence, candidate verifier and proposal queue." width="100%">
</p>

| Layer | Responsibility | Technology |
|---|---|---|
| **Conversation UI** | Chat, history, workspace controls and proposal review | React 18 · TypeScript · Vite |
| **Local orchestration** | Request routing, job state, workspace selection and tool permissions | Node.js 20+ |
| **Model runtime** | Local inference for chat, patch generation and review | Ollama by default |
| **Research runtime** | Experimental native model implementation | Python 3.12+ · PyTorch |
| **Project intelligence** | File graph, symbol index, semantic retrieval and saved memory | Workspace-local files and metadata |
| **Verification** | Candidate checks and whole-workspace evidence | Docker isolation by default |

### Local service map

| Service | Default address | Role |
|---|---|---|
| Frontend | `127.0.0.1:3000` | Local chat interface |
| Workspace API | `127.0.0.1:3001` | Jobs, tools and project operations |
| Native AI service | `127.0.0.1:3002` | Optional Python research backend |
| Ollama | `127.0.0.1:11434` | Local model inference |

The browser frontend communicates with the loopback Node service. The service coordinates local model inference and workspace operations; SYNTRA does not require a hosted AI API.

## ⛨ Safety & control

Local-first is an architectural boundary, not just a marketing label.

| Protection | How SYNTRA handles it |
|---|---|
| **Local inference** | Uses a locally installed pretrained Ollama model by default; no hosted AI API is required. |
| **Loopback services** | The workspace API binds to `127.0.0.1`; browser requests are checked against allowed local origins. |
| **Permissioned tools** | A fixed tool registry validates inputs, permission classes and approval requirements. It is not a general-purpose terminal. |
| **Read-only investigation** | Model-selected context retrieval cannot write files, run commands, apply proposals or access the network. |
| **Candidate isolation** | Project checks use an offline Docker container by default, with a read-only container root, dropped capabilities, bounded resources and only the candidate workspace mounted. |
| **Fail-closed verification** | Docker absence blocks isolated checks. Direct host execution requires the explicit `SYNTRA_ALLOW_UNSANDBOXED_CHECKS=true` opt-in. |
| **Proposal integrity** | Exact base hashes, file anchors, backups and transaction-aware apply checks protect source changes. |
| **Local audit** | Bounded tool events, plans, run records and proposals are stored under the selected workspace's `.syntra` directory. |

<details>
  <summary><strong>Important operational boundaries</strong></summary>

  - Docker verification has no network access. Dependencies must already exist in the candidate workspace or in a prepared custom image.
  - Enabling unsandboxed checks runs project code with the current user's operating-system permissions. Use that opt-in only for repositories whose scripts you trust.
  - The agent does not provide an arbitrary shell or unrestricted network tool.
  - Project memory stores only notes explicitly entered by the user. It does not automatically ingest chat history, source code or documentation.
  - Credential detection and secret-path exclusions are best-effort protections, not a guarantee that every secret can be identified.
  - The local `.syntra` directory contains recovery and proposal state. Removing it removes that workspace's local history.
</details>

## 🚀 Get started

### Windows — one-click launch

**Requirements**

- Windows 10/11
- Node.js 20 or newer
- Git
- Ollama with a compatible pretrained local model

Python 3.12+ is needed only for the experimental native research backend.

**Launch an existing checkout**

1. Install and start Ollama, then make sure your preferred local model is available.
2. Open your SYNTRA repository folder and pull the latest changes:

   ```powershell
   git pull
   ```

3. Double-click **`Start-SYNTRA.bat`**.
4. Open **[http://127.0.0.1:3000](http://127.0.0.1:3000)**.

Keep the service windows open while using the application. On later runs, double-click the launcher again.

The launcher checks the expected SYNTRA page signature, detects stale processes on the chat port, and installs missing frontend dependencies instead of blindly accepting an unrelated page.

### Manual development setup

Clone the repository and install the frontend/runtime dependencies:

```bash
git clone https://github.com/EldarJamiyev/SYNTRA.git
cd SYNTRA
npm ci
```

Start the local workspace API in one terminal:

```bash
npm run server
```

Start the Vite frontend in a second terminal:

```bash
npm run dev
```

Open **[http://127.0.0.1:3000](http://127.0.0.1:3000)**.

The Vite configuration can start the local API automatically. Set `SYNTRA_AUTOSTART_LOCAL_API=0` only when you manage that process yourself.

### Local model configuration

| Variable | Purpose |
|---|---|
| `SYNTRA_LOCAL_MODEL` | Select an exact installed local Ollama model. |
| `SYNTRA_REVIEW_MODEL` | Optionally select a separate local reviewer model. |
| `SYNTRA_EMBED_MODEL` | Enable local Ollama embeddings for semantic source and hybrid project-memory search. |
| `SYNTRA_OLLAMA_RETRIES` | Configure bounded retries for transient Ollama failures; `0` disables retries. |
| `SYNTRA_MODEL_PROVIDER=native` | Explicitly select the experimental native research backend instead of the production Ollama path. |
| `SYNTRA_ALLOW_UNSANDBOXED_CHECKS=true` | Explicitly opt into running project checks directly on the host. Avoid this for untrusted repositories. |

The native PyTorch Transformer starts from randomly initialized weights. It is a research implementation for experimenting with tokenizer, training and checkpoint mechanics—not a pretrained coding model or an automatic Ollama fallback.

## ◫ Benchmarks

SYNTRA has a versioned local coding benchmark suite. The suite currently contains **eight intentionally failing fixtures** across JavaScript, TypeScript, Python, configuration and multi-file API work.

<p align="center">
  <img src="docs/assets/syntra-benchmark.svg" alt="SYNTRA benchmark coverage: eight fixtures, expected-failure baseline validation, exact patch paths and passing candidate verification." width="100%">
</p>

### Run the benchmark

Start a supported local model, then run:

```bash
# List benchmark tasks
npm run benchmark -- --list

# Run the full suite
npm run benchmark

# Run one task
npm run benchmark -- --task utility-clamp

# Save a report to a chosen path
npm run benchmark -- --output reports/local-model-run.json

# Review saved runs and compare model results
npm run benchmark:reports -- history
```

A task is complete only when all three conditions hold:

1. Its baseline fails for the declared diagnostic before the model is invoked.
2. The changed-file set exactly matches the task's expected paths.
3. Candidate verification passes.

Reports include model/provider identity, task-level outcomes, latency, token measurements and benchmark-process RSS. RSS measures the SYNTRA benchmark Node.js process—not the separate Ollama daemon, GPU memory or total system memory.

See **[Benchmark methodology and limitations](docs/BENCHMARKS.md)** for the full task list and measurement details.

## ◈ Project map

```text
SYNTRA/
├── src/                    # React + TypeScript application
├── local/                  # Local API, model routing and agent runtime
├── syntra_ai/              # Experimental Python/PyTorch research backend
├── scripts/                # Launch, test and benchmark tooling
├── benchmarks/
│   ├── fixtures/            # Disposable coding tasks
│   └── suite.v1.json        # Versioned benchmark suite
├── tests/                  # Node, frontend and integration tests
├── tests/python/            # Native backend tests
├── docs/
│   ├── assets/              # README and documentation visuals
│   ├── ARCHITECTURE.md
│   ├── BENCHMARKS.md
│   └── ROADMAP.md
├── Start-SYNTRA.bat
└── package.json
```

## ◇ Documentation

| Resource | What you'll find |
|---|---|
| **[Architecture](docs/ARCHITECTURE.md)** | Runtime components, request lifecycle, tool permissions, local state and security boundaries. |
| **[Benchmarks](docs/BENCHMARKS.md)** | Fixture definitions, scoring, report history, comparisons and measurement limits. |
| **[Roadmap](docs/ROADMAP.md)** | Delivered capabilities, remaining engineering work and phase acceptance criteria. |

---

<p align="center">
  <strong>Built for local control. Designed for deliberate engineering.</strong>
  <br>
  <sub>SYNTRA · Understand the project. Make the change. Verify the evidence.</sub>
</p>

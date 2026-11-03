<div align="center">
  <img src="docs/assets/syntra-hero.svg" alt="SYNTRA — Local intelligence. Real engineering." width="100%" />

  <p><strong>A local-first AI workspace for conversation, repository understanding, and carefully controlled code changes.</strong></p>

  <p>
    <a href="https://github.com/EldarJamiyev/SYNTRA"><img src="https://img.shields.io/badge/PROJECT-SYNTRA-7BE4CF?style=for-the-badge&labelColor=101827" alt="SYNTRA project" /></a>
    <img src="https://img.shields.io/badge/INTERFACE-React%20%2B%20TypeScript-8AB8FF?style=for-the-badge&labelColor=101827" alt="React and TypeScript" />
    <img src="https://img.shields.io/badge/INFERENCE-Standalone%20PyTorch%20Brain-B69BFF?style=for-the-badge&labelColor=101827" alt="Standalone local PyTorch brain" />
    <a href="LICENSE"><img src="https://img.shields.io/badge/LICENSE-MIT-7BE4CF?style=for-the-badge&labelColor=101827" alt="MIT license" /></a>
  </p>

  <p>
    <a href="#-why-syntra">Why SYNTRA</a> &nbsp;·&nbsp;
    <a href="#-capabilities">Capabilities</a> &nbsp;·&nbsp;
    <a href="#-how-the-agent-works">Agent workflow</a> &nbsp;·&nbsp;
    <a href="#-get-started">Get started</a> &nbsp;·&nbsp;
    <a href="#-documentation">Documentation</a>
  </p>
</div>

---

## The idea

SYNTRA brings a focused AI chat experience and a repository-aware coding agent into one application. It is designed to run on your machine, work with a project you select, gather relevant evidence, and make code changes through a reviewable workflow.

**The principle is simple: local control, real tools, observable work, and no pretending.**

It is an evolving engineering project—not a claim of frontier-model intelligence or a finished autonomous software engineer.

## ✦ Why SYNTRA

<table>
  <tr>
    <td width="50%" valign="top">
      <h3>⌂ Local by design</h3>
      Run inference through SYNTRA's own PyTorch Transformer and locally saved checkpoint. The workspace service launches the Python inference process directly; no external model daemon or hosted coding endpoint is used.
    </td>
    <td width="50%" valign="top">
      <h3>⌘ Understands the repository</h3>
      Inspects project structure, searches source, follows imports and symbols, and retrieves task-specific context before proposing changes.
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <h3>◇ Changes you can inspect</h3>
      Produces structured patches, reviews proposed edits, checks candidates, and records proposal and run history locally.
    </td>
    <td width="50%" valign="top">
      <h3>⛨ Bounded permissions</h3>
      Separates read-only investigation from operations that change files or run project checks. There is no general-purpose terminal tool in the chat.
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <h3>◎ More than code generation</h3>
      Supports ordinary conversation and bounded local network diagnostics, including ping, DNS lookup, and a single TCP port check.
    </td>
    <td width="50%" valign="top">
      <h3>▤ Built for continuity</h3>
      Keeps named conversations on the device and preserves workspace run records and recoverable proposals across local-service restarts.
    </td>
  </tr>
</table>

## ✧ Capabilities

### A focused chat workspace

SYNTRA keeps the conversation at the center instead of surrounding it with a large control dashboard.

- Multiple named conversations with search, rename, resume, and delete.
- Local session persistence and migration from the earlier single-thread history.
- Real-time Server-Sent Events (SSE) from the standalone PyTorch brain, with incremental token text and generation metrics.
- Markdown, code blocks, tables, and safe external links.
- A selected project folder that stays associated with the local workspace.

### Practical local diagnostics

Ask naturally, or use the explicit command forms:

| Request | Command |
| --- | --- |
| Ping a host | `ping example.com` or `/ping example.com` |
| Resolve a hostname | `DNS lookup for example.com` or `/dns example.com` |
| Check one TCP port | `check port 443 on example.com` or `/port example.com 443` |
| Check local model availability | `/status` |
| See available commands | `/help` |

These are intentionally bounded local checks. Ping is limited to four packets, and a TCP check targets one host and port. Targets are validated before the local network operation runs.

### Repository-aware coding

For an explicit software-change request, SYNTRA can:

- Map the selected project and build a local source/import context.
- Search files, read bounded source ranges, and retrieve symbol, documentation, memory, or optional semantic-search results.
- Plan larger requests as ordered implementation slices with acceptance criteria.
- Generate structured, exact-anchor patches rather than blindly replacing whole files.
- Run a read-only review pass against the request, candidate changes, and retrieved evidence.
- Verify candidate changes using project checks that SYNTRA can actually discover.
- Stage reviewable proposals with source hashes, previews, backups, and local audit records.

The model-selected investigation loop is read-only. It cannot use that retrieval stage to run commands, write files, apply proposals, or access the network. Changes to tests, verification configuration, and dependency/build manifests receive additional review rather than being treated as ordinary source edits.

## ⟡ How the agent works

The workflow is evidence-driven. SYNTRA inspects the project before patching, checks the proposed result separately from the original workspace, and keeps incomplete work from being reported as verified.

<p align="center">
  <img src="docs/assets/syntra-agent-loop.svg" alt="SYNTRA agent workflow: request, inspect, retrieve evidence, patch, review, verify, and approve application." width="100%" />
</p>

**The important distinction:** generating a patch is not the same as verifying it. SYNTRA records the checks it actually ran and does not turn missing evidence into a successful verification claim.

## ⌘ Architecture

SYNTRA uses a self-contained local stack. The React/Vite interface streams chat through the authenticated loopback Node service, which starts the Python/PyTorch brain and coordinates the repository agent. Model weights, tokenizer, training data, and checkpoints remain local.

<p align="center">
  <img src="docs/assets/syntra-architecture.svg" alt="SYNTRA local architecture connecting the React interface, loopback service, local model, repository intelligence, verification, and proposal workflow." width="100%" />
</p>

| Component | Responsibility |
| --- | --- |
| React + Vite | Conversation UI, local sessions, workspace selection |
| Node.js service | Local API, chat routing, diagnostics, agent orchestration |
| Python + PyTorch brain | Randomly initialized Transformer, local tokenizer, training, checkpointing, cached inference |
| Workspace engine | Project discovery, retrieval, symbol index, proposals, audit |
| SSE bridge | Authenticated localhost streaming, metrics, cancellation, training status |

### Model reality

- **The standalone PyTorch brain is the only model provider in the workspace chat.** There is no external model-provider fallback.
- **The model starts from random weights.** It must be trained locally before model-backed chat and coding assistance can run.
- **No hosted AI API or external model registry is used.** The application launches the local Python generator as a child process and streams its output through the authenticated Node API.
- **Capability remains experimental.** Local training and passing software tests do not establish professional coding ability; instruction-following and task-level performance depend on training data, compute, and evaluation.

### Standalone `/brain` training and inference (Phase 2)

The standalone `brain/` package can now train its own randomly initialized PyTorch Transformer from local UTF-8 text/code files. It does not download model weights or contact an external model registry or hosted AI API.

Train from the default `dataset.txt`:

```powershell
python -m brain.train --dataset dataset.txt --epochs 3 --device auto
```

Train directly from a selected local project directory (source files are discovered recursively):

```powershell
python -m brain.train --dataset .\path\to\your\project --epochs 3 --device auto
```

Or add multiple local text/code files and folders by repeating `--dataset`:

```powershell
python -m brain.train --dataset .\corpus\python.txt --dataset .\corpus\docs.txt --epochs 2 --device auto
```

Checkpoints are written to `checkpoints/syntra-brain.pt` by default. They include the model weights, tokenizer merge table, AdamW optimizer state, training progress, dataset fingerprint, and random-number-generator state. Checkpoints are atomically replaced and can be resumed against the exact same corpus:

```powershell
python -m brain.train --dataset dataset.txt --epochs 2 --resume
```

Generate text from a trained checkpoint with temperature, top-k, top-p, and repetition controls:

```powershell
python -m brain.generate --checkpoint checkpoints/syntra-brain.pt --prompt "def calculate_total(" --max-new-tokens 128 --temperature 0.7 --top-k 40
```

The generator emits decoded text incrementally and uses a KV cache between tokens. Training reports epoch progress, loss, validation loss, and throughput. The dataset reader enforces a configurable size limit and rejects invalid UTF-8. Keep private keys, credentials, customer data, and other secrets out of training corpora.

### Phase 3 integration

The workspace UI now uses the standalone brain directly. The loopback Node service exposes an authenticated `POST /api/chat/stream` endpoint that emits SSE events:

- `start` and `phase` describe the current chat turn.
- `delta` streams decoded text as the Python generator produces it.
- `metrics` reports checkpoint epoch, step, saved train/validation loss, device, and live tokens per second.
- `done` or `error` closes the turn with its final state.

The existing Stop control aborts the active chat job and terminates its Python generation process. The interface also shows checkpoint state, recent loss, epoch, and generation speed. Open **Active workspace** and select **Train local brain on this workspace** to start a local training job. Progress and loss are polled from the local service; the last atomic checkpoint is preserved when training is cancelled.

The SSE endpoint is loopback-only and requires the same short-lived local API session token as the workspace API. The frontend sends it as a Bearer token. There is no external provider fallback.

A from-scratch small model is an experiment, not a substitute for a pretrained professional coding model, and low validation loss alone does not establish coding ability.


### Phase 4 — instruction corpus and intelligence evaluation

The Phase 4 training path converts approved local projects into grounded prompt/completion examples. It trains on assistant responses while masking instruction tokens from cross-entropy loss, holds out source files for validation where possible, uses AdamW with gradient accumulation, warmup, cosine learning-rate decay, weight decay, and gradient clipping, then runs a bounded local fixture evaluation at every epoch boundary.

Prepare a reusable JSONL corpus:

\`\`\`powershell
$env:SYNTRA_ALLOWED_CORPUS_ROOTS = "D:\Datasets\MySource"
python scripts/prepare_corpus.py --root "D:\Datasets\MySource" --output "data\corpus\phase4-instructions.jsonl"
\`\`\`

Train directly from that prepared corpus:

\`\`\`powershell
python -m brain.train --dataset "data\corpus\phase4-instructions.jsonl" --instruction-mode --epochs 1 --device auto
\`\`\`

Or train from the active workspace in the application. The checked-in \`brain/config.json\` is the source of truth for model dimensions, a 512-token context, micro-batch size 1 with four-step gradient accumulation, learning rate, cosine floor, warmup, checkpoint cadence, and benchmark task selection. Device auto-selection prefers CUDA, then Apple MPS, and falls back to CPU.

At each epoch boundary, the configured benchmark runs only the small \`utility-clamp\` JavaScript and \`python-slugify\` fixtures against the just-saved checkpoint. Reports are written to \`benchmark-results/phase4-epoch-*.json\`. The workspace panel exposes task completion, patch validity, and verification rates alongside train/validation loss. Run the full suite manually with \`npm run benchmark\`.

**Interpretation matters:** lower token loss measures next-token prediction on the selected corpus; it does not prove the model can solve software tasks. Fixture completion requires the expected changed-file set and successful candidate checks. These small tasks are a regression signal, not a comprehensive coding benchmark. The model still starts from random weights; this pipeline does not create pretrained knowledge or guarantee ChatGPT/Claude-level capability. Use clean, legally usable source material and never include customer data, credentials, secrets, or proprietary content without authorization.

#### Approving local training data directories

The native corpus loader only reads from the repository itself and explicitly approved local directories. Set `SYNTRA_ALLOWED_CORPUS_ROOTS` to one or more existing absolute directory paths before running native training or corpus preparation. Separate multiple paths with the platform path separator (a semicolon on Windows, a colon on Linux/macOS).

PowerShell example:

```powershell
$env:SYNTRA_ALLOWED_CORPUS_ROOTS = "D:\Datasets\MySource"
python scripts/run_syntra_module.py cli train --root "D:\Datasets\MySource"
```

The selected corpus root is resolved before traversal. Relative paths, home-directory shorthand, paths outside approved roots, and symlinks that resolve outside an approved root are rejected.

## 🚀 Get started

### Windows

**Requirements:** Node.js 20 or later, Python 3.11 or later, and Git. The launcher installs the repository's hash-locked CPU PyTorch environment into `.venv`. No external model registry or AI service is required.

1. Clone the repository:

   ```powershell
   git clone https://github.com/EldarJamiyev/SYNTRA.git
   cd SYNTRA
   ```

2. Double-click **`Start-SYNTRA.bat`**. It prepares the local Python environment and starts the Node/Vite services.
3. Open **http://127.0.0.1:3000** in your browser.
4. Open **Active workspace**, select your project folder, and choose **Train local brain on this workspace**.
5. Wait for the local training job to complete, then use the chat. Keep the service windows open while using SYNTRA.

The launcher prepares frontend and hash-locked Python dependencies, checks the Node workspace service, and starts the application. It does not automatically spend time training a model at startup. When no checkpoint exists, train it explicitly from the Active workspace panel. On later launches, use the same `Start-SYNTRA.bat` file.

### Development

From the repository root:

```bash
npm ci
npm run dev
```

The Vite development configuration starts or reuses the local workspace API. The application is intended for local use at `127.0.0.1`, not direct public internet exposure.

## ⛨ Security and execution boundaries

SYNTRA is designed around explicit boundaries rather than unrestricted model access.

- The application service binds to loopback and validates browser origins and local API requests.
- Workspace access is scoped to the selected project. Path traversal, symlink escapes, and recognised secret paths are rejected.
- Model-selected context retrieval is limited to audited, read-only tools.
- File changes are validated against source hashes and staged with previews and backups.
- Project checks can run in a restricted Docker container with network access disabled and resource limits. Host execution requires an explicit opt-in and is **not** sandboxed.
- Git inspection uses fixed, read-only operations and does not fetch from remotes.

These controls reduce risk; they do not make arbitrary repositories or the host machine risk-free. Do not run untrusted project scripts with host execution enabled. See the [security model](docs/OPERATING_SECURITY.md) for the full boundary and residual-risk notes.

## ◈ Benchmarking and evidence

SYNTRA includes a versioned local coding-agent benchmark. The documented suite covers small JavaScript utilities, configuration merging, API validation, Python, and TypeScript tasks. It checks the expected changed-file set and candidate verification—not merely whether the model returned code.

<p align="center">
  <img src="docs/assets/syntra-benchmark.svg" alt="SYNTRA coding benchmark coverage and verification criteria." width="100%" />
</p>

Run the benchmark against the local model configured on your machine:

```bash
npm run benchmark
```

List tasks, run an individual fixture, or compare saved reports:

```bash
npm run benchmark -- --list
npm run benchmark -- --task utility-clamp
npm run benchmark:reports -- history
```

The suite is a development baseline, not a broad industry benchmark and not evidence of general coding ability. Reports preserve task-level outcomes and runtime metadata; memory measurements do not represent total system/GPU memory.

## ◫ Project structure

```text
SYNTRA/
├── src/                 # React interface and local chat experience
├── local/               # Node service, SSE bridge, agent, tools, workspace engine
├── brain/               # Standalone PyTorch model, tokenizer, trainer, checkpoints, inference
├── syntra_ai/           # Earlier native Python research runtime
├── scripts/             # Setup, launch, tests, benchmark utilities
├── benchmarks/          # Versioned coding-agent fixtures
├── tests/               # Unit and integration coverage
├── e2e/                 # Browser-level tests
└── docs/
    ├── assets/          # README diagrams and visual assets
    └── ...              # Architecture, security, roadmap, and model notes
```

## ◫ Quality checks

Run the checks relevant to your change:

| Command | Purpose |
| --- | --- |
| `npm run build` | TypeScript project build and Vite production build |
| `npm run lint` | TypeScript validation |
| `npm test` | Node test suite |
| `npm run test:e2e` | Playwright end-to-end tests |
| `npm run benchmark` | Local coding-agent benchmark |

## ◇ Documentation

| Guide | What it covers |
| --- | --- |
| [Architecture](docs/ARCHITECTURE.md) | Runtime components, request lifecycle, agent orchestration, recovery |
| [Operating security](docs/OPERATING_SECURITY.md) | Local API boundaries, project-check isolation, residual risks |
| [Model capabilities](docs/MODEL_CAPABILITIES.md) | Local inference, review behavior, evaluation limits |
| [Benchmarks](docs/BENCHMARKS.md) | Fixtures, scoring, reports, measurement boundaries |
| [Data and hardware](docs/DATA_AND_HARDWARE.md) | Native model data handling and hardware constraints |
| [Product roadmap](docs/ROADMAP.md) | Current baseline, delivery phases, planned work |
| [Release checklist](docs/RELEASE_CHECKLIST.md) | Validation and release preparation |

## About

SYNTRA is developed by **[Eldar Jamiyev](https://github.com/EldarJamiyev)**.

<p align="center">
  <strong>Local intelligence. Real engineering. Your workspace stays yours.</strong><br />
  <sub>Open source under the MIT License.</sub>
</p>

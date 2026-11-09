<div align="center">
  <img src="docs/assets/syntra-hero.svg" alt="SYNTRA — Local intelligence. Real engineering." width="100%" />

  <h1>SYNTRA</h1>
  <p><strong>Native local intelligence for real engineering work.</strong></p>
  <p>A local-first AI workspace for repository understanding, bounded coding-agent workflows, model training, and practical network diagnostics — powered by SYNTRA's own PyTorch model.</p>

  <p>
    <a href="https://github.com/EldarJamiyev/SYNTRA/actions/workflows/checks.yml"><img src="https://img.shields.io/github/actions/workflow/status/EldarJamiyev/SYNTRA/checks.yml?branch=main&label=CI&logo=githubactions&logoColor=white" alt="CI status" /></a>
    <a href="https://github.com/EldarJamiyev/SYNTRA/actions/workflows/codeql.yml"><img src="https://img.shields.io/github/actions/workflow/status/EldarJamiyev/SYNTRA/codeql.yml?branch=main&label=CodeQL&logo=github&logoColor=white" alt="CodeQL status" /></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-5B8DEF.svg" alt="MIT License" /></a>
    <img src="https://img.shields.io/badge/runtime-local--first-16A085.svg" alt="Local first" />
    <img src="https://img.shields.io/badge/model-native%20PyTorch-EE6A35.svg" alt="Native PyTorch" />
    <img src="https://img.shields.io/badge/benchmark-Python%20%7C%20JS%20%7C%20TS%20%7C%20Go%20%7C%20Rust-6C63B5.svg" alt="Five-language benchmark coverage" />
  </p>

  <p>
    <a href="#what-is-syntra">Overview</a> ·
    <a href="#capabilities">Capabilities</a> ·
    <a href="#architecture">Architecture</a> ·
    <a href="#native-model--training">Native model</a> ·
    <a href="#repository-curriculum">Curriculum</a> ·
    <a href="#benchmarking--evidence">Benchmarks</a> ·
    <a href="#get-started">Get started</a> ·
    <a href="#documentation">Docs</a>
  </p>
</div>

---

## What is SYNTRA?

SYNTRA is a local engineering workspace built around a model that the project owns and trains. It combines a focused chat interface, repository-aware investigation, reviewable patch proposals, bounded agent tools, local network diagnostics, and a native Python/PyTorch Transformer.

The product is designed around a strict engineering principle: **an operation must be real, bounded, reviewable, and supported by observable evidence.** SYNTRA does not turn guessed output into telemetry, claim that an untested patch works, or present a generated response as proof of model capability.

### The non-negotiable boundaries

| Principle | Implementation |
| --- | --- |
| Native model ownership | Inference, tokenization, training, embeddings, and checkpoints belong to SYNTRA's PyTorch runtime. |
| No alternate model provider | No Ollama, hosted model API, Hugging Face Hub model download, or external inference fallback. |
| Local source of truth | Repository context is read from the selected local workspace and treated as untrusted project content. |
| Review before mutation | Generated edits are structured candidate patches; they are not silently applied as trusted changes. |
| Bounded execution | Tools have explicit inputs and resource limits. The agent is not given a general-purpose unrestricted shell. |
| Evidence over claims | Test outcomes, checkpoint identity, training history, diagnostics, and benchmark results must come from real operations. |

SYNTRA is an actively developed engineering and machine-learning project. The native model starts from random initialization; it is not a pretrained frontier coding model and is not represented as Claude-equivalent. Model progress is measured through repeatable task execution and actual pass/fail outcomes.

## Project status

The current development line builds on the merged Phase 6 context-recovery work and adds Phase 7 curriculum capabilities.

| Area | Delivered behavior |
| --- | --- |
| Native runtime | Local PyTorch Transformer, corpus-trained BPE tokenizer, checkpoint loading, inference, and supervised fine-tuning. |
| Context integrity | Full prompts are retained or rejected with a structured context_window_exceeded response (HTTP 422). |
| Context recovery | One bounded retry can recount with the selected checkpoint tokenizer and compact optional retrieved context while preserving the task and output contract. |
| Repository agent | Bounded source retrieval, planning, structured patch proposals, review, and project checks. |
| Curriculum | Python AST extraction plus bounded JavaScript, TypeScript, Go, and Rust declaration/block curation. |
| Reasoning traces | Source-linked two-file localization tasks with a thought completion and multi-file SYNTRA_AGENT_PATCH_V1 output. |
| Benchmarking | Separate full-agent and direct-model regression suites with saved reports and explicit limitations. |

Curriculum generation creates training examples; it does not prove that the model learned them. Fine-tuning must be followed by the same held-out task evaluation used for the pre-training baseline.

## Capabilities

### Chat-first local workspace

- A focused React and TypeScript chat interface with streamed local model responses.
- Multiple named conversations, local history, and workspace selection.
- Local checkpoint and training metadata status.
- Markdown, code, and structured output rendering.

Conversation history and workspace selection are stored locally in the browser on that device. They are not a cloud-synchronized account history.

### Repository-aware coding workflow

For a coding request, SYNTRA can perform a bounded sequence of operations:

1. Inspect the selected workspace and gather relevant source context.
2. Search files and symbols and read bounded source ranges.
3. Use the available static import/symbol information to understand local relationships.
4. Build a structured plan for larger changes.
5. Generate candidate edits using the versioned SYNTRA_AGENT_PATCH_V1 format.
6. Review the proposed change against the request and inspected source.
7. Run project checks that can be identified and safely executed.
8. Stage a reviewable proposal with diffs, source fingerprints, backups, and local run history.

The system can retain verified intermediate candidate state so a multi-step task can be recovered after interruption. Its static repository graph is deliberately bounded; it is not a full language server, compiler, or complete cross-language data-flow engine.

**Review boundary:** generation and review use the same native checkpoint. Review is an additional model pass, not an independent auditor or a correctness guarantee.

### Local network diagnostics

SYNTRA can perform bounded diagnostics from the machine hosting the application.

| Request | Example | Operation |
| --- | --- | --- |
| ICMP reachability | ping example.com | Real bounded ping request |
| DNS resolution | DNS lookup for example.com | Local resolver lookup |
| TCP connectivity | check port 443 on example.com | One host / one TCP port check |

These are real local checks, not simulated network telemetry. Diagnostic limits are intentional: for example, ping is constrained to a small packet count and a TCP check targets one host and one port. Network diagnostics do not require a trained model checkpoint.

### Repository context and memory

- Full-text and symbol-oriented retrieval over the selected workspace.
- A bounded static import/symbol graph for supported source patterns.
- Local semantic retrieval and project-memory embeddings produced by SYNTRA's own checkpoint.
- Deterministic lexical retrieval remains available for supported fallback paths.
- Retrieved source is context, not authority: text inside a repository cannot override the agent's system instructions or safety boundaries.

### Agent skill library

SYNTRA includes a curated local skill library for repeatable engineering workflows. Skills provide task procedures and domain guidance; they do not replace source inspection, real execution, or verification.

---

## Architecture

SYNTRA separates the user interface, local orchestration, and native model runtime so the agent can gather evidence without granting the language model direct unrestricted system access.

| Layer | Location | Responsibility |
| --- | --- | --- |
| User interface | src/ | React + TypeScript chat, conversation history, workspace selection, and response rendering. |
| Local orchestration | local/ | Loopback API, agent workflow, workspace access, tools, diagnostics, session history, and patch proposals. |
| Native AI service | syntra_ai/ | PyTorch model, tokenizer, training, inference, instruction data, and embeddings. |
| Standalone training package | brain/ | Separate native Transformer training/generation and checkpoint utilities. |
| Evaluation | benchmarks/, tests/benchmark_engine/ | Versioned task fixtures, test harnesses, scoring, and reports. |
| Operational tooling | scripts/ | Environment setup, launcher, CLI bridge, evaluation, and checkpoint management. |

### Request lifecycle

~~~mermaid
flowchart TD
    U[User request] --> UI[React chat interface]
    UI --> API[Loopback application API]
    API --> INT[Intent and task orchestration]
    INT --> CTX[Bounded workspace retrieval]
    CTX --> MODEL[Native SYNTRA PyTorch model]
    MODEL --> PATCH[Structured candidate patch]
    PATCH --> REVIEW[Review and policy checks]
    REVIEW --> VERIFY[Project checks where supported]
    VERIFY --> RESULT[Evidence-backed result]
    RESULT --> UI
~~~

The model proposes language and structure. Local application code controls file access, tool invocation, patch validation, approvals, and verification. This division is intentional: a natural-language response cannot authorize itself to perform an unrestricted operation.

---

## Native model & training

SYNTRA uses its own decoder-only PyTorch Transformer with causal attention, residual MLP blocks, cross-entropy training, AdamW, gradient clipping, atomic checkpoint writes, and checkpoint reload.

### Model lifecycle

1. **Local source pretraining** builds a corpus from approved source roots and trains from random initialization.
2. **Tokenizer persistence** stores the corpus-trained BPE merge table in the checkpoint so training, inference, token counting, and fine-tuning share the same vocabulary.
3. **Instruction fine-tuning** consumes audited local JSONL prompt/completion records and masks prompt-token loss.
4. **Evaluation** measures next-token loss and task-level behavior separately.
5. **Promotion** validates checkpoint provenance, checksum, architecture, and loadability before an explicit promotion step.

The default model context is determined by the selected checkpoint. The runtime must not silently enlarge a smaller checkpoint's context or discard the beginning of a prompt.

### Phase 6 — context integrity and recovery

The native generation API preserves the complete prompt. When the prompt cannot fit the selected checkpoint's usable context, it returns a structured context_window_exceeded response (HTTP 422) with context and overflow information. It does not truncate away the task or system instructions.

The repository agent can recover once from that specific response:

- Recount tokens with the checkpoint's actual tokenizer.
- Remove lower-priority retrieved context and optional metadata.
- Keep the user task, system instructions, and required output contract.
- Retry generation once.
- Return an explicit failure when the required context still cannot fit.

The recovery path is bounded. It is not an infinite retry loop or permission to discard required instructions.

### Phase 7 — polyglot structural curation

The curriculum generator applies structural quality scoring, boilerplate filtering, and identifier-normalized structural deduplication to the same five languages exercised by the raw-model benchmark.

| Language | Extraction approach | Repair example family |
| --- | --- | --- |
| Python | Standard-library ast | Non-identity comparisons against the None singleton |
| JavaScript | Bounded declaration/block parser | typeof comparisons using strict equality |
| TypeScript | Bounded declaration/block parser | typeof comparisons using strict equality |
| Go | Bounded declaration/block parser | Missing error value in a guarded (result, error) return |
| Rust | Bounded declaration/block parser | Unchecked slice access where Option<&T> permits .get() |

The JavaScript, TypeScript, Go, and Rust parser uses only Python standard-library text processing. It is intentionally conservative; it is not a compiler, type checker, or complete grammar. Oversized, incomplete, malformed, and unrecognized blocks are not presented as complete implementations.

Quality signals include documentation, type information, control flow, calls, meaningful operations, substantive statements, and container members. The pipeline excludes empty stubs, trivial accessors, low-quality declarations, and structurally equivalent declarations. Fingerprint memory is bounded, and reaching that bound is reported rather than silently treating unseen symbols as duplicates.

Repairs are narrow and mechanically checked. The exact search anchor must be unique in its source file. Unsupported or ambiguous patterns produce no repair example.

### Multi-file reasoning curriculum

The dedicated multi-file generator selects two files only when their relationship is present in local source evidence, such as a relative import, a same-package Go test/implementation relationship, or a Rust module/use relationship. It also checks a referenced declaration and requires a separate supported, unique repair anchor in each file.

Each generated prompt contains the two labeled source files. Its completion contains:

1. A deterministic thought localization trace that describes how to locate the implementation, consumer, and exact anchors.
2. A two-edit SYNTRA_AGENT_PATCH_V1 JSON payload using repository-relative paths.
3. The terminal EOS token added by the existing SFT tokenizer path.

The thought trace is a source-derived training target, not a hidden transcript from a model. If the generator cannot prove the file relationship and both repairs, it emits no record and does not fabricate a patch.

### Supervised loss and prompt masking

The instruction loader tokenizes each record as:

~~~text
BOS + prompt + completion + EOS
~~~

Prompt target positions are masked with -100. Completion tokens remain supervised. For multi-file examples this means the thought trace, patch JSON, and EOS contribute to SFT loss, while the prompt does not. Long responses use overlapping context windows; repeated overlap targets are masked so the same completion token is not counted twice.

### Training and model commands

Run these from the repository root:

~~~powershell
# Report actual local checkpoint and training metadata
npm run ai:cli -- status

# Inspect a local corpus without training
npm run ai:cli -- inspect-data --root "C:\Projects\SYNTRA"

# Train or continue the native model from an approved source root
npm run ai:cli -- train --root "C:\Projects\SYNTRA" --seq-len 1024 --steps 1000

# Generate repository-grounded curriculum records for all five languages
npm run ai:curriculum -- --mode all --languages python javascript typescript go rust --max-examples 240

# Generate only the regular module/symbol/repair dataset
npm run ai:curriculum -- --mode repository --languages python javascript typescript go rust --max-examples 240

# Generate only paired multi-file reasoning examples
npm run ai:curriculum -- --mode multifile --languages javascript typescript go rust --max-examples 120

# Audit a generated instruction dataset before training
npm run ai:cli -- audit-instructions --dataset "data/curricula/workspace-instructions.jsonl" --seq-len 1024

# Fine-tune the local checkpoint on an audited dataset
npm run ai:cli -- finetune --dataset "data/curricula/workspace-instructions.jsonl" --steps 1000
~~~

Curriculum output defaults to:

- data/curricula/workspace-instructions.jsonl
- data/curricula/workspace-reasoning.jsonl

Use --root to scan one approved repository instead of the project plus configured roots. Use --output or --multifile-output to choose output paths inside the project or selected workspace. --languages selects the source languages, and --mode selects repository, multifile, or all.

External corpus roots must be explicitly approved. In PowerShell:

~~~powershell
$env:SYNTRA_ALLOWED_CORPUS_ROOTS = "D:\Datasets"
npm run ai:cli -- inspect-data --root "D:\Datasets\MySource"
~~~

The corpus loader enforces canonical path containment, regular-file checks, symlink rejection, file-size/count/token limits, generated-directory exclusions, and likely-secret path exclusions. These are defensive controls, not a guarantee that every secret can be found. Review data before training and do not include credentials, private keys, customer information, or material you are not authorized to use.

**Training loss is not coding accuracy.** A low next-token loss on training data does not prove that generated patches are correct, that an agent can complete unfamiliar tasks, or that fine-tuning improved held-out performance.

### First-run behavior

On first launch, the Windows startup path prepares the local environments and trains or validates a starter checkpoint from the checked-out repository. No pretrained model weights are downloaded. Dependency installation may require access to npm and Python package indexes; after the required packages and checkpoints are present, inference can run locally.

---

## Security & execution boundaries

SYNTRA is designed for local use with explicit trust boundaries.

- **Loopback services:** application APIs bind to 127.0.0.1; the native API is not intended to be exposed directly to a network.
- **Approved corpus roots:** caller-provided paths are canonicalized and checked against the project or explicitly configured local roots at the filesystem boundary.
- **Filesystem safety:** corpus reads reject symlink escapes and verify regular-file identity and metadata before and after reading. Curriculum output paths are revalidated before writing.
- **Hidden benchmark isolation:** tests/benchmark_engine/hidden is excluded from curriculum scanning. Private evaluation targets are not valid training labels.
- **Patch safety:** candidate patch paths and exact source anchors are validated by the local patch pipeline. A model-generated diff is not trusted solely because it is valid JSON.
- **Tool limits:** diagnostics and project checks use bounded parameters. The agent does not have a general-purpose unrestricted shell.
- **Checkpoint safety:** checkpoint loading uses PyTorch's weights_only=True path where supported by the native runtime. Checkpoint identity is tracked so stale inference companions cannot shadow a replacement checkpoint.

No single path filter or secret scanner can guarantee that a corpus contains no sensitive data. Users remain responsible for approving the data they select.

---

## Benchmarking & evidence

SYNTRA maintains separate benchmark layers because they answer different questions.

| Evaluation | Measures | Does not establish |
| --- | --- | --- |
| Unit and integration tests | Application behavior, API contracts, parser behavior, path controls, and regressions | Model coding ability |
| Full-agent benchmark (npm run benchmark) | End-to-end agent behavior across versioned workspace tasks | Broad real-world engineering competence |
| Raw-model benchmark (npm run benchmark:model) | Direct checkpoint patch output on controlled fixtures across Python, JavaScript, TypeScript, Go, and Rust | Generalization to arbitrary repositories |
| Loss evaluation (npm run ai:cli -- evaluate) | Next-token prediction loss on a selected local corpus | Patch correctness or independent generalization when the corpus overlaps training data |

Useful commands:

~~~powershell
# Run the full agent benchmark
npm run benchmark

# List direct-model tasks without loading a checkpoint
npm run ai:cli -- benchmark-model --list

# Run the direct-model suite against the selected local checkpoint
npm run benchmark:model

# Produce benchmark reports
npm run benchmark:reports
~~~

The raw-model evaluator uses pre-existing local runtime images, disables network access in Docker mode, and does not pull images during evaluation. Host execution is unsandboxed and requires the explicit --isolation host --allow-host-execution opt-in. Pre-provision the required images and locked dependencies before Docker-mode evaluation.

Benchmark fixtures are narrow, controlled regression cases. They are not an industry-wide benchmark, and the test harness passing does not mean the model has passed those tasks. Only a real run against the actual local checkpoint creates model-performance evidence. Reports record checkpoint identity, task outcomes, and relevant generation/context metrics; missing historical model metrics are not invented or backfilled.

### Phase 4 and Phase 5 workflows

The standalone training/evaluation workflow and checkpoint promotion tools remain available:

~~~powershell
# Run the Phase 4 local checkpoint workflow
npm run brain:milestone4 -- --device auto

# Train with a selected local accelerator
npm run brain:milestone4 -- --device cpu
npm run brain:milestone4 -- --device cuda
npm run brain:milestone4 -- --device mps

# Use an existing checkpoint for SFT only
npm run brain:milestone4 -- --skip-pretraining

# Benchmark an existing checkpoint without pretraining or SFT
npm run brain:milestone4 -- --skip-pretraining --skip-sft

# Validate checkpoint provenance and loadability before promotion
npm run brain:promote -- --dry-run

# Review and explicitly promote after validation
npm run brain:promote
~~~

These workflows use separate staging/checkpoint paths and should be run with the corresponding documented configuration. Review the generated reports and actual local hardware measurements rather than assuming a profile is optimal for every CPU, CUDA, or MPS device.

---

## Get started

### Windows — one-click launch

**Requirements:** Windows, PowerShell, Node.js 20 or newer, and Python 3.12 or newer.

~~~powershell
git clone https://github.com/EldarJamiyev/SYNTRA.git
cd SYNTRA
~~~

Double-click Start-SYNTRA.bat or run:

~~~powershell
.\Start-SYNTRA.bat
~~~

The launcher prepares the local environments, validates or trains the native checkpoint, starts the required services, and opens:

**http://127.0.0.1:3000**

Keep the service windows open while using SYNTRA. On later launches, the same launcher reuses a valid checkpoint instead of retraining from scratch.

### Development

Install the locked Node dependencies and run the application checks:

~~~bash
npm ci
npm run lint
npm run build
npm test
npm run test:e2e
~~~

The application also requires the native Python service and the Node workspace service. The Windows launcher is the maintained one-click startup path; use the architecture and operating documentation when starting services separately.

---

## Project structure

~~~text
SYNTRA/
├── src/                         # React + TypeScript chat interface
├── local/                       # Node API, agent, workspace and diagnostics
├── syntra_ai/                   # Native PyTorch model, data, training and inference
├── brain/                       # Standalone native Transformer training package
├── .agents/skills/              # Curated local agent skill library
├── benchmarks/                  # Versioned public benchmark fixtures
├── tests/benchmark_engine/      # Raw-model harness and isolated hidden targets
├── scripts/                     # Setup, launcher, CLI bridge and evaluation tools
├── tests/                       # Python, Node and integration tests
├── e2e/                          # Browser end-to-end tests
└── docs/                         # Architecture, security, model and benchmark docs
~~~

## Documentation

| Document | Scope |
| --- | --- |
| [Architecture](docs/ARCHITECTURE.md) | Runtime components, request flow, agent orchestration, and recovery |
| [Native model runtime](docs/NATIVE_MODEL.md) | PyTorch model, tokenizer, context integrity, training, and inference |
| [Repository curriculum](docs/REPOSITORY_CURRICULUM.md) | Five-language curation, repair evidence, paired reasoning, and SFT masks |
| [Operating security](docs/OPERATING_SECURITY.md) | Local API boundaries, project-check isolation, path safety, and residual risks |
| [Model capabilities](docs/MODEL_CAPABILITIES.md) | Native inference, self-review, benchmark evidence, and limits |
| [Benchmarks](docs/BENCHMARKS.md) | Fixture suite, scoring, report history, and measurement boundaries |
| [Data and hardware](docs/DATA_AND_HARDWARE.md) | Local corpus handling, checkpoint limits, memory estimates, and hardware notes |
| [Product roadmap](docs/ROADMAP.md) | Delivered baseline, remaining phases, and acceptance criteria |
| [Release checklist](docs/RELEASE_CHECKLIST.md) | Repeatable CI, local runtime, security, and release gates |

## Contributing

Changes should preserve the local-model boundary, existing product scope, and real-operation guarantees. New tools must have bounded inputs and observable outcomes. New model or agent capabilities must be backed by tests against real code and must not rely on fabricated telemetry or placeholder responses.

Before proposing a change, run the relevant Python and Node tests. For model-related work, include the checkpoint identity, dataset provenance, and actual benchmark outcomes needed to reproduce the claim. For security-sensitive changes, add a regression test for the underlying boundary, not just the reported alert.

## License

SYNTRA is released under the [MIT License](LICENSE). Third-party skill and asset notices are documented in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

---

<div align="center">
  <strong>Local intelligence. Real engineering.</strong><br />
  Developed by <a href="https://github.com/EldarJamiyev">Eldar Jamiyev</a>.
</div>

# SYNTRA — Local AI Chat and Coding Agent

SYNTRA is a local-first chat interface backed by a native PyTorch Transformer and a repository-aware coding-agent runtime. The browser UI is intentionally simple: one conversation and one message box. The service does not call OpenAI, Anthropic, Ollama, or another hosted model API, and the repository does not ship pretrained weights.

## One-click Windows launch

Requirements: **Node.js 20+**, **Python 3.11+**, and Git. Install Node.js from https://nodejs.org/ and Python from https://www.python.org/downloads/ (enable **Add Python to PATH** during setup).

From the existing SYNTRA repository folder, open PowerShell and run this once to receive the latest launcher:

~~~powershell
git pull
~~~

Then **double-click Start-SYNTRA.bat** in the root of the repository. It automatically:

1. Checks Node.js and Python versions.
2. Installs web dependencies if needed and creates/repairs the local .venv with CPU PyTorch and NumPy.
3. Trains the local checkpoint on first launch if one does not exist; if that first training was interrupted, it resumes from the last saved checkpoint.
4. Starts the AI, workspace, and web services, then opens the chat at http://127.0.0.1:3000.

Keep the opened service windows running while using SYNTRA. Next time, double-click Start-SYNTRA.bat again; setup and training are skipped when already complete. If a port is occupied by a service that is not SYNTRA, the launcher stops and identifies the port rather than silently opening duplicate services.

## What the system actually does

- A decoder-only causal Transformer trains from random initialization and saves a local checkpoint with a corpus-trained byte-pair (BPE) tokenizer.
- The coding agent can inspect a repository, map imports, retrieve relevant files, prepare structured patches, run available checks in a temporary copy, and retry a bounded number of times.
- Changes are applied only after supported checks pass. Otherwise, proposals remain unapplied unless you explicitly type /apply.
- Type /workspace C:\path\to\project in chat to switch the project. By default, SYNTRA uses its own repository folder.
- The evaluate command measures sampled next-token loss. This is not a code-correctness benchmark.

## Manual commands

From PowerShell in the repository folder:

~~~powershell
npm install
npm run ai:install
npm run ai:cli -- status
npm run build
npm test
~~~

The local launcher is scripts/start-syntra.ps1; Start-SYNTRA.bat is the double-click entry point.

## Important capability limit

SYNTRA starts without pretrained weights. First-run training learns from the source files selected in this repository; that is enough to create a runnable local checkpoint, but it does **not** give the model the broad knowledge or instruction-following ability of Claude or ChatGPT. This launcher fixes setup and startup friction; it does not manufacture intelligence absent from the weights. More capable use requires substantial high-quality, properly licensed training data, instruction tuning, and task-level benchmarks.

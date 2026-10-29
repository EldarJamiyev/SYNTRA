# SYNTRA — Local AI Software Engineering Agent

SYNTRA combines a minimal chat interface, repository-aware patch generation, local project tools, and automated verification. It uses no hosted model API. In automatic mode it prefers a suitable model already installed in local Ollama; otherwise it can use the native PyTorch model trained on your source tree.

## One-click Windows launch

Requirements: **Node.js 20+** and Git. Python 3.11+ is required only when using SYNTRA's native from-scratch fallback. Install Node.js from https://nodejs.org/. Install Python from https://www.python.org/downloads/ if you do not have a local Ollama model.

From your existing repository folder, run this once in PowerShell:

~~~powershell
git pull
~~~

Then double-click **Start-SYNTRA.bat**. The launcher checks dependencies, detects and starts the local Ollama service if needed, and chooses the best installed local model that fits its memory-safety limit. If a suitable Ollama model is installed, it skips native training. Otherwise it repairs the local Python environment, trains a native checkpoint if required, starts the services, and opens http://127.0.0.1:3000.

Keep the service windows open. On future launches, double-click the same file again.

## Stronger local coding model

SYNTRA automatically prefers installed coding-capable local models (for example Qwen3-Coder, Devstral, Qwen2.5-Coder, and similar models), avoids cloud-only model tags, and never sends prompts to a remote model endpoint. The actual selected model depends on which weights are installed and available RAM.

For a stronger local coding model on a machine with 32 GB RAM, open a terminal and install Qwen3-Coder 30B:

~~~powershell
ollama pull qwen3-coder:30b
~~~

The model is about 19 GB in the Ollama library; the first pull requires internet and disk space. Subsequent inference is local. Review the model license and available memory before installing; another smaller coding model may be more responsive on your PC. Official model page: https://ollama.com/library/qwen3-coder

You can pin an installed model through the Windows environment variable SYNTRA_LOCAL_MODEL. To disable local Ollama selection and force native mode, set SYNTRA_MODEL_PROVIDER=native. To require a local Ollama model, set SYNTRA_MODEL_PROVIDER=ollama. The default is auto.

## What the agent does

- Builds a local source-file inventory and import graph, then ranks project files relevant to your request.
- Requests structured JSON edits from the selected model, validating path boundaries, exact anchors, sensitive-file restrictions, and source hashes.
- Applies proposed changes in a temporary project copy and runs discovered build, test, lint, or typecheck commands.
- Feeds failures into bounded repair attempts; it applies verified proposals only after discovered checks pass. Unverified proposals remain staged until you explicitly type /apply.
- Change the workspace by typing /workspace followed by an absolute folder path.

## Honest capability limits

SYNTRA can be stronger than a plain chat session on a particular codebase when its local model, repository retrieval, test suite, and repair loop work well together. It is **not guaranteed to outperform Claude**, and no launcher or agent scaffolding can make a small randomly initialized model equivalent to a frontier model. Evaluate real tasks using clean test suites and held-out benchmarks, not training loss or marketing claims.

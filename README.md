# SYNTRA

SYNTRA is an experimental coding-focused language model trained from random initialization. This repository contains the model architecture, byte tokenizer, training loop, checkpoint handling, inference/generation, evaluation, CLI, local API, and React GUI.

**No pretrained model, pretrained weights, hosted inference, Ollama, or external AI API is used.** The tokenizer, transformer layers, training procedure, checkpoint handling, and generation loop are implemented here. PyTorch provides tensor operations and automatic differentiation; a fully custom tensor/GPU engine is a separate project and is not claimed here.

**Reality check:** the initial model is a small research model, not a Claude-level coding model. It starts with random weights and will initially produce poor output. Capability depends on high-quality permitted training data, compute, evaluation, and iteration.

## Included

- Decoder-only causal transformer with learned token and position embeddings.
- Lossless UTF-8 byte tokenizer with four reserved tokens and 260 token IDs.
- Training from random initialization, validation loss, gradient clipping, cosine learning-rate schedule, checkpoint/resume, and CSV metrics.
- Local-only corpus preparation from source files you provide.
- CLI to prepare data, train, generate, evaluate, and serve the UI backend.
- FastAPI service bound to loopback by default.
- React + TypeScript GUI.
- Unit tests and GitHub Actions workflow.

## Requirements

- Windows 11 or another Python 3.11+ environment.
- 32 GB RAM is sufficient for the starter model and small-to-moderate datasets, depending on other applications and corpus size.
- Dependencies in requirements.txt.
- Node.js 20+ and npm for the GUI.
- Disk space for your corpus, checkpoints, virtual environment, and Windows pagefile.

The default model is approximately 7–8 million parameters with a 128-token context. CPU training on a laptop can be slow. Virtual memory may prevent some memory-exhaustion failures, but it is much slower than physical RAM and adds no compute throughput. The integrated Intel Arc GPU is not used by the initial CUDA-only acceleration path.

## Windows quick start

Run PowerShell from the repository root.

1. Set up Python dependencies and local directories:

   ~~~powershell
   .\scripts\setup_windows.ps1
   ~~~

2. Put source code you are allowed to use for training in data/raw. Supported extensions include Python, JavaScript, TypeScript, Java, Rust, Go, C/C++, C#, HTML, CSS, SQL, JSON, Markdown, and plain text. Start with a small, curated corpus.

3. Prepare the token stream:

   ~~~powershell
   .\.venv\Scripts\python.exe -m syntra.cli prepare
   ~~~

4. Train from randomly initialized weights:

   ~~~powershell
   .\.venv\Scripts\python.exe -m syntra.cli train --config config/default.json
   ~~~

5. Generate from your trained checkpoint:

   ~~~powershell
   .\.venv\Scripts\python.exe -m syntra.cli generate --checkpoint checkpoints/latest.pt --prompt "def add(a, b):" --max-new-tokens 80
   ~~~

6. Start the local backend:

   ~~~powershell
   .\.venv\Scripts\python.exe -m syntra.cli serve
   ~~~

7. In a second PowerShell window:

   ~~~powershell
   cd frontend
   npm install
   npm run dev
   ~~~

Open the local URL reported by Vite, normally http://127.0.0.1:5173.

The GUI reports that no checkpoint exists until training creates checkpoints/latest.pt.

## CLI reference

~~~powershell
python -m syntra.cli --help
python -m syntra.cli prepare --raw-dir data/raw --out data/processed/tokens.bin
python -m syntra.cli train --config config/default.json
python -m syntra.cli train --config config/default.json --resume checkpoints/latest.pt
python -m syntra.cli generate --checkpoint checkpoints/latest.pt --prompt "class Queue:" --max-new-tokens 120
python -m syntra.cli evaluate --checkpoint checkpoints/latest.pt --data-file data/processed/tokens.bin
python -m syntra.cli serve --host 127.0.0.1 --port 8000
~~~

For a first pipeline test, edit config/default.json and use smaller values for embedding size, layers, context, batch size, and training steps. The embedding dimension must be divisible by the number of attention heads. Vocabulary size must remain 260 for the built-in tokenizer.

## Data and safety

- SYNTRA does not download training data or model weights.
- Corpus preparation reads only local files placed under the configured raw-data directory.
- Checkpoints are local and excluded from Git.
- Do not commit datasets, model checkpoints, secrets, or private source code.
- Only load checkpoint files you trust. PyTorch checkpoint formats can be risky when obtained from untrusted sources.
- Generated code is output as text only. This version does not execute model-generated code.

## Development checks

~~~powershell
python -m pip install -e ".[dev]"
python -m pytest
~~~

Run npm install and npm run build in the frontend directory to build the GUI. See docs/ARCHITECTURE.md and docs/DATA_AND_HARDWARE.md before scaling training.

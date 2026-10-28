# SYNTRA — Local Coding Model and Project Workspace

SYNTRA is a local-first software-development system with a native model-training runtime and a project workspace. It does not call OpenAI, Anthropic, Ollama, or any hosted model API. The neural network is initialized from random weights. PyTorch supplies tensor operations, automatic differentiation and optimizers; it is not a pretrained model.

## What is real and implemented

- A decoder-only causal Transformer implemented in `syntra_ai/model.py`, with causal self-attention, residual MLP blocks, token/position embeddings, cross-entropy training loss, backpropagation, gradient clipping and AdamW updates.
- Local corpus discovery for source and documentation files, with limits and exclusions for common build/dependency folders and likely secret files.
- A CPU/CUDA/MPS-aware training CLI that records training and held-out validation loss and atomically saves a model/optimizer checkpoint.
- A local checkpoint loader, real autoregressive token sampling, and a loopback-only Python HTTP API.
- A project workspace that inspects/searches files, creates proposals, backs up existing files, checks the source hash before apply, writes an audit record, and runs explicit build/test/lint/typecheck commands.
- A UI panel connected to the real Python service. It can start training on a selected local repository, read checkpoint/validation status, and generate from the trained local checkpoint into the project editor.
- Automated Node/UI and Python/neural-model checks in GitHub Actions.

## Important capability boundary

A working Transformer implementation is not the same as a capable coding model. The model starts with random weights and learns next-byte patterns from the local files you provide. It is not pre-trained, and the repository does not contain bundled pretrained weights or a large coding dataset. Initial training will not make it equivalent to Claude, ChatGPT, or an established coding model.

The present generator performs **raw continuation** of a prompt. It does not yet reliably follow natural-language instructions, produce a complete multi-file application, or autonomously repair a project. Its completion may be malformed or irrelevant. The UI labels this output and routes it through the existing review/proposal path. The next real milestone for useful task-following is a supervised instruction-tuning pipeline and a model trained on a much larger, carefully licensed code/instruction corpus. No quality is implied merely because loss decreases.

The model's byte-level vocabulary also means its context length is measured in bytes, not words or source-code tokens. The default context is 1,024 bytes. This is intentionally a small local experiment suited to development and training-loop verification, not a claim of EJI-scale capability.

## Requirements

- Node.js 20+
- Python 3.11+
- PyTorch 2.4–2.10 CPU build by default; CUDA/MPS require a matching PyTorch GPU build
- Enough local source code to fill both train and validation splits. With the default 1,024-byte context and an 80/20 split, provide at least about 5,126 byte tokens; in practice, use a substantial repository rather than a few tiny files.

## Install

From PowerShell in the repository root:

```powershell
npm install
npm run ai:install
```

`npm run ai:install` creates an isolated `.venv` in the repository and installs CPU-only PyTorch there, avoiding global Python package changes and the NVIDIA/CUDA stack by default. SYNTRA does not download model weights. NVIDIA users can install a matching CUDA-enabled PyTorch build separately; Intel Arc acceleration is not implemented yet.

## Start the application

The easiest route after installing requirements:

```powershell
.\scripts\start-syntra.ps1
```

Or run each service in its own PowerShell terminal:

Terminal 1 — native Transformer API:
```powershell
npm run ai:server
```

Terminal 2 — project workspace API:
```powershell
npm run server
```

Terminal 3 — web UI:
```powershell
npm run dev
```

Open http://127.0.0.1:3000. The Python model API listens on `127.0.0.1:3002`; the workspace API listens on `127.0.0.1:3001`. Both services bind to loopback only.

## Train your own weights

Use a local repository you own or are authorized to process as the source corpus. In the UI, enter and inspect its absolute path and select **Train from local source**. Or use the CLI:

```powershell
npm run ai:cli -- inspect-data --root "C:\Users\You\Documents\YourProject"
npm run ai:cli -- train --root "C:\Users\You\Documents\YourProject" --steps 1000
npm run ai:cli -- status
npm run ai:cli -- generate --prompt "def parse_config(" --max-new-tokens 200
npm run ai:cli -- evaluate --device cpu
npm run ai:cli -- finetune --dataset "C:\\Users\\You\\Documents\\syntra-instructions.jsonl" --steps 1000
```

Default architecture: 4 decoder blocks, 4 attention heads, width 128, context 1,024 bytes. The default checkpoint is written to `data/checkpoints/syntra.pt`, with measured training/validation metrics alongside it. Checkpoints are ignored by Git and stay on the local machine. Continue training an existing matching architecture with `--resume`. To use a CUDA device explicitly, pass `--device cuda`; by default SYNTRA chooses CUDA, Apple MPS, or CPU based on availability.

Training operates on bytes from local text/source files, excluding common generated folders, dependencies, and likely credential files. Review the corpus source and permissions before training. This is a simple source corpus loader, not a guarantee that every sensitive pattern can be recognized.

### Supervised instruction fine-tuning

After a source-pretraining checkpoint exists, prepare a local UTF-8 JSONL file with one object per line. Each object must have `prompt` and `completion` string fields:

```jsonl
{"prompt":"Write a Python add function.","completion":"def add(a, b):\\n    return a + b"}
{"prompt":"Write a TypeScript clamp helper.","completion":"export function clamp(n: number, low: number, high: number): number { return Math.min(high, Math.max(low, n)); }"}
```

Then run `npm run ai:cli -- finetune --dataset "C:\\Users\\You\\Documents\\syntra-instructions.jsonl" --steps 1000` or enter that local path in the UI and choose **Supervised instruction fine-tuning**. The loss masks the prompt tokens and trains the model on completion tokens. The current byte context limits examples and completions to a small window; curate many high-quality examples, keep a separate hold-out set, and measure whether task performance actually improves. Fine-tuning does not guarantee useful or correct code generation.

## Project builder workflow

1. Inspect the project with its absolute local path.
2. Search and load the source file you intend to work on.
3. Train a local checkpoint from an authorized code corpus.
4. Ask the local model to continue the loaded file from a specific point. The current model performs raw continuation; it does not create a reliable full-file patch from a natural-language request.
5. Read and edit the generated text in the editor. Click **Create proposal** to stage the exact reviewed contents.
6. Review the path and diff. Explicitly authorize apply only when satisfied; SYNTRA checks the original hash and backs up existing content before writing.
7. Run the actual project's tests/build/lint and inspect the captured output.

Do not apply raw model output to important projects without review and verification. A passing build/test result is evidence for that particular check, not proof of correctness or security.

## CLI for local project operations

```powershell
node bin/syntra.mjs help
node bin/syntra.mjs inspect --root "C:\Users\You\Documents\YourProject"
node bin/syntra.mjs search --root "C:\Users\You\Documents\YourProject" --query "service"
node bin/syntra.mjs check --root "C:\Users\You\Documents\YourProject" --task test
```

## Local API and checkpoints

- Python AI health: `GET http://127.0.0.1:3002/health`
- Native model status: `GET http://127.0.0.1:3002/model/status`
- Training start: `POST http://127.0.0.1:3002/model/train`
- Generate: `POST http://127.0.0.1:3002/model/generate`
- Workspace/API bridge: `http://127.0.0.1:3001/local-api`

Training and generation payloads are JSON. The Python API accepts requests only over loopback and enforces body-size and generation limits. Do not expose it to a LAN or the public internet.

## Development checks

```powershell
npm test
npm run build
python -m compileall -q syntra_ai tests/python
python -m unittest discover -s tests/python -v
```

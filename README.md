# SYNTRA — Local AI Chat and Coding Agent

SYNTRA is a local-first chat interface backed by a native PyTorch Transformer and a repository-aware coding-agent runtime. The browser UI is intentionally simple: one conversation and one message box. The service runs on your own machine; it does not call OpenAI, Anthropic, Ollama, or another hosted model API, and the repository does not ship pretrained weights.

## What the system actually does

- A decoder-only causal Transformer trains from random initialization, writes local checkpoints, and generates tokens using a checkpoint-matched corpus-trained byte-pair (BPE) tokenizer.
- Context length is configurable from 16 to 16,384 BPE tokens. The default model configuration is 4,096 tokens. Real usage depends on vocabulary compression, selected context, available RAM, and CPU/GPU performance.
- Coding requests run through a bounded agent loop: repository inspection, a local import/dependency graph, relevance-ranked file retrieval, token-budget pruning, structured JSON find/replace patches, strict path and exact-anchor validation, temporary-copy build/test checks, and up to three correction attempts with failure output returned to the model.
- If discovered checks pass, SYNTRA applies the hash-checked proposals to the selected workspace and saves backups/audit records under `.syntra`. If validation does not pass or there is no runnable check, proposals are left unapplied; type `/apply` only if you explicitly choose to apply those unverified proposals.
- Type `/workspace C:\\path\\to\\project` in chat to switch the project. By default, SYNTRA uses its own repository folder.
- The `evaluate` command runs a fresh sampled next-token loss measurement on a local corpus. It is not a code-correctness benchmark.

## Install and run (Windows PowerShell)

Install Node.js 20+ and Python 3.11+, then from the SYNTRA repository folder:

```powershell
npm install
npm run ai:install
```

The installer creates a repository-local `.venv` and installs CPU-only PyTorch. Start the local AI service, workspace service, and web UI in three terminals:

```powershell
npm run ai:server
npm run server
npm run dev
```

Then open http://127.0.0.1:3000. Or launch all three services with:

```powershell
.\\scripts\\start-syntra.ps1
```

### One-time local training

SYNTRA cannot answer from an untrained checkpoint. Train with code/documentation that you are allowed to use. The initial weights are random, so this is a prerequisite—not a guarantee of useful general chat or coding output.

```powershell
npm run ai:cli -- train --root "C:\\path\\to\\a\\source\\repository" --seq-len 1024 --steps 1000
```

Once training completes, return to the chat and ask a question or describe a code change. A bigger context consumes more compute and does not increase underlying reasoning ability by itself. For instruction-following, prepare your own or properly licensed prompt/completion JSONL dataset and use `npm run ai:cli -- finetune --dataset "C:\\path\\to\\examples.jsonl"`.

Useful local commands:

```powershell
npm run ai:cli -- status
npm run ai:cli -- tokenize --text "def add(a, b): return a + b"
npm run ai:cli -- evaluate --root "C:\\path\\to\\heldout\\corpus"
npm test
npm run build
```

## Verification

GitHub Actions runs the frontend TypeScript/Vite build, Node workspace/agent tests, Python compilation, and native model/API training tests. These automated tests establish that code paths execute; they do not prove model-generated changes are correct for every real-world project.

## Important capability limit

The coding-agent architecture is real, but the neural model still starts randomly initialized and is small. Local source pretraining alone does **not** make it comparable to Claude or ChatGPT. BPE, context management, tools, tests and repair loops make the system less fragile; they cannot create knowledge or reasoning absent from the trained weights. Do not treat falling training loss or a passing project test as a general measure of coding intelligence. For dependable app development, SYNTRA still needs substantial high-quality licensed training data, instruction tuning, and repeatable task-level coding benchmarks.

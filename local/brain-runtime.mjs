import { spawn } from "node:child_process";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { StringDecoder } from "node:string_decoder";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const DEFAULT_CHECKPOINT = path.join(ROOT, "checkpoints", "syntra-brain.pt");
const MAX_PROMPT_BYTES = 256 * 1024;
const MAX_GENERATION_TOKENS = 2048;
let trainingProcess = null;
let trainingCancelled = false;
let trainingState = {
  status: "idle",
  epoch: 0,
  step: 0,
  train_loss: null,
  validation_loss: null,
  tokens_per_second: 0,
  message: "No training run has started in this session.",
  updated_at: new Date().toISOString()
};

export function brainCheckpointPath() {
  const configured = process.env.SYNTRA_BRAIN_CHECKPOINT || process.env.SYNTRA_CHECKPOINT;
  return path.resolve(configured || DEFAULT_CHECKPOINT);
}

export function pythonExecutable() {
  if (process.env.SYNTRA_PYTHON?.trim()) return process.env.SYNTRA_PYTHON.trim();
  const virtualEnvironment = path.join(
    ROOT,
    ".venv",
    process.platform === "win32" ? "Scripts" : "bin",
    process.platform === "win32" ? "python.exe" : "python"
  );
  return virtualEnvironment;
}

async function selectedPython() {
  const candidate = pythonExecutable();
  try {
    await fs.access(candidate);
    return candidate;
  } catch {
    return process.platform === "win32" ? "python" : "python3";
  }
}

function runtimeEnvironment() {
  return {
    ...process.env,
    PYTHONUTF8: "1",
    PYTHONIOENCODING: "utf-8",
    PYTHONUNBUFFERED: "1",
    SYNTRA_BRAIN_CHECKPOINT: brainCheckpointPath()
  };
}

function metadataPath(checkpoint = brainCheckpointPath()) {
  return checkpoint + ".json";
}

let tokenizerCache = null;

function tokenizerSidecarPath(checkpoint = brainCheckpointPath()) {
  return checkpoint + ".tokenizer.json";
}

function validateTokenizerJson(value) {
  if (!value || typeof value !== "object" || value.format !== "syntra-local-byte-bpe-v1"
    || !Array.isArray(value.merges) || value.merges.length > 8192 - 258
    || value.vocab_size !== value.merges.length + 258) {
    throw new Error("The local checkpoint tokenizer sidecar is missing or invalid.");
  }
  let nextId = 258;
  const seen = new Set();
  for (const pair of value.merges) {
    if (!Array.isArray(pair) || pair.length !== 2
      || !pair.every(token => Number.isInteger(token) && token >= 0 && token < nextId)
      || pair.some(token => token === 256 || token === 257)) {
      throw new Error("The local checkpoint tokenizer merge table is malformed.");
    }
    const key = pair[0] + ":" + pair[1];
    if (seen.has(key)) throw new Error("The local checkpoint tokenizer contains duplicate merges.");
    seen.add(key);
    nextId += 1;
  }
  return value;
}

async function exportTokenizerSidecar(checkpoint) {
  const python = await selectedPython();
  const script = [
    "import json, sys",
    "from brain.checkpoint import load_checkpoint",
    "payload = load_checkpoint(sys.argv[1])",
    "print(json.dumps(payload['tokenizer'], ensure_ascii=False, separators=(',', ':')))"
  ].join("; ");
  return await new Promise((resolve, reject) => {
    const child = spawn(python, ["-c", script, checkpoint], {
      cwd: ROOT,
      env: runtimeEnvironment(),
      stdio: ["ignore", "pipe", "pipe"],
      windowsHide: true
    });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", chunk => {
      stdout += chunk.toString("utf8");
      if (stdout.length > 2 * 1024 * 1024) child.kill("SIGTERM");
    });
    child.stderr.on("data", chunk => {
      stderr += chunk.toString("utf8");
      if (stderr.length > 8000) stderr = stderr.slice(-8000);
    });
    child.on("error", reject);
    child.on("close", async code => {
      if (code !== 0) {
        reject(new Error(stderr.trim() || "Could not export the tokenizer from the existing local checkpoint."));
        return;
      }
      try {
        const tokenizer = validateTokenizerJson(JSON.parse(stdout));
        const destination = tokenizerSidecarPath(checkpoint);
        const temporary = destination + "." + process.pid + ".tmp";
        await fs.writeFile(temporary, JSON.stringify(tokenizer), { encoding: "utf8", flag: "wx" });
        await fs.rename(temporary, destination);
        resolve(tokenizer);
      } catch (error) {
        reject(error);
      }
    });
  });
}

async function readLocalTokenizer() {
  const checkpoint = brainCheckpointPath();
  const sidecar = tokenizerSidecarPath(checkpoint);
  const checkpointStat = await fs.stat(checkpoint);
  try {
    const sidecarStat = await fs.stat(sidecar);
    if (sidecarStat.isFile() && sidecarStat.size <= 2 * 1024 * 1024
      && sidecarStat.mtimeMs >= checkpointStat.mtimeMs) {
      if (tokenizerCache?.path === sidecar && tokenizerCache.mtimeMs === sidecarStat.mtimeMs) {
        return tokenizerCache.value;
      }
      const value = validateTokenizerJson(JSON.parse(await fs.readFile(sidecar, "utf8")));
      tokenizerCache = { path: sidecar, mtimeMs: sidecarStat.mtimeMs, value };
      return value;
    }
  } catch {}
  const value = await exportTokenizerSidecar(checkpoint);
  const stat = await fs.stat(sidecar);
  tokenizerCache = { path: sidecar, mtimeMs: stat.mtimeMs, value };
  return value;
}

function countByteBpe(text, tokenizer) {
  let tokens = [...Buffer.from(text, "utf8")];
  const ranks = new Map(tokenizer.merges.map((pair, rank) => [pair[0] + ":" + pair[1], rank]));
  while (tokens.length > 1) {
    let bestRank = Infinity;
    let bestLeft = -1;
    let bestRight = -1;
    for (let index = 0; index < tokens.length - 1; index += 1) {
      const rank = ranks.get(tokens[index] + ":" + tokens[index + 1]);
      if (rank !== undefined && rank < bestRank) {
        bestRank = rank;
        bestLeft = tokens[index];
        bestRight = tokens[index + 1];
      }
    }
    if (bestLeft < 0) break;
    const merged = [];
    for (let index = 0; index < tokens.length; index += 1) {
      if (index + 1 < tokens.length && tokens[index] === bestLeft && tokens[index + 1] === bestRight) {
        merged.push(258 + bestRank);
        index += 1;
      } else {
        merged.push(tokens[index]);
      }
    }
    tokens = merged;
  }
  return tokens.length;
}

export async function tokenizeBrainText(text) {
  if (typeof text !== "string") throw new TypeError("Text must be a string.");
  if (Buffer.byteLength(text, "utf8") > 300_000) {
    throw new Error("Text exceeds the 300 KB local tokenizer limit.");
  }
  const checkpoint = brainCheckpointPath();
  try { await fs.access(checkpoint); }
  catch { throw new Error("No trained SYNTRA brain checkpoint exists. Train the local brain first."); }
  const tokenizer = await readLocalTokenizer();
  const tokenCount = countByteBpe(text, tokenizer);
  const status = await getBrainStatus();
  return {
    token_count: tokenCount,
    context_window_tokens: status.context_tokens,
    vocabulary_size: tokenizer.vocab_size,
    tokenizer: "byte-BPE",
    tokens_over_context: Math.max(0, tokenCount - Math.max(0, status.context_tokens - 1)),
    checkpoint,
    trained_steps: status.trained_steps,
    exact_token_count: true
  };
}

export async function getBrainStatus() {
  const checkpoint = brainCheckpointPath();
  let checkpointExists = false;
  let metadata = {};
  try {
    const stat = await fs.stat(checkpoint);
    checkpointExists = stat.isFile() && stat.size > 0;
  } catch {}
  try {
    const stat = await fs.stat(metadataPath(checkpoint));
    if (stat.isFile() && stat.size <= 128 * 1024) {
      const value = JSON.parse(await fs.readFile(metadataPath(checkpoint), "utf8"));
      if (value && typeof value === "object" && !Array.isArray(value)) metadata = value;
    }
  } catch {}
  return {
    service: "SYNTRA standalone PyTorch brain",
    provider: "syntra-brain-local",
    framework: "PyTorch",
    checkpoint,
    checkpoint_exists: checkpointExists,
    trained_steps: Number(metadata.step || 0),
    epoch: Number(metadata.epoch || 0),
    parameter_count: Number(metadata.parameter_count || 0),
    train_loss: typeof metadata.train_loss === "number" && Number.isFinite(metadata.train_loss) ? metadata.train_loss : null,
    validation_loss: typeof metadata.validation_loss === "number" && Number.isFinite(metadata.validation_loss) ? metadata.validation_loss : null,
    best_validation_loss: typeof metadata.best_validation_loss === "number" && Number.isFinite(metadata.best_validation_loss) ? metadata.best_validation_loss : null,
    config: metadata.config || null,
    context_tokens: Number(metadata.config?.block_size || 0),
    tokenizer_vocab_size: Number(metadata.vocab_size || metadata.config?.vocab_size || 0),
    dataset_files: Array.isArray(metadata.dataset_files) ? metadata.dataset_files.length : 0,
    pretrained_weights: false,
    external_ai_api: false,
    local_only: true,
    is_estimate: false,
    capabilities: { tokenCountIsEstimate: false, maxOutputTokens: 512 },
    training: { ...trainingState }
  };
}

function parseProgressLine(line, callback) {
  const trimmed = line.trim();
  if (!trimmed) return false;
  try {
    const value = JSON.parse(trimmed);
    if (!value || typeof value !== "object" || Array.isArray(value)) return false;
    callback?.(value);
    return true;
  } catch {
    return false;
  }
}

function abortError() {
  const error = new Error("Local model generation was cancelled.");
  error.name = "AbortError";
  return error;
}

export async function generateBrainText({
  prompt,
  maxNewTokens = 512,
  temperature = 0.7,
  topK = 40,
  topP = 1,
  repetitionPenalty = 1.05,
  seed = undefined,
  device = "auto",
  signal = undefined,
  onToken = undefined,
  onMetrics = undefined
}) {
  if (typeof prompt !== "string" || !prompt.trim()) throw new Error("A non-empty prompt is required.");
  if (Buffer.byteLength(prompt, "utf8") > MAX_PROMPT_BYTES) {
    throw new Error("Prompt exceeds the local brain limit of " + MAX_PROMPT_BYTES + " UTF-8 bytes.");
  }
  if (!Number.isInteger(maxNewTokens) || maxNewTokens < 1 || maxNewTokens > MAX_GENERATION_TOKENS) {
    throw new Error("max_new_tokens must be between 1 and " + MAX_GENERATION_TOKENS + ".");
  }
  if (!Number.isFinite(temperature) || temperature < 0 || temperature > 5) {
    throw new Error("temperature must be between 0 and 5.");
  }
  if (!Number.isInteger(topK) || topK < 1 || topK > 8192) {
    throw new Error("top_k must be between 1 and 8,192.");
  }
  if (!Number.isFinite(topP) || topP <= 0 || topP > 1) {
    throw new Error("top_p must be greater than 0 and at most 1.");
  }
  if (!Number.isFinite(repetitionPenalty) || repetitionPenalty < 1 || repetitionPenalty > 2) {
    throw new Error("repetition_penalty must be between 1 and 2.");
  }
  if (signal?.aborted) throw abortError();

  const python = await selectedPython();
  const args = [
    "-m", "brain.generate",
    "--checkpoint", brainCheckpointPath(),
    "--prompt-stdin",
    "--max-new-tokens", String(maxNewTokens),
    "--temperature", String(temperature),
    "--top-k", String(topK),
    "--top-p", String(topP),
    "--repetition-penalty", String(repetitionPenalty),
    "--device", device
  ];
  if (seed !== undefined) args.push("--seed", String(seed));

  return await new Promise((resolve, reject) => {
    const child = spawn(python, args, {
      cwd: ROOT,
      env: runtimeEnvironment(),
      stdio: ["pipe", "pipe", "pipe"],
      windowsHide: true
    });
    const decoder = new StringDecoder("utf8");
    let text = "";
    let stderr = "";
    let stderrPending = "";
    let latestMetrics = {};
    let settled = false;
    const startedAt = Date.now();

    const abort = () => {
      if (!child.killed) child.kill("SIGTERM");
    };
    signal?.addEventListener("abort", abort, { once: true });

    const finish = (error, code, signalName) => {
      if (settled) return;
      settled = true;
      signal?.removeEventListener("abort", abort);
      const tail = decoder.end();
      if (tail) {
        text += tail;
        onToken?.(tail, text, latestMetrics);
      }
      if (signal?.aborted || signalName === "SIGTERM" && signal?.aborted) {
        reject(abortError());
        return;
      }
      if (error || code !== 0) {
        const detail = stderr.trim().slice(-5000);
        reject(new Error(detail || (error instanceof Error ? error.message : "Local model process exited with code " + code + ".")));
        return;
      }
      resolve({
        text,
        tokens_generated: Number(latestMetrics.tokens_generated || 0),
        tokens_per_second: Number(latestMetrics.tokens_per_second || 0),
        finish_reason: latestMetrics.finish_reason || "completed",
        epoch: Number(latestMetrics.epoch || 0),
        step: Number(latestMetrics.checkpoint_step || 0),
        train_loss: latestMetrics.train_loss ?? null,
        validation_loss: latestMetrics.validation_loss ?? null,
        elapsed_ms: Date.now() - startedAt,
        device: latestMetrics.device || device,
        checkpoint: brainCheckpointPath()
      });
    };

    child.stdout.on("data", chunk => {
      const decoded = decoder.write(chunk);
      if (!decoded) return;
      text += decoded;
      onToken?.(decoded, text, latestMetrics);
    });

    child.stderr.on("data", chunk => {
      stderr += chunk.toString("utf8");
      if (stderr.length > 20_000) stderr = stderr.slice(-20_000);
      stderrPending += chunk.toString("utf8");
      let newline;
      while ((newline = stderrPending.indexOf("\n")) >= 0) {
        const line = stderrPending.slice(0, newline);
        stderrPending = stderrPending.slice(newline + 1);
        const parsed = parseProgressLine(line, value => {
          latestMetrics = { ...latestMetrics, ...value };
          onMetrics?.(latestMetrics);
        });
        if (parsed) {
          const last = line.trim();
          if (last) {
            const at = stderr.lastIndexOf(last);
            if (at >= 0) stderr = stderr.slice(0, at) + stderr.slice(at + last.length);
          }
        }
      }
    });

    child.on("error", error => finish(error, null, null));
    child.on("close", (code, signalName) => {
      if (stderrPending.trim()) {
        parseProgressLine(stderrPending, value => {
          latestMetrics = { ...latestMetrics, ...value };
          onMetrics?.(latestMetrics);
        });
      }
      finish(null, code, signalName);
    });
    child.stdin.on("error", () => {});
    child.stdin.end(prompt, "utf8");
  });
}

export function getBrainTrainingStatus() {
  return { ...trainingState };
}

export async function startBrainTraining({
  datasetPaths,
  epochs = 1,
  batchSize = 4,
  blockSize = 256,
  layers = 4,
  heads = 4,
  width = 128,
  vocabSize = 512,
  learningRate = 3e-4,
  device = "auto",
  resume = false
}) {
  if (trainingProcess && trainingState.status === "running") {
    throw new Error("A SYNTRA brain training process is already running.");
  }
  if (!Array.isArray(datasetPaths) || datasetPaths.length < 1 || datasetPaths.length > 32
    || datasetPaths.some(item => typeof item !== "string" || !item.trim())) {
    throw new Error("Select between 1 and 32 local dataset files or project directories.");
  }
  const python = await selectedPython();
  const args = [
    "-m", "brain.train",
    "--checkpoint", brainCheckpointPath(),
    "--epochs", String(epochs),
    "--batch-size", String(batchSize),
    "--block-size", String(blockSize),
    "--layers", String(layers),
    "--heads", String(heads),
    "--width", String(width),
    "--vocab-size", String(vocabSize),
    "--learning-rate", String(learningRate),
    "--device", device
  ];
  for (const dataset of datasetPaths) args.push("--dataset", path.resolve(dataset));
  if (resume) args.push("--resume");

  const child = spawn(python, args, {
    cwd: ROOT,
    env: runtimeEnvironment(),
    stdio: ["ignore", "pipe", "pipe"],
    windowsHide: true
  });
  trainingProcess = child;
  trainingCancelled = false;
  trainingState = {
    status: "running",
    epoch: 0,
    step: 0,
    train_loss: null,
    validation_loss: null,
    tokens_per_second: 0,
    message: "Starting local PyTorch training.",
    updated_at: new Date().toISOString()
  };

  let stdoutPending = "";
  let stderr = "";
  const update = value => {
    trainingState = {
      ...trainingState,
      ...value,
      status: value.status === "completed" ? "completed" : value.status === "failed" ? "failed" : "running",
      updated_at: new Date().toISOString()
    };
  };
  const consumeLine = line => {
    parseProgressLine(line, update);
  };
  child.stdout.on("data", chunk => {
    stdoutPending += chunk.toString("utf8");
    let newline;
    while ((newline = stdoutPending.indexOf("\n")) >= 0) {
      consumeLine(stdoutPending.slice(0, newline));
      stdoutPending = stdoutPending.slice(newline + 1);
    }
  });
  child.stderr.on("data", chunk => {
    stderr += chunk.toString("utf8");
    if (stderr.length > 20_000) stderr = stderr.slice(-20_000);
  });
  child.on("error", error => {
    update({ status: "failed", message: error.message, error: error.message });
    trainingProcess = null;
  });
  child.on("close", (code, signalName) => {
    if (stdoutPending.trim()) consumeLine(stdoutPending);
    if (trainingCancelled) {
      update({ status: "cancelled", message: "Training cancelled. The last atomic checkpoint is preserved.", error: null });
    } else if (code === 0) {
      update({ status: "completed", message: "Local model training completed.", error: null });
    } else {
      const message = stderr.trim().slice(-4000) || "Training process exited with code " + code + (signalName ? " (" + signalName + ")" : "") + ".";
      update({ status: "failed", message, error: message });
    }
    trainingProcess = null;
  });
  return { ...trainingState, pid: child.pid || null };
}

export function cancelBrainTraining() {
  if (!trainingProcess || trainingState.status !== "running") {
    return { ok: false, message: "No local training process is running." };
  }
  trainingCancelled = true;
  trainingProcess.kill("SIGTERM");
  trainingState = {
    ...trainingState,
    status: "cancelling",
    message: "Stopping local training and preserving the last atomic checkpoint.",
    updated_at: new Date().toISOString()
  };
  return { ok: true, message: trainingState.message };
}

export async function getBrainCheckpointForPrompt() {
  const checkpoint = brainCheckpointPath();
  try {
    await fs.access(checkpoint);
  } catch {
    return { exists: false, checkpoint };
  }
  return { exists: true, checkpoint };
}

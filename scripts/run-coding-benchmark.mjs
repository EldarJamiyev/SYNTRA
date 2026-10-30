import crypto from "node:crypto";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { performance } from "node:perf_hooks";

import {
  CODING_BENCHMARK_SCHEMA_VERSION,
  CODING_BENCHMARK_TASKS,
  CODING_BENCHMARK_VERSION,
  validateBenchmarkManifest
} from "../benchmarks/coding-v1/tasks.mjs";
import { evaluateTaskResult, summarizeBenchmark } from "../benchmarks/coding-v1/metrics.mjs";

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const DEFAULT_API = process.env.SYNTRA_BENCHMARK_API || "http://127.0.0.1:3001";
const DEFAULT_TASK_TIMEOUT_MS = 15 * 60 * 1000;
const DEFAULT_CHECK_TIMEOUT_MS = 180 * 1000;
const ORIGIN = process.env.SYNTRA_BENCHMARK_ORIGIN || "http://127.0.0.1:3000";

function parseArguments(argv) {
  const options = {
    api: DEFAULT_API,
    timeoutMs: DEFAULT_TASK_TIMEOUT_MS,
    checkTimeoutMs: DEFAULT_CHECK_TIMEOUT_MS,
    tasks: [],
    keepWorkspaces: false,
    output: ""
  };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--help" || arg === "-h") options.help = true;
    else if (arg === "--keep-workspaces") options.keepWorkspaces = true;
    else if (arg === "--api") options.api = argv[++index] || "";
    else if (arg === "--timeout-seconds") options.timeoutMs = Number(argv[++index]) * 1000;
    else if (arg === "--check-timeout-seconds") options.checkTimeoutMs = Number(argv[++index]) * 1000;
    else if (arg === "--task") options.tasks.push(argv[++index] || "");
    else if (arg === "--output") options.output = argv[++index] || "";
    else throw new Error("Unknown argument: " + arg);
  }
  if (!options.api || !/^https?:\/\/(127\.0\.0\.1|localhost)(:\d+)?$/i.test(options.api)) {
    throw new Error("--api must point to a loopback HTTP service.");
  }
  if (!Number.isFinite(options.timeoutMs) || options.timeoutMs < 10000 || options.timeoutMs > 60 * 60 * 1000) {
    throw new Error("--timeout-seconds must be between 10 and 3600.");
  }
  if (!Number.isFinite(options.checkTimeoutMs) || options.checkTimeoutMs < 1000 || options.checkTimeoutMs > 600000) {
    throw new Error("--check-timeout-seconds must be between 1 and 600.");
  }
  return options;
}

function hash(content) {
  return crypto.createHash("sha256").update(content).digest("hex");
}

async function writeFixture(root, files) {
  for (const [relative, content] of Object.entries(files)) {
    const target = path.join(root, ...relative.split("/"));
    await fs.mkdir(path.dirname(target), { recursive: true });
    await fs.writeFile(target, content, { encoding: "utf8", flag: "wx" });
  }
}

async function hashFiles(root, paths) {
  const result = {};
  for (const relative of paths) {
    try {
      result[relative] = hash(await fs.readFile(path.join(root, ...relative.split("/"))));
    } catch {
      result[relative] = null;
    }
  }
  return result;
}

async function requestJson(baseUrl, route, { method = "GET", body, timeoutMs = 15000 } = {}) {
  const response = await fetch(baseUrl + route, {
    method,
    headers: method === "GET" ? {} : {
      "Content-Type": "application/json",
      Origin: ORIGIN
    },
    body: method === "GET" ? undefined : JSON.stringify(body || {}),
    signal: AbortSignal.timeout(timeoutMs)
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.error || "SYNTRA local API returned HTTP " + response.status + " for " + route + ".");
  }
  return payload;
}

async function readOllamaMemory(modelName) {
  try {
    const response = await fetch("http://127.0.0.1:11434/api/ps", {
      signal: AbortSignal.timeout(1500)
    });
    if (!response.ok) return { available: false, reason: "ollama_ps_http_" + response.status };
    const payload = await response.json();
    const models = Array.isArray(payload.models) ? payload.models : [];
    const model = models.find(item => item.name === modelName || item.model === modelName);
    if (!model) return { available: false, reason: "selected_model_not_reported_as_loaded" };
    return {
      available: true,
      source: "Ollama /api/ps",
      scope: "reported model allocation; size_vram is not total Ollama process RSS",
      model: model.name || model.model || modelName,
      modelBytes: Number.isFinite(Number(model.size)) ? Number(model.size) : null,
      vramBytes: Number.isFinite(Number(model.size_vram)) ? Number(model.size_vram) : null
    };
  } catch (error) {
    return { available: false, reason: error instanceof Error ? error.message : String(error) };
  }
}

async function startMemorySampler() {
  let peakRssBytes = process.memoryUsage().rss;
  const timer = setInterval(() => {
    peakRssBytes = Math.max(peakRssBytes, process.memoryUsage().rss);
  }, 250);
  return {
    stop() {
      clearInterval(timer);
      peakRssBytes = Math.max(peakRssBytes, process.memoryUsage().rss);
      return peakRssBytes;
    }
  };
}

async function waitForChat(baseUrl, jobId, timeoutMs) {
  const started = Date.now();
  let latest = null;
  while (Date.now() - started < timeoutMs) {
    latest = await requestJson(baseUrl, "/local-api/chat/status", {
      method: "POST", body: { jobId }, timeoutMs: 15000
    });
    if (latest.status !== "running") return latest;
    await new Promise(resolve => setTimeout(resolve, 750));
  }
  await requestJson(baseUrl, "/local-api/chat/cancel", {
    method: "POST", body: { jobId }, timeoutMs: 10000
  }).catch(() => {});
  throw new Error("Coding task exceeded the configured timeout of " + Math.round(timeoutMs / 1000) + " seconds.");
}

async function runOneTask(task, options, runStartedAt) {
  const workspace = await fs.mkdtemp(path.join(os.tmpdir(), "syntra-" + task.id + "-"));
  const startedAt = new Date().toISOString();
  const startedClock = performance.now();
  const sampler = await startMemorySampler();
  let keepPath = null;
  let result = {
    taskId: task.id, category: task.category, title: task.title,
    startedAt, workspace: null, status: "failed", error: null,
    model: null, metrics: null
  };
  try {
    await writeFixture(workspace, task.files);
    const protectedPaths = [...new Set([...(task.protectedFiles || []), task.sourceFile])];
    const beforeHashes = await hashFiles(workspace, protectedPaths);
    await requestJson(options.api, "/local-api/workspace/switch", {
      method: "POST", body: { root: workspace }
    });
    const modelStatus = await requestJson(options.api, "/local-api/ai/status", { timeoutMs: 10000 });
    if (!modelStatus.checkpoint_exists) {
      throw new Error("No usable local model is available. Install a local Ollama model or start SYNTRA's trained native model before running the coding benchmark.");
    }
    const modelMemoryBefore = modelStatus.provider === "ollama-local"
      ? await readOllamaMemory(modelStatus.model_name)
      : { available: false, reason: "native_model_process_memory_is_not_exposed_by_the_current_api" };

    const chatStart = await requestJson(options.api, "/local-api/chat", {
      method: "POST",
      body: { message: task.prompt, history: [] },
      timeoutMs: 15000
    });
    const chat = await waitForChat(options.api, chatStart.jobId, options.timeoutMs);
    if (chat.status !== "completed") {
      throw new Error(chat.error || chat.reply || "SYNTRA coding task did not complete successfully.");
    }
    if (!chat.agentRunId) {
      throw new Error("The chat completed without a persisted agent run ID; check that the task was routed to the coding agent.");
    }

    // The API keeps one active workspace. Re-select this isolated fixture before reading
    // its persisted run or issuing the independent verification call.
    await requestJson(options.api, "/local-api/workspace/switch", {
      method: "POST", body: { root: workspace }
    });
    const agentResponse = await requestJson(options.api, "/local-api/agent/runs/get", {
      method: "POST", body: { runId: chat.agentRunId }
    });
    const agentRun = agentResponse.run;
    if (!agentRun) throw new Error("The agent run record could not be retrieved after completion.");

    const independentCheck = await requestJson(options.api, "/local-api/tools/execute", {
      method: "POST",
      body: {
        root: workspace,
        name: "workspace.check",
        input: { task: "test", timeoutMs: options.checkTimeoutMs },
        approval: true
      },
      timeoutMs: options.checkTimeoutMs + 15000
    });

    const afterHashes = await hashFiles(workspace, protectedPaths);
    const protectedIntegrity = Object.fromEntries((task.protectedFiles || []).map(relative => [
      relative, beforeHashes[relative] !== null && beforeHashes[relative] === afterHashes[relative]
    ]));
    const modelMemoryAfter = modelStatus.provider === "ollama-local"
      ? await readOllamaMemory(modelStatus.model_name)
      : modelMemoryBefore;
    const elapsedMs = Math.max(0, Math.round(performance.now() - startedClock));
    const peakControllerRssBytes = sampler.stop();
    const measured = evaluateTaskResult({
      task,
      agentRun,
      independentCheck,
      protectedIntegrity,
      sourceChanged: beforeHashes[task.sourceFile] !== null
        && beforeHashes[task.sourceFile] !== afterHashes[task.sourceFile],
      elapsedMs,
      peakControllerRssBytes,
      modelMemory: { before: modelMemoryBefore, after: modelMemoryAfter }
    });
    result = {
      ...result,
      status: measured.taskCompleted ? "passed" : "failed",
      workspace: options.keepWorkspaces ? workspace : null,
      finishedAt: new Date().toISOString(),
      model: {
        provider: modelStatus.provider || "unknown",
        name: modelStatus.model_name || "unknown",
        contextWindowTokens: modelStatus.context_tokens || modelStatus.context_window_tokens || null,
        tokenizer: modelStatus.tokenizer || null,
        pretrainedWeights: modelStatus.pretrained_weights === true,
        trainedSteps: modelStatus.trained_steps ?? null,
        externalAiApi: modelStatus.external_ai_api === true
      },
      metrics: measured,
      agentRun: {
        id: agentRun.id,
        status: agentRun.status,
        verification: agentRun.verification,
        attempts: agentRun.attempts,
        changedFiles: (agentRun.changedFiles || []).map(file => file.path),
        selectedContextCount: (agentRun.selectedContext || []).length,
        toolCallCount: (agentRun.toolCalls || []).length
      },
      independentCheck: {
        ok: independentCheck.ok === true,
        status: independentCheck.status || null,
        passed: independentCheck.output?.passed === true,
        exitCode: independentCheck.output?.exitCode ?? null,
        durationMs: independentCheck.output?.durationMs ?? null,
        error: independentCheck.error || null
      },
      protectedIntegrity
    };
    return result;
  } catch (error) {
    const elapsedMs = Math.max(0, Math.round(performance.now() - startedClock));
    const peakControllerRssBytes = sampler.stop();
    result = {
      ...result,
      status: "error",
      workspace: options.keepWorkspaces ? workspace : null,
      finishedAt: new Date().toISOString(),
      error: error instanceof Error ? error.message : String(error),
      metrics: {
        elapsedMs,
        peakControllerRssBytes,
        modelProcessMemoryBytes: null,
        modelProcessMemoryNote: "Not reported by the current local API."
      }
    };
    return result;
  } finally {
    if (options.keepWorkspaces) keepPath = workspace;
    if (!keepPath) await fs.rm(workspace, { recursive: true, force: true }).catch(() => {});
  }
}

async function main() {
  const options = parseArguments(process.argv.slice(2));
  if (options.help) {
    process.stdout.write([
      "SYNTRA versioned local coding benchmark",
      "",
      "Usage: npm run benchmark:coding -- [--task ID] [--api http://127.0.0.1:3001]",
      "       [--timeout-seconds 900] [--check-timeout-seconds 180]",
      "       [--keep-workspaces] [--output path]",
      "",
      "Requires the SYNTRA local workspace service and an installed local model.",
      "Tasks run sequentially in disposable temporary workspaces. Benchmark fixture tests are immutable."
    ].join("\n") + "\n");
    return 0;
  }
  validateBenchmarkManifest();
  const selected = options.tasks.length
    ? CODING_BENCHMARK_TASKS.filter(task => options.tasks.includes(task.id))
    : CODING_BENCHMARK_TASKS;
  const unknown = options.tasks.filter(id => !CODING_BENCHMARK_TASKS.some(task => task.id === id));
  if (unknown.length) throw new Error("Unknown benchmark task ID(s): " + unknown.join(", "));
  if (!selected.length) throw new Error("No benchmark tasks were selected.");

  const health = await requestJson(options.api, "/local-api/health");
  if (health.status !== "ok" || health.service !== "SYNTRA local chat and coding agent") {
    throw new Error("The selected loopback endpoint is not a healthy SYNTRA workspace service.");
  }

  const benchmarkStartedAt = new Date().toISOString();
  const startedClock = performance.now();
  const results = [];
  for (const task of selected) {
    process.stdout.write("\n[" + task.id + "] " + task.title + "\n");
    const item = await runOneTask(task, options, startedClock);
    results.push(item);
    process.stdout.write("  Result: " + item.status + "\n");
    if (item.error) process.stdout.write("  Detail: " + item.error + "\n");
    if (item.metrics) {
      process.stdout.write("  Latency: " + (item.metrics.elapsedMs ?? 0) + " ms");
      if (Number.isFinite(item.metrics.totalTokens)) process.stdout.write(" | tokens: " + item.metrics.totalTokens);
      process.stdout.write("\n");
    }
  }

  const finishedAt = new Date().toISOString();
  const summary = summarizeBenchmark(results);
  const report = {
    schemaVersion: CODING_BENCHMARK_SCHEMA_VERSION,
    benchmarkVersion: CODING_BENCHMARK_VERSION,
    runId: crypto.randomUUID(),
    startedAt: benchmarkStartedAt,
    finishedAt,
    durationMs: Math.max(0, Math.round(performance.now() - startedClock)),
    environment: {
      node: process.version,
      platform: process.platform,
      architecture: process.arch,
      cpus: os.cpus().length,
      totalMemoryBytes: os.totalmem(),
      freeMemoryAtStartBytes: os.freemem(),
      api: options.api,
      runner: "SYNTRA local coding agent API"
    },
    methodology: {
      taskCount: selected.length,
      disposableWorkspaces: true,
      benchmarkTestsMutableByAgent: false,
      independentPostRunTest: true,
      modelCalls: "Real calls to the selected local Ollama model or trained SYNTRA native checkpoint.",
      controllerMemory: "Peak RSS of this benchmark runner process only; it is not model/server memory.",
      modelMemory: "Ollama model and VRAM allocation are recorded from /api/ps when available. Native model process memory is not exposed and is reported as unavailable.",
      completionRule: "A task passes only when the agent verification passes, an independent post-run test passes, the target source changes, and protected fixture files remain unchanged."
    },
    summary,
    results
  };
  const output = options.output
    ? path.resolve(options.output)
    : path.join(REPO_ROOT, "benchmarks", "results", "coding-v1-" + finishedAt.replace(/[:.]/g, "-") + ".json");
  await fs.mkdir(path.dirname(output), { recursive: true });
  await fs.writeFile(output, JSON.stringify(report, null, 2) + "\n", "utf8");
  process.stdout.write("\nBenchmark report: " + output + "\n");
  process.stdout.write("Completed: " + summary.completed + "/" + summary.taskCount + " tasks\n");
  process.stdout.write("Patch valid: " + summary.patchValid + "/" + summary.taskCount
    + " | independent tests passed: " + summary.independentTestsPassed + "/" + summary.taskCount + "\n");
  return summary.completed === summary.taskCount ? 0 : 2;
}

main().then(code => {
  process.exitCode = code;
}).catch(error => {
  process.stderr.write("Benchmark failed: " + (error instanceof Error ? error.message : String(error)) + "\n");
  process.exitCode = 1;
});

import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import { fileURLToPath } from "node:url";
import { performance } from "node:perf_hooks";
import { analyzeCheckFailure, runAgentTask } from "./agent.mjs";
const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
export const DEFAULT_BENCHMARK_SUITE = path.join(REPO_ROOT, "benchmarks", "suite.v1.json");
const OUTPUT_LIMIT = 12000;
function isInside(parent, target) {
  const relative = path.relative(parent, target);
  return relative !== ".." && !relative.startsWith(".." + path.sep) && !path.isAbsolute(relative);
}
function relativePath(value) { return String(value || "").replace(/\\/g, "/").replace(/^\.\/+/, ""); }

export function validateBenchmarkSuite(suite) {
  if (!suite || typeof suite !== "object" || Array.isArray(suite)) throw new Error("Benchmark suite must be a JSON object.");
  if (typeof suite.version !== "string" || !/^\d+\.\d+\.\d+$/.test(suite.version)) throw new Error("Benchmark suite version must use semantic version format.");
  if (!Array.isArray(suite.tasks) || suite.tasks.length < 1 || suite.tasks.length > 100) throw new Error("Benchmark suite must contain between 1 and 100 tasks.");
  const ids = new Set();
  for (const task of suite.tasks) {
    if (!task || typeof task !== "object" || Array.isArray(task)) throw new Error("Each benchmark task must be an object.");
    if (typeof task.id !== "string" || !/^[a-z0-9][a-z0-9-]{1,63}$/.test(task.id) || ids.has(task.id)) throw new Error("Benchmark task IDs must be unique lowercase kebab-case identifiers.");
    ids.add(task.id);
    if (typeof task.title !== "string" || !task.title.trim() || task.title.length > 120) throw new Error("Benchmark task " + task.id + " has an invalid title.");
    if (typeof task.goal !== "string" || !task.goal.trim() || task.goal.length > 5000) throw new Error("Benchmark task " + task.id + " has an invalid goal.");
    if (typeof task.fixture !== "string" || !task.fixture.trim() || path.isAbsolute(task.fixture) || task.fixture.split(/[\\/]+/).some(part => part === ".." || part === ".")) throw new Error("Benchmark task " + task.id + " has an unsafe fixture path.");
    if (!["node-script", "python-unittest", "go-test", "rust-test"].includes(task.baselineCheck)) {
      throw new Error("Benchmark task " + task.id + " uses an unsupported baseline check.");
    }
    if (typeof task.testEntry !== "string" || !task.testEntry.trim() || path.isAbsolute(task.testEntry)
      || task.testEntry.split(/[\\/]+/).some(part => part === ".." || part === ".")) {
      throw new Error("Benchmark task " + task.id + " has an unsafe test-entry path.");
    }
    if (!Array.isArray(task.expectedChangedPaths) || task.expectedChangedPaths.length < 1 || task.expectedChangedPaths.length > 10) throw new Error("Benchmark task " + task.id + " must declare expected changed paths.");
    for (const value of task.expectedChangedPaths) {
      if (typeof value !== "string" || !value || path.isAbsolute(value) || value.split(/[\\/]+/).some(part => part === ".." || part === ".")) throw new Error("Benchmark task " + task.id + " contains an unsafe expected changed path.");
    }
    task.maxIterations = Math.max(1, Math.min(6, Number(task.maxIterations) || 2));
    task.maxSlices = Math.max(1, Math.min(8, Number(task.maxSlices) || 1));
  }
  return suite;
}

export async function loadBenchmarkSuite(filePath = DEFAULT_BENCHMARK_SUITE) {
  const raw = await fs.readFile(filePath, "utf8");
  let suite;
  try { suite = JSON.parse(raw); } catch (error) {
    throw new Error("Benchmark suite is not valid JSON: " + (error instanceof Error ? error.message : String(error)));
  }
  validateBenchmarkSuite(suite);
  return { suite, suiteDir: path.dirname(path.resolve(filePath)) };
}
function boundedOutput(current, chunk) {
  const next = current + chunk.toString("utf8");
  return next.length > OUTPUT_LIMIT ? next.slice(0, OUTPUT_LIMIT) : next;
}
export async function runNodeBaselineCheck(cwd, testEntry, timeoutMs = 30000) {
  const root = await fs.realpath(cwd);
  const requestedEntry = path.resolve(root, testEntry);
  if (!isInside(root, requestedEntry)) throw new Error("Benchmark test entry escapes its fixture workspace.");
  const entry = await fs.realpath(requestedEntry);
  if (!isInside(root, entry) || !(await fs.stat(entry)).isFile()) throw new Error("Benchmark test entry is missing or resolves outside its fixture workspace.");
  const entryRelative = path.relative(root, entry);
  return new Promise(resolve => {
    let stdout = "", stderr = "", timedOut = false, settled = false, child;
    try {
      child = spawn(process.execPath, [entryRelative], { cwd: root, shell: false, windowsHide: true, stdio: ["ignore", "pipe", "pipe"] });
    } catch (error) {
      resolve({ passed: false, timedOut: false, exitCode: null, error: error instanceof Error ? error.message : String(error) });
      return;
    }
    const timer = setTimeout(() => { timedOut = true; child.kill(); }, timeoutMs);
    child.stdout.on("data", chunk => { stdout = boundedOutput(stdout, chunk); });
    child.stderr.on("data", chunk => { stderr = boundedOutput(stderr, chunk); });
    child.on("error", error => {
      if (settled) return;
      settled = true; clearTimeout(timer);
      resolve({ passed: false, timedOut, exitCode: null, error: error.message, stdout, stderr });
    });
    child.on("close", code => {
      if (settled) return;
      settled = true; clearTimeout(timer);
      resolve({ passed: code === 0 && !timedOut, timedOut, exitCode: code, stdout, stderr });
    });
  });
}

export async function runBenchmarkBaselineCheck(cwd, task, timeoutMs = 30000) {
  if (!task || !["node-script", "python-unittest", "go-test", "rust-test"].includes(task.baselineCheck)) {
    throw new Error("Unsupported benchmark baseline runner.");
  }
  if (task.baselineCheck === "node-script") return runNodeBaselineCheck(cwd, task.testEntry, timeoutMs);
  const root = await fs.realpath(cwd);
  const requestedEntry = path.resolve(root, task.testEntry);
  if (!isInside(root, requestedEntry)) throw new Error("Benchmark test entry escapes its fixture workspace.");
  const entry = await fs.realpath(requestedEntry);
  if (!isInside(root, entry) || !(await fs.stat(entry)).isFile()) {
    throw new Error("Benchmark test entry is missing or resolves outside its fixture workspace.");
  }
  const relative = path.relative(root, entry);
  const commands = {
    "python-unittest": process.platform === "win32"
      ? ["python", ["-m", "unittest", "discover", "-s", path.dirname(relative) || ".", "-p", path.basename(relative), "-v"]]
      : ["python3", ["-m", "unittest", "discover", "-s", path.dirname(relative) || ".", "-p", path.basename(relative), "-v"]],
    "go-test": ["go", ["test", "./..."]],
    "rust-test": ["cargo", ["test", "--offline"]]
  };
  const [executable, args] = commands[task.baselineCheck];
  return await new Promise(resolve => {
    let stdout = "", stderr = "", timedOut = false, settled = false, child;
    try {
      child = spawn(executable, args, { cwd: root, shell: false, windowsHide: true, stdio: ["ignore", "pipe", "pipe"] });
    } catch (error) {
      resolve({ passed: false, skipped: error?.code === "ENOENT", timedOut: false, exitCode: null, error: String(error?.message || error) });
      return;
    }
    const timer = setTimeout(() => { timedOut = true; child.kill(); }, timeoutMs);
    child.stdout.on("data", chunk => { stdout = boundedOutput(stdout, chunk); });
    child.stderr.on("data", chunk => { stderr = boundedOutput(stderr, chunk); });
    child.on("error", error => {
      if (settled) return;
      settled = true; clearTimeout(timer);
      resolve({ passed: false, skipped: error?.code === "ENOENT", timedOut, exitCode: null, error: error.message, stdout, stderr });
    });
    child.on("close", code => {
      if (settled) return;
      settled = true; clearTimeout(timer);
      resolve({ passed: code === 0 && !timedOut, skipped: false, timedOut, exitCode: code, stdout, stderr });
    });
  });
}

export function evaluateBenchmarkTask(task, agentResult, metrics = {}) {
  const skipped = metrics.skipped === true;
  const actualPaths = Array.isArray(agentResult?.changedFiles) ? agentResult.changedFiles.map(item => relativePath(item?.path)).filter(Boolean).sort() : [];
  const expectedPaths = task.expectedChangedPaths.map(relativePath).sort();
  const exactPaths = actualPaths.length === expectedPaths.length && actualPaths.every((item, index) => item === expectedPaths[index]);
  const patchValid = actualPaths.length > 0 && exactPaths;
  const verificationPassed = agentResult?.verification?.passed === true;
  const attempts = Array.isArray(agentResult?.attempts) ? agentResult.attempts : [];
  const attemptCount = attempts.length;
  const repairAttempts = Math.max(0, attemptCount - 1);
  const promptTokens = attempts.reduce((sum, item) => sum + Math.max(0, Number(item?.promptTokens) || 0), 0);
  const outputTokens = attempts.reduce((sum, item) => sum + Math.max(0, Number(item?.outputTokens) || 0), 0);
  const taskComplete = !skipped && patchValid && verificationPassed;
  const diagnosisCategory = agentResult?.verification?.diagnosis?.category
    || analyzeCheckFailure(agentResult?.verification?.checks || [])?.category
    || null;
  const failureCategory = skipped
    ? "skipped"
    : taskComplete
      ? null
      : !patchValid
        ? "invalid_or_incomplete_patch"
        : (diagnosisCategory || "verification_failed");
  return {
    id: task.id, category: String(task.category || "general"), title: task.title,
    agentStatus: skipped ? "skipped" : String(agentResult?.status || "unknown"),
    skipped,
    patchValid: !skipped && patchValid,
    verificationPassed: !skipped && verificationPassed,
    taskComplete,
    attemptCount,
    repairAttempts,
    failureCategory,
    expectedChangedPaths: expectedPaths, changedPaths: actualPaths,
    verificationStatus: String(agentResult?.verification?.status || "not_run"),
    latencyMs: Math.max(0, Math.round(Number(metrics.latencyMs) || 0)),
    tokens: { prompt: promptTokens, output: outputTokens, total: promptTokens + outputTokens },
    peakProcessRssBytes: Math.max(0, Math.round(Number(metrics.peakProcessRssBytes) || 0)),
    ...(metrics.error ? { error: String(metrics.error).slice(0, 1000) } : {})
  };
}
function median(values) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b), middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : Math.round((sorted[middle - 1] + sorted[middle]) / 2);
}
export function summarizeBenchmarkResults(results) {
  const count = results.length;
  const executed = results.filter(item => item.skipped !== true);
  const executedCount = executed.length;
  const skippedCount = count - executedCount;
  const latencies = executed.map(item => item.latencyMs).filter(Number.isFinite);
  const promptTokens = executed.reduce((sum, item) => sum + item.tokens.prompt, 0);
  const outputTokens = executed.reduce((sum, item) => sum + item.tokens.output, 0);
  const patchCount = executed.filter(item => item.patchValid).length;
  const verificationCount = executed.filter(item => item.verificationPassed).length;
  const completeCount = executed.filter(item => item.taskComplete).length;
  const categoryGroups = new Map();
  for (const item of results) {
    const category = String(item.category || "uncategorized").slice(0, 64);
    const group = categoryGroups.get(category) || [];
    group.push(item);
    categoryGroups.set(category, group);
  }
  const categories = Object.fromEntries([...categoryGroups.entries()].sort(([a], [b]) => a.localeCompare(b)).map(([category, items]) => {
    const measured = items.filter(item => item.skipped !== true);
    const total = measured.length;
    const valid = measured.filter(item => item.patchValid).length;
    const verified = measured.filter(item => item.verificationPassed).length;
    const complete = measured.filter(item => item.taskComplete).length;
    return [category, {
      taskCount: total,
      patchValidityRate: total ? Number((valid / total).toFixed(4)) : 0,
      verificationPassRate: total ? Number((verified / total).toFixed(4)) : 0,
      taskCompletionRate: total ? Number((complete / total).toFixed(4)) : 0
    }];
  }));
  return {
    taskCount: count, executedTaskCount: executedCount, skippedTaskCount: skippedCount,
    patchValidCount: patchCount, verificationPassedCount: verificationCount, completedTaskCount: completeCount,
    categories,
    patchValidityRate: executedCount ? Number((patchCount / executedCount).toFixed(4)) : 0,
    verificationPassRate: executedCount ? Number((verificationCount / executedCount).toFixed(4)) : 0,
    taskCompletionRate: executedCount ? Number((completeCount / executedCount).toFixed(4)) : 0,
    latencyMs: {
      mean: latencies.length ? Math.round(latencies.reduce((sum, value) => sum + value, 0) / latencies.length) : 0,
      median: median(latencies), min: latencies.length ? Math.min(...latencies) : 0, max: latencies.length ? Math.max(...latencies) : 0
    },
    tokens: { prompt: promptTokens, output: outputTokens, total: promptTokens + outputTokens },
    peakProcessRssBytes: results.reduce((maximum, item) => Math.max(maximum, item.peakProcessRssBytes), 0)
  };
}
async function createFixtureWorkspace(suiteDir, task) {
  const suiteRoot = await fs.realpath(suiteDir);
  const requested = path.resolve(suiteRoot, task.fixture);
  if (!isInside(suiteRoot, requested)) throw new Error("Benchmark fixture path escapes the suite directory.");
  const source = await fs.realpath(requested);
  if (!isInside(suiteRoot, source)) throw new Error("Benchmark fixture resolves outside the suite directory.");
  if (!(await fs.stat(source)).isDirectory()) throw new Error("Benchmark fixture is not a directory: " + task.fixture);
  const parent = await fs.mkdtemp(path.join(os.tmpdir(), "syntra-benchmark-"));
  const workspace = path.join(parent, "workspace");
  try {
    await fs.cp(source, workspace, { recursive: true, errorOnExist: true, force: false });
    return { parent, workspace };
  } catch (error) {
    await fs.rm(parent, { recursive: true, force: true }).catch(() => {});
    throw error;
  }
}

export async function runBenchmarkSuite({ suite, suiteDir, taskIds = null, modelRequest, agentRunner = runAgentTask, onProgress = () => {}, outputPath = null }) {
  validateBenchmarkSuite(suite);
  if (typeof modelRequest !== "function") throw new Error("A local model request bridge is required.");
  const requested = taskIds ? new Set(taskIds) : null;
  const selected = suite.tasks.filter(task => !requested || requested.has(task.id));
  if (!selected.length) throw new Error("No benchmark tasks matched the requested selection.");
  if (requested && selected.length !== requested.size) {
    const missing = [...requested].filter(id => !suite.tasks.some(task => task.id === id));
    throw new Error("Unknown benchmark task ID(s): " + missing.join(", "));
  }
  const modelStatus = await modelRequest("GET", "/model/status", undefined, 15000);
  if (!modelStatus?.checkpoint_exists) throw new Error("No trained SYNTRA checkpoint is available. Complete native model training before running benchmarks.");
  const startedAt = new Date().toISOString(), results = [];
  let selectedModel = null;
  for (const task of selected) {
    let fixture = null, agentResult = null, taskError = null, progressModel = null;
    const started = performance.now();
    let peakRss = process.memoryUsage().rss;
    const sampler = setInterval(() => { peakRss = Math.max(peakRss, process.memoryUsage().rss); }, 100);
    try {
      fixture = await createFixtureWorkspace(suiteDir, task);
      const baseline = await runBenchmarkBaselineCheck(fixture.workspace, task);
      if (baseline.skipped) {
        taskError = "Required benchmark runtime is not installed: " + task.baselineCheck + ". " + String(baseline.error || "");
      } else {
        if (baseline.timedOut || baseline.passed || baseline.exitCode === null) {
          throw new Error("Fixture baseline is invalid: its " + task.baselineCheck + " assertions must fail before the requested change and finish within 30 seconds.");
        }
        await onProgress({ taskId: task.id, phase: "started", message: "Running the actual local coding agent against an isolated fixture." });
        agentResult = await agentRunner({
          root: fixture.workspace, goal: task.goal, requestedPath: task.expectedChangedPaths[0],
          maxIterations: task.maxIterations, maxSlices: task.maxSlices,
          runChecks: true, modelToolSelection: true, modelSelfReview: true, modelRequest,
          onProgress: async update => {
            if (update?.model) progressModel = update.model;
            await onProgress({ taskId: task.id, ...update });
          }
        });
        if (progressModel && !selectedModel) selectedModel = progressModel;
      }
    } catch (error) {
      taskError = error instanceof Error ? error.message : String(error);
    } finally {
      clearInterval(sampler);
      peakRss = Math.max(peakRss, process.memoryUsage().rss);
      if (fixture?.parent) await fs.rm(fixture.parent, { recursive: true, force: true }).catch(() => {});
    }
    const result = evaluateBenchmarkTask(task, agentResult, {
      latencyMs: performance.now() - started, peakProcessRssBytes: peakRss, error: taskError,
      skipped: !agentResult && /Required benchmark runtime is not installed/.test(taskError || "")
    });
    if (!agentResult && result.skipped) {
      result.patchValid = false; result.verificationPassed = false;
      result.taskComplete = false; result.agentStatus = "skipped";
    } else if (!agentResult) {
      result.patchValid = false; result.verificationPassed = false;
      result.taskComplete = false; result.agentStatus = "error";
    }
    results.push(result);
    await onProgress({
      taskId: task.id, phase: "finished",
      message: result.skipped ? "Skipped because the required language runtime is not installed."
        : result.taskComplete ? "Expected patch and candidate verification both passed."
          : "Task did not meet every benchmark acceptance criterion.",
      result
    });
  }
  const report = {
    schemaVersion: 1, suiteVersion: suite.version, runId: "bench-" + randomUUID(),
    startedAt, completedAt: new Date().toISOString(),
    runtime: { node: process.version, platform: process.platform, architecture: process.arch },
    hardware: {
      cpuModel: os.cpus()[0]?.model || "unknown",
      logicalCores: os.cpus().length,
      totalMemoryBytes: os.totalmem()
    },
    model: {
      engine: "syntra-native",
      name: String(selectedModel?.name || "SYNTRA native Transformer"),
      contextWindowTokens: Number(selectedModel?.contextWindowTokens || modelStatus.context_tokens || 0) || null
    },
    summary: summarizeBenchmarkResults(results), tasks: results,
    measurementNotes: [
      "Token counts come from SYNTRA's saved byte-BPE tokenizer and checkpoint runtime.",
      "Peak RSS measures the SYNTRA benchmark Node.js process only, not the separate native AI service, GPU, or total system memory.",
      "Task completion requires the exact expected changed-file set and passing candidate verification.",
      "Python, Go, or Rust fixtures are explicitly skipped when their required runtime is unavailable and excluded from completion-rate denominators.",
      "Each task uses a disposable fixture copy. The original fixture and active workspace are not modified."
    ]
  };
  let savedPath = null;
  if (outputPath) {
    savedPath = path.resolve(outputPath);
    await fs.mkdir(path.dirname(savedPath), { recursive: true });
    await fs.writeFile(savedPath, JSON.stringify(report, null, 2) + "\n", { encoding: "utf8", flag: "wx" });
  }
  return { report, outputPath: savedPath };
}

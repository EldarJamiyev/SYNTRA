import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import test from "node:test";
import { compareBenchmarkReports, validateBenchmarkReport } from "../local/benchmark-comparison.mjs";

function report({ runId, model, completion, patch, verification, latency, tokens, rss, taskComplete, platform = "win32", cpu = "Test CPU", memory = 16 * 1024 ** 3 }) {
  const task = {
    id: "utility-clamp", title: "Clamp", patchValid: patch,
    verificationPassed: verification, taskComplete, latencyMs: latency,
    tokens: { prompt: tokens, output: 10, total: tokens + 10 },
    peakProcessRssBytes: rss
  };
  return {
    schemaVersion: 1, suiteVersion: "1.0.0", runId, model: { provider: "ollama", name: model },
    startedAt: "2026-10-31T10:00:00.000Z",
    runtime: { node: "v24.16.0", platform, architecture: "x64" },
    hardware: { cpuModel: cpu, logicalCores: 8, totalMemoryBytes: memory },
    summary: {
      taskCount: 1, completedTaskCount: taskComplete ? 1 : 0,
      taskCompletionRate: completion, patchValidityRate: patch ? 1 : 0,
      verificationPassRate: verification ? 1 : 0,
      latencyMs: { mean: latency, median: latency, min: latency, max: latency },
      tokens: { prompt: tokens, output: 10, total: tokens + 10 },
      peakProcessRssBytes: rss
    },
    tasks: [task]
  };
}

test("report validation rejects malformed schema, duplicate tasks, and invalid metrics", () => {
  assert.throws(() => validateBenchmarkReport({}), /schema version/i);
  const valid = report({ runId: "a", model: "model-a", completion: 1, patch: true, verification: true, latency: 100, tokens: 50, rss: 1000, taskComplete: true });
  assert.equal(validateBenchmarkReport(valid), valid);
  assert.throws(() => validateBenchmarkReport({ ...valid, tasks: [...valid.tasks, valid.tasks[0]] }), /duplicate task/i);
  assert.throws(() => validateBenchmarkReport({ ...valid, tasks: [{ ...valid.tasks[0], latencyMs: -1 }] }), /latency/i);
  assert.throws(() => validateBenchmarkReport({ ...valid, summary: { ...valid.summary, taskCompletionRate: 1.5 } }), /summary/i);
  assert.throws(() => validateBenchmarkReport({ ...valid, tasks: [{ ...valid.tasks[0], taskComplete: false }] }), /inconsistent completion/i);
  assert.throws(() => validateBenchmarkReport({
    ...valid,
    tasks: [{ ...valid.tasks[0], baselineValidated: false }]
  }), /inconsistent completion/i,
  "legacy-shaped reports remain readable, but a reported invalid baseline cannot be marked complete");
});

test("1.4 benchmark reports require explicit baseline validation evidence", () => {
  const legacy = report({
    runId: "legacy-1", model: "model-a", completion: 1, patch: true,
    verification: true, latency: 100, tokens: 50, rss: 1000, taskComplete: true
  });
  const current = { ...legacy, suiteVersion: "1.4.0" };
  assert.throws(() => validateBenchmarkReport(current), /missing required baseline validation evidence/i);
  const unvalidated = {
    ...current,
    tasks: [{ ...current.tasks[0], baselineValidated: false }]
  };
  assert.throws(() => validateBenchmarkReport(unvalidated), /inconsistent completion/i);
  const validated = {
    ...current,
    tasks: [{ ...current.tasks[0], baselineValidated: true }]
  };
  assert.equal(validateBenchmarkReport(validated), validated);
});

test("comparison identifies model improvements and task-level changes", () => {
  const baseline = report({
    runId: "baseline-1", model: "qwen2.5-coder:14b",
    completion: 0, patch: false, verification: false, latency: 1000,
    tokens: 1000, rss: 300000, taskComplete: false
  });
  const candidate = report({
    runId: "candidate-1", model: "qwen2.5-coder:32b",
    completion: 1, patch: true, verification: true, latency: 800,
    tokens: 750, rss: 280000, taskComplete: true
  });
  const result = compareBenchmarkReports(baseline, candidate);
  assert.equal(result.status, "improved");
  assert.equal(result.environment.runtimeComparable, true);
  assert.equal(result.environment.hardwareComparable, true);
  assert.equal(result.metrics.find(item => item.key === "taskCompletionRate").direction, "improved");
  assert.equal(result.metrics.find(item => item.key === "meanLatencyMs").direction, "improved");
  assert.equal(result.tasks[0].status, "improved");
  assert.equal(result.tasks[0].latency.direction, "improved");
});

test("comparison flags task and metric regressions", () => {
  const baseline = report({
    runId: "baseline-2", model: "model-a", completion: 1, patch: true,
    verification: true, latency: 500, tokens: 500, rss: 100000, taskComplete: true
  });
  const candidate = report({
    runId: "candidate-2", model: "model-b", completion: 0, patch: true,
    verification: false, latency: 700, tokens: 600, rss: 150000, taskComplete: false
  });
  const result = compareBenchmarkReports(baseline, candidate);
  assert.equal(result.status, "regressed");
  assert.equal(result.tasks[0].status, "regressed");
  assert.equal(result.metrics.find(item => item.key === "verificationPassRate").direction, "regressed");
});

test("comparison rejects different suite versions or task selections", () => {
  const baseline = report({
    runId: "baseline-3", model: "model-a", completion: 1, patch: true,
    verification: true, latency: 500, tokens: 500, rss: 100000, taskComplete: true
  });
  assert.throws(() => compareBenchmarkReports(baseline, { ...baseline, suiteVersion: "2.0.0" }), /suite versions/i);
  assert.throws(() => compareBenchmarkReports(baseline, { ...baseline, tasks: [{ ...baseline.tasks[0], id: "different-task" }] }), /different task sets/i);
});

test("comparison marks latency and RSS incomparable when machine environments differ", () => {
  const baseline = report({
    runId: "baseline-4", model: "model-a", completion: 1, patch: true,
    verification: true, latency: 500, tokens: 500, rss: 100000, taskComplete: true
  });
  const candidate = report({
    runId: "candidate-4", model: "model-b", completion: 1, patch: true,
    verification: true, latency: 300, tokens: 400, rss: 90000, taskComplete: true,
    platform: "linux", cpu: "Different CPU", memory: 32 * 1024 ** 3
  });
  const result = compareBenchmarkReports(baseline, candidate);
  assert.equal(result.status, "improved");
  assert.equal(result.environment.runtimeComparable, false);
  assert.equal(result.environment.hardwareComparable, false);
  assert.equal(result.metrics.find(item => item.key === "meanLatencyMs").direction, "incomparable");
  assert.equal(result.metrics.find(item => item.key === "peakProcessRssBytes").direction, "incomparable");
  assert.ok(result.warnings.length >= 2);
});

test("comparison treats missing hardware metadata as unverified and avoids performance claims", () => {
  const baseline = report({
    runId: "baseline-no-hardware", model: "model-a", completion: 1, patch: true,
    verification: true, latency: 500, tokens: 500, rss: 100000, taskComplete: true
  });
  const candidate = report({
    runId: "candidate-no-hardware", model: "model-b", completion: 1, patch: true,
    verification: true, latency: 300, tokens: 400, rss: 90000, taskComplete: true
  });
  delete baseline.hardware;
  delete candidate.hardware;
  const result = compareBenchmarkReports(baseline, candidate);
  assert.equal(result.environment.hardwareComparable, false);
  assert.equal(result.metrics.find(item => item.key === "meanLatencyMs").direction, "incomparable");
  assert.equal(result.metrics.find(item => item.key === "peakProcessRssBytes").direction, "incomparable");
});

test("benchmark report CLI accepts --help without requiring a subcommand", () => {
  const result = spawnSync(process.execPath, ["scripts/benchmark-report.mjs", "--help"], {
    cwd: process.cwd(), encoding: "utf8", windowsHide: true
  });
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /SYNTRA benchmark reports/);
  assert.match(result.stdout, /compare/);
});

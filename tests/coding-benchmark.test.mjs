import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import {
  CODING_BENCHMARK_TASKS,
  CODING_BENCHMARK_VERSION,
  validateBenchmarkManifest
} from "../benchmarks/coding-v1/tasks.mjs";
import {
  evaluateTaskResult,
  summarizeBenchmark
} from "../benchmarks/coding-v1/metrics.mjs";

test("coding benchmark manifest is versioned, bounded, and contains unique task fixtures", () => {
  assert.equal(CODING_BENCHMARK_VERSION, "coding-v1");
  assert.equal(validateBenchmarkManifest(), true);
  assert.equal(CODING_BENCHMARK_TASKS.length, 3);
  assert.equal(new Set(CODING_BENCHMARK_TASKS.map(task => task.id)).size, 3);
  for (const task of CODING_BENCHMARK_TASKS) {
    assert.ok(task.files[task.sourceFile]);
    assert.ok(task.protectedFiles.includes("package.json"));
    assert.ok(task.protectedFiles.some(file => file.startsWith("tests/")));
  }
  assert.throws(() => validateBenchmarkManifest([
    { ...CODING_BENCHMARK_TASKS[0], id: "../escape" }
  ]), /invalid identifier/);
});

test("every benchmark task starts with a failing test suite", async () => {
  for (const task of CODING_BENCHMARK_TASKS) {
    const root = await fs.mkdtemp(path.join(os.tmpdir(), "syntra-benchmark-red-"));
    try {
      for (const [relative, content] of Object.entries(task.files)) {
        const target = path.join(root, ...relative.split("/"));
        await fs.mkdir(path.dirname(target), { recursive: true });
        await fs.writeFile(target, content, "utf8");
      }
      const run = spawnSync(process.execPath, ["--test"], {
        cwd: root,
        encoding: "utf8",
        timeout: 15000,
        windowsHide: true,
        maxBuffer: 1024 * 1024
      });
      assert.equal(run.error, undefined, task.id + ": " + run.error?.message);
      assert.notEqual(run.status, 0, task.id + " must begin with a failing test suite.");
    } finally {
      await fs.rm(root, { recursive: true, force: true });
    }
  }
});

test("benchmark task scoring requires patch validity, agent verification, independent tests, and fixture integrity", () => {
  const task = CODING_BENCHMARK_TASKS[0];
  const base = {
    task,
    agentRun: {
      attempts: [{ status: "verification_passed", promptTokens: 900, outputTokens: 140 }],
      changedFiles: [{ path: task.sourceFile }],
      verification: { passed: true, status: "passed" }
    },
    independentCheck: { ok: true, output: { passed: true, exitCode: 0, durationMs: 34 } },
    protectedIntegrity: { "package.json": true, "tests/slugify.test.js": true },
    sourceChanged: true,
    elapsedMs: 2500,
    peakControllerRssBytes: 123456,
    modelMemory: { before: null, after: null }
  };
  const pass = evaluateTaskResult(base);
  assert.equal(pass.taskCompleted, true);
  assert.equal(pass.patchValid, true);
  assert.equal(pass.totalTokens, 1040);
  assert.equal(pass.repairAttempts, 0);

  const tampered = evaluateTaskResult({
    ...base,
    protectedIntegrity: { "package.json": true, "tests/slugify.test.js": false }
  });
  assert.equal(tampered.taskCompleted, false);
  assert.ok(tampered.failureReasons.some(reason => reason.includes("protected")));

  const invalidPatch = evaluateTaskResult({
    ...base,
    agentRun: {
      attempts: [{ status: "invalid_patch", promptTokens: 900, outputTokens: 140 }],
      changedFiles: [{ path: task.sourceFile }],
      verification: { passed: false, status: "failed" }
    }
  });
  assert.equal(invalidPatch.patchValid, false);
  assert.equal(invalidPatch.taskCompleted, false);
});

test("benchmark summary computes transparent rates, latency, and per-model results", () => {
  const tasks = [
    { status: "passed", model: { provider: "ollama-local", name: "model-a" }, metrics: { taskCompleted: true, patchValid: true, independentTestPassed: true, elapsedMs: 1000, totalTokens: 100 } },
    { status: "failed", model: { provider: "ollama-local", name: "model-a" }, metrics: { taskCompleted: false, patchValid: true, independentTestPassed: false, elapsedMs: 3000, totalTokens: 200 } },
    { status: "passed", model: { provider: "native-local", name: "checkpoint" }, metrics: { taskCompleted: true, patchValid: true, independentTestPassed: true, elapsedMs: 2000, totalTokens: 300 } }
  ];
  const summary = summarizeBenchmark(tasks);
  assert.equal(summary.taskCount, 3);
  assert.equal(summary.completed, 2);
  assert.equal(summary.completionRate, 2 / 3);
  assert.equal(summary.patchValid, 3);
  assert.equal(summary.independentTestsPassed, 2);
  assert.equal(summary.averageLatencyMs, 2000);
  assert.equal(summary.medianLatencyMs, 2000);
  assert.equal(summary.totalTokens, 600);
  assert.equal(summary.byModel.length, 2);
  assert.equal(summary.byModel.find(item => item.model.includes("model-a")).completed, 1);
});

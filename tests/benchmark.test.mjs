import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { evaluateBenchmarkTask, loadBenchmarkSuite, runBenchmarkSuite, summarizeBenchmarkResults, validateBenchmarkSuite } from "../local/benchmark.mjs";

test("versioned benchmark manifest contains representative coding tasks", async () => {
  const { suite } = await loadBenchmarkSuite();
  assert.equal(suite.version, "1.2.0");
  assert.equal(suite.tasks.length, 8);
  assert.ok(suite.tasks.some(task => task.category === "multi-file"));
  assert.ok(suite.tasks.some(task => task.category === "async-control"));
  assert.equal(new Set(suite.tasks.flatMap(task => task.expectedChangedPaths)).size, 9);
});
test("suite validation rejects duplicate IDs and fixture path traversal", () => {
  const task = { id: "safe-task", title: "Safe task", goal: "Implement the requested behavior.", fixture: "fixtures/safe", baselineCheck: "node-script", testEntry: "checks/safe.test.mjs", expectedChangedPaths: ["src/main.mjs"] };
  assert.throws(() => validateBenchmarkSuite({ version: "1.0.0", tasks: [task, { ...task }] }), /unique/i);
  assert.throws(() => validateBenchmarkSuite({ version: "1.0.0", tasks: [{ ...task, fixture: "../outside" }] }), /unsafe fixture/i);
});
test("task evaluator requires exact changed paths and successful verification", () => {
  const task = { id: "sample-task", category: "utility", title: "Sample task", expectedChangedPaths: ["src/math.mjs"] };
  const good = evaluateBenchmarkTask(task, {
    status: "verified_proposals_ready", changedFiles: [{ path: "src/math.mjs" }],
    verification: { passed: true, status: "passed" },
    attempts: [{ promptTokens: 120, outputTokens: 45 }, { promptTokens: 80, outputTokens: 30 }]
  }, { latencyMs: 1500, peakProcessRssBytes: 123456 });
  assert.equal(good.patchValid, true);
  assert.equal(good.verificationPassed, true);
  assert.equal(good.taskComplete, true);
  assert.deepEqual(good.tokens, { prompt: 200, output: 75, total: 275 });
  assert.equal(good.latencyMs, 1500);
  const extra = evaluateBenchmarkTask(task, { changedFiles: [{ path: "src/math.mjs" }, { path: "test/math.test.mjs" }], verification: { passed: true, status: "passed" } });
  assert.equal(extra.patchValid, false);
  assert.equal(extra.taskComplete, false);
  const unverified = evaluateBenchmarkTask(task, { changedFiles: [{ path: "src/math.mjs" }], verification: { passed: false, status: "failed" } });
  assert.equal(unverified.patchValid, true);
  assert.equal(unverified.taskComplete, false);
});
test("benchmark summary calculates rates, latency and token totals", () => {
  const summary = summarizeBenchmarkResults([
    { patchValid: true, verificationPassed: true, taskComplete: true, latencyMs: 100, tokens: { prompt: 20, output: 10, total: 30 }, peakProcessRssBytes: 200 },
    { patchValid: true, verificationPassed: false, taskComplete: false, latencyMs: 300, tokens: { prompt: 40, output: 15, total: 55 }, peakProcessRssBytes: 250 }
  ]);
  assert.equal(summary.taskCount, 2);
  assert.equal(summary.patchValidityRate, 1);
  assert.equal(summary.verificationPassRate, 0.5);
  assert.equal(summary.taskCompletionRate, 0.5);
  assert.deepEqual(summary.latencyMs, { mean: 200, median: 200, min: 100, max: 300 });
  assert.deepEqual(summary.tokens, { prompt: 60, output: 25, total: 85 });
  assert.equal(summary.peakProcessRssBytes, 250);
  assert.deepEqual(summary.categories, {
    uncategorized: { taskCount: 2, patchValidityRate: 1, verificationPassRate: 0.5, taskCompletionRate: 0.5 }
  });
});
test("benchmark runner checks failing baseline in isolated copy and writes a report", async () => {
  const parent = await fs.mkdtemp(path.join(os.tmpdir(), "syntra-benchmark-test-"));
  const suiteDir = path.join(parent, "suite");
  const fixtureDir = path.join(suiteDir, "fixture");
  const sourceDir = path.join(fixtureDir, "src");
  const testDir = path.join(fixtureDir, "checks");
  const original = "export const value = 1;\n";
  const outputPath = path.join(parent, "report.json");
  await fs.mkdir(sourceDir, { recursive: true });
  await fs.mkdir(testDir, { recursive: true });
  await fs.writeFile(path.join(sourceDir, "target.mjs"), original);
  await fs.writeFile(path.join(testDir, "baseline.test.mjs"), [
    'import assert from "node:assert/strict";', 'assert.equal(1, 2);'
  ].join("\n"));
  const suite = { version: "1.0.0", tasks: [{
    id: "fixture-task", category: "test", title: "Disposable fixture test",
    fixture: "fixture", baselineCheck: "node-script", testEntry: "checks/baseline.test.mjs", expectedChangedPaths: ["src/target.mjs"],
    goal: "Implement the expected source behavior without changing tests.", maxIterations: 1, maxSlices: 1
  }] };
  try {
    const { report, outputPath: savedPath } = await runBenchmarkSuite({
      suite, suiteDir, outputPath,
      modelRequest: async (method, route) => {
        assert.equal(method, "GET"); assert.equal(route, "/model/status");
        return { checkpoint_exists: true, provider: "ollama", model_name: "test-local-model", context_tokens: 4096 };
      },
      agentRunner: async options => {
        assert.notEqual(options.root, fixtureDir);
        assert.equal(await fs.readFile(path.join(options.root, "src", "target.mjs"), "utf8"), original);
        await options.onProgress({ phase: "plan-created", model: { provider: "ollama", name: "test-local-model", contextWindowTokens: 4096 } });
        return {
          status: "verified_proposals_ready", changedFiles: [{ path: "src/target.mjs" }],
          verification: { passed: true, status: "passed" }, attempts: [{ promptTokens: 33, outputTokens: 14 }]
        };
      }
    });
    assert.equal(savedPath, outputPath);
    assert.equal(report.model.name, "test-local-model");
    assert.equal(report.summary.completedTaskCount, 1, JSON.stringify(report.tasks));
    assert.equal(report.tasks[0].taskComplete, true);
    assert.deepEqual(JSON.parse(await fs.readFile(outputPath, "utf8")), report);
    assert.equal(await fs.readFile(path.join(sourceDir, "target.mjs"), "utf8"), original);
  } finally { await fs.rm(parent, { recursive: true, force: true }); }
});

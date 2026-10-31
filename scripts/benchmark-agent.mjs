#!/usr/bin/env node
import path from "node:path";
import { detectLocalModel, handleLocalModelRequest } from "../local/model-provider.mjs";
import { DEFAULT_BENCHMARK_SUITE, loadBenchmarkSuite, runBenchmarkSuite } from "../local/benchmark.mjs";
function parseArgs(argv) {
  const result = { tasks: [] };
  for (let index = 0; index < argv.length; index += 1) {
    const value = argv[index];
    if (value === "--task") {
      const id = argv[index + 1];
      if (!id || id.startsWith("--")) throw new Error("--task requires a benchmark task ID.");
      result.tasks.push(id); index += 1;
    } else if (value === "--output") {
      const output = argv[index + 1];
      if (!output || output.startsWith("--")) throw new Error("--output requires a report file path.");
      result.output = output; index += 1;
    } else if (value === "--list") result.list = true;
    else if (value === "--help" || value === "-h") result.help = true;
    else throw new Error("Unknown argument: " + value);
  }
  return result;
}
function help() {
  console.log([
    "SYNTRA local coding benchmark", "",
    "Usage:",
    "  npm run benchmark -- --list",
    "  npm run benchmark",
    "  npm run benchmark -- --task utility-clamp",
    "  npm run benchmark -- --task utility-clamp --task string-slugify --output reports/run.json",
    "",
    "Uses the configured local Ollama model or SYNTRA's local native model service.",
    "Runs disposable fixture copies; does not apply proposals to the active workspace.",
    "Reports measurements and scores, not source contents, prompts, or model output."
  ].join("\n"));
}
function createModelRequest() {
  let providerPromise = null;
  const aiBase = process.env.SYNTRA_BENCHMARK_AI_BASE || "http://127.0.0.1:" + Number(process.env.SYNTRA_AI_PORT || 3002);
  return async (method, route, body, timeoutMs, options = {}) => {
    if (!providerPromise) providerPromise = detectLocalModel();
    const provider = await providerPromise;
    if (provider) return handleLocalModelRequest(provider, method, route, body || {}, { timeoutMs, signal: options.signal });
    const timeout = timeoutMs || (route === "/model/generate" ? 300000 : 30000);
    const signal = options.signal ? AbortSignal.any([options.signal, AbortSignal.timeout(timeout)]) : AbortSignal.timeout(timeout);
    let response;
    try {
      response = await fetch(aiBase + route, {
        method, headers: method === "GET" ? {} : { "Content-Type": "application/json" },
        body: method === "GET" ? undefined : JSON.stringify(body || {}), signal
      });
    } catch (error) {
      if (options.signal?.aborted || error?.name === "AbortError") throw error;
      throw new Error("The local SYNTRA AI service is not responding at " + aiBase + ". Start it with npm run ai:server, or install and start a local Ollama model.");
    }
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.error || "Local AI request failed with HTTP " + response.status + ".");
    return payload;
  };
}
async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    console.log("SYNTRA benchmark: npm run benchmark -- --list | npm run benchmark [--task <id>] [--output <file>]");
    return;
  }
  const { suite, suiteDir } = await loadBenchmarkSuite(DEFAULT_BENCHMARK_SUITE);
  if (args.list) {
    console.log("Benchmark suite " + suite.version);
    for (const task of suite.tasks) console.log("  " + task.id + " — " + task.title);
    return;
  }
  const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
  const outputPath = args.output ? path.resolve(args.output) : path.resolve("benchmark-results", "syntra-" + timestamp + ".json");
  const { report, outputPath: savedPath } = await runBenchmarkSuite({
    suite, suiteDir, taskIds: args.tasks.length ? args.tasks : null,
    modelRequest: createModelRequest(), outputPath,
    onProgress: update => {
      if (update.phase === "started" || update.phase === "finished") console.log("[" + update.taskId + "] " + update.message);
      else if (update.phase && update.message) console.log("[" + update.taskId + "] " + update.phase + ": " + update.message);
    }
  });
  console.log("");
  console.log("Model: " + report.model.provider + " / " + report.model.name);
  console.log("Tasks complete: " + report.summary.completedTaskCount + "/" + report.summary.taskCount);
  console.log("Patch validity: " + Math.round(report.summary.patchValidityRate * 100) + "%");
  console.log("Verification pass rate: " + Math.round(report.summary.verificationPassRate * 100) + "%");
  console.log("Mean latency: " + report.summary.latencyMs.mean + " ms");
  console.log("Prompt/output tokens: " + report.summary.tokens.prompt + " / " + report.summary.tokens.output);
  console.log("Peak benchmark process RSS: " + report.summary.peakProcessRssBytes + " bytes");
  console.log("Report: " + savedPath);
  if (report.summary.completedTaskCount !== report.summary.taskCount) process.exitCode = 1;
}
main().catch(error => {
  console.error("SYNTRA benchmark: " + (error instanceof Error ? error.message : String(error)));
  process.exitCode = 1;
});

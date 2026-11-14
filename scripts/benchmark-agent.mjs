#!/usr/bin/env node
import path from "node:path";
import {
  estimateHybridTokenCount,
  generateHybridCompletion,
  getHybridModelStatus,
  isHybridModelEnabled
} from "../local/model-provider.mjs";
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
    "SYNTRA native coding benchmark", "",
    "Usage:",
    "  npm run benchmark -- --list",
    "  npm run benchmark",
    "  npm run benchmark -- --task utility-clamp",
    "  npm run benchmark -- --task utility-clamp --task string-slugify --output reports/run.json",
    "",
    "Runs the actual SYNTRA native checkpoint against disposable fixture copies.",
    "Does not apply proposals to the active workspace or use an external model service.",
    "Reports measurements and scores, not source contents, prompts, or model output."
  ].join("\n"));
}

function createModelRequest() {
  if (isHybridModelEnabled()) {
    return async (method, route, body, timeoutMs, options = {}) => {
      if (method === "GET" && route === "/model/status") return getHybridModelStatus({ signal: options.signal });
      if (method === "POST" && route === "/model/tokenize") {
        return estimateHybridTokenCount(typeof body?.text === "string" ? body.text : "");
      }
      if (method === "POST" && route === "/model/generate") {
        return generateHybridCompletion(body, { signal: options.signal, timeoutMs: timeoutMs || 300000 });
      }
      throw new Error("Benchmark operation " + route + " is unavailable in hybrid mode.");
    };
  }
  const configuredPort = Number(process.env.SYNTRA_AI_PORT || 3002);
  if (!Number.isInteger(configuredPort) || configuredPort < 1024 || configuredPort > 65535) {
    throw new Error("SYNTRA_AI_PORT must be an integer between 1024 and 65535.");
  }
  const aiBase = "http://127.0.0.1:" + configuredPort;
  return async (method, route, body, timeoutMs, options = {}) => {
    const timeout = timeoutMs || (route === "/model/generate" ? 300000 : 30000);
    const timeoutSignal = AbortSignal.timeout(timeout);
    const signal = options.signal
      ? AbortSignal.any([options.signal, timeoutSignal])
      : timeoutSignal;
    let response;
    try {
      response = await fetch(aiBase + route, {
        method,
        headers: method === "GET" ? {} : { "Content-Type": "application/json" },
        body: method === "GET" ? undefined : JSON.stringify(body || {}),
        redirect: "error",
        signal
      });
    } catch (error) {
      if (options.signal?.aborted || error?.name === "AbortError") throw error;
      throw new Error("SYNTRA's native AI service is not responding at " + aiBase + ". Start it with npm run ai:server and confirm that a trained SYNTRA checkpoint is available.");
    }
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.error || "SYNTRA native AI request failed with HTTP " + response.status + ".");
    return payload;
  };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    help();
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
    modelRequest: createModelRequest(),
    outputPath,
    onProgress: update => {
      if (update.phase === "started" || update.phase === "finished") console.log("[" + update.taskId + "] " + update.message);
      else if (update.phase && update.message) console.log("[" + update.taskId + "] " + update.phase + ": " + update.message);
    }
  });
  console.log("");
  console.log("Engine: " + report.model.engine);
  console.log("Model: " + report.model.name);
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

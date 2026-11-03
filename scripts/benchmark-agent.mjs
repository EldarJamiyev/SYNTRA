#!/usr/bin/env node
import path from "node:path";
import { createNativeBenchmarkModelRequest } from "../local/native-benchmark-bridge.mjs";
import { DEFAULT_BENCHMARK_SUITE, loadBenchmarkSuite, runBenchmarkSuite } from "../local/benchmark.mjs";

function parseArgs(argv) {
  const result = { tasks: [] };
  for (let index = 0; index < argv.length; index += 1) {
    const value = argv[index];
    if (value === "--task") {
      const id = argv[index + 1];
      if (!id || id.startsWith("--")) throw new Error("--task requires a benchmark task ID.");
      result.tasks.push(id);
      index += 1;
    } else if (value === "--output") {
      const output = argv[index + 1];
      if (!output || output.startsWith("--")) throw new Error("--output requires a report file path.");
      result.output = output;
      index += 1;
    } else if (value === "--list") result.list = true;
    else if (value === "--help" || value === "-h") result.help = true;
    else throw new Error("Unknown argument: " + value);
  }
  return result;
}

function help() {
  console.log([
    "SYNTRA native local coding benchmark",
    "",
    "Usage:",
    "  npm run benchmark -- --list",
    "  npm run benchmark",
    "  npm run benchmark -- --task utility-clamp",
    "  npm run benchmark -- --task utility-clamp --task python-slugify --output reports/phase4.json",
    "",
    "Uses only the standalone SYNTRA PyTorch checkpoint; no cloud or external provider fallback exists.",
    "Runs disposable fixture copies and does not apply proposals to the active workspace.",
    "Reports measured task outcomes, not source contents, prompts, or model output."
  ].join("\n"));
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) return help();
  const { suite, suiteDir } = await loadBenchmarkSuite(DEFAULT_BENCHMARK_SUITE);
  if (args.list) {
    console.log("Benchmark suite " + suite.version);
    for (const task of suite.tasks) console.log("  " + task.id + " — " + task.title);
    return;
  }
  const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
  const outputPath = args.output
    ? path.resolve(args.output)
    : path.resolve("benchmark-results", "syntra-" + timestamp + ".json");
  const { report, outputPath: savedPath } = await runBenchmarkSuite({
    suite,
    suiteDir,
    taskIds: args.tasks.length ? args.tasks : null,
    modelRequest: createNativeBenchmarkModelRequest(),
    outputPath,
    onProgress: update => {
      if (update.phase === "started" || update.phase === "finished") {
        console.log("[" + update.taskId + "] " + update.message);
      } else if (update.phase && update.message) {
        console.log("[" + update.taskId + "] " + update.phase + ": " + update.message);
      }
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

#!/usr/bin/env node
import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { compareBenchmarkReports, validateBenchmarkReport } from "../local/benchmark-comparison.mjs";

const DEFAULT_REPORT_DIR = path.resolve("benchmark-results");
const MAX_REPORT_BYTES = 5 * 1024 * 1024;
function parseArgs(argv) {
  const result = { positional: [] };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--directory" || arg === "--limit" || arg === "--baseline" || arg === "--candidate" || arg === "--output") {
      const value = argv[index + 1];
      if (!value || value.startsWith("--")) throw new Error(arg + " requires a value.");
      result[arg.slice(2)] = value; index += 1;
    } else if (arg.startsWith("--")) throw new Error("Unknown option: " + arg);
    else result.positional.push(arg);
  }
  return result;
}
function formatPct(value) { return (value * 100).toFixed(1) + "%"; }
function formatBytes(value) {
  if (!value) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  let current = value, index = 0;
  while (current >= 1024 && index < units.length - 1) { current /= 1024; index += 1; }
  return current.toFixed(index === 0 ? 0 : 1) + " " + units[index];
}
async function readReport(filePath) {
  const resolved = path.resolve(filePath);
  const stat = await fs.lstat(resolved);
  if (!stat.isFile() || stat.isSymbolicLink()) throw new Error("Report path must be a regular, non-symlink file: " + resolved);
  if (stat.size > MAX_REPORT_BYTES) throw new Error("Benchmark report exceeds the 5 MB input limit: " + resolved);
  let report;
  try { report = JSON.parse(await fs.readFile(resolved, "utf8")); }
  catch (error) { throw new Error("Could not parse report " + resolved + ": " + (error instanceof Error ? error.message : String(error))); }
  return validateBenchmarkReport(report);
}
async function listHistory(directory, limit) {
  const root = path.resolve(directory);
  let entries;
  try { entries = await fs.readdir(root, { withFileTypes: true }); }
  catch (error) {
    if (error?.code === "ENOENT") { console.log("No benchmark history found in " + root); return; }
    throw error;
  }
  const files = entries.filter(entry => entry.isFile() && /^syntra-.*\.json$/i.test(entry.name)).map(entry => entry.name).sort().reverse();
  const reports = [];
  for (const name of files) {
    if (reports.length >= limit) break;
    try {
      const report = await readReport(path.join(root, name));
      reports.push({ file: name, report });
    } catch (error) {
      console.warn("Skipping " + name + ": " + (error instanceof Error ? error.message : String(error)));
    }
  }
  if (!reports.length) { console.log("No valid benchmark reports found in " + root); return; }
  console.log("SYNTRA benchmark history — " + root);
  console.log("File | Model | Suite | Completed | Completion | Mean latency");
  console.log("--- | --- | --- | ---: | ---: | ---:");
  for (const { file, report } of reports) {
    console.log([
      file, report.model.name, report.suiteVersion,
      report.summary.completedTaskCount + "/" + report.summary.taskCount,
      formatPct(report.summary.taskCompletionRate),
      report.summary.latencyMs.mean.toLocaleString("en-US") + " ms"
    ].join(" | "));
  }
}
function printComparison(result) {
  const symbol = { improved: "+", regressed: "-", unchanged: "=", incomparable: "~" };
  console.log("SYNTRA benchmark comparison");
  console.log("Suite: " + result.suiteVersion);
  console.log("Baseline: " + result.baseline.model + " (" + result.baseline.runId + ")");
  console.log("Candidate: " + result.candidate.model + " (" + result.candidate.runId + ")");
  console.log("Overall: " + result.status.toUpperCase());
  console.log("");
  console.log("Metric | Baseline | Candidate | Change | Result");
  console.log("--- | ---: | ---: | ---: | ---");
  for (const metric of result.metrics) {
    const percent = metric.relativeChangePct === null ? "n/a" : (metric.relativeChangePct > 0 ? "+" : "") + metric.relativeChangePct + "%";
    const value = metric.key.includes("Rate") ? [formatPct(metric.baseline), formatPct(metric.candidate)] : [metric.baseline.toLocaleString("en-US"), metric.candidate.toLocaleString("en-US")];
    console.log([metric.key, value[0], value[1], percent, symbol[metric.direction] + " " + metric.direction].join(" | "));
  }
  console.log("");
  console.log("Task changes");
  for (const task of result.tasks) console.log("- " + task.id + ": " + task.baselineComplete + " → " + task.candidateComplete + " (" + task.status + ")");
  if (result.warnings.length) {
    console.log("");
    console.log("Comparison warnings");
    for (const warning of result.warnings) console.log("- " + warning);
  }
}
function help() {
  console.log([
    "SYNTRA benchmark reports",
    "",
    "Commands:",
    "  npm run benchmark:reports -- history",
    "  npm run benchmark:reports -- history --directory benchmark-results --limit 30",
    "  npm run benchmark:reports -- compare --baseline <report.json> --candidate <report.json>",
    "  npm run benchmark:reports -- compare --baseline <old.json> --candidate <new.json> --output comparison.json"
  ].join("\n"));
}
async function main() {
  const args = parseArgs(process.argv.slice(2));
  const command = args.positional[0];
  if (!command || command === "help" || command === "--help") return help();
  if (command === "history") {
    const limit = Math.max(1, Math.min(100, Math.floor(Number(args.limit) || 20)));
    await listHistory(args.directory || DEFAULT_REPORT_DIR, limit);
    return;
  }
  if (command === "compare") {
    if (!args.baseline || !args.candidate) throw new Error("compare requires --baseline and --candidate report paths.");
    const baseline = await readReport(args.baseline);
    const candidate = await readReport(args.candidate);
    const result = compareBenchmarkReports(baseline, candidate);
    printComparison(result);
    if (args.output) {
      const outputPath = path.resolve(args.output);
      await fs.mkdir(path.dirname(outputPath), { recursive: true });
      await fs.writeFile(outputPath, JSON.stringify(result, null, 2) + "\n", { encoding: "utf8", flag: "wx" });
      console.log("Comparison JSON: " + outputPath);
    }
    if (result.status === "regressed") process.exitCode = 1;
    return;
  }
  throw new Error("Unknown command: " + command + ". Run npm run benchmark:reports -- --help.");
}
main().catch(error => {
  console.error("SYNTRA benchmark reports: " + (error instanceof Error ? error.message : String(error)));
  process.exitCode = 1;
});

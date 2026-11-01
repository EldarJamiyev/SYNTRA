#!/usr/bin/env node
import { readdir } from "node:fs/promises";
import path from "node:path";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";

const repositoryRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const testsRoot = path.join(repositoryRoot, "tests");

async function main() {
  const entries = await readdir(testsRoot, { withFileTypes: true });
  const testFiles = entries
    .filter(entry => entry.isFile() && entry.name.endsWith(".test.mjs"))
    .map(entry => path.join(testsRoot, entry.name))
    .sort();

  if (testFiles.length === 0) {
    throw new Error("No top-level Node.js regression tests were found in the tests directory.");
  }

  const child = spawn(process.execPath, ["--test", ...testFiles], {
    cwd: repositoryRoot,
    shell: false,
    windowsHide: true,
    env: {
      ...process.env,
      SYNTRA_CHECK_ISOLATION: "host",
      SYNTRA_ALLOW_UNSANDBOXED_CHECKS: "1"
    },
    stdio: "inherit"
  });

  child.on("error", error => {
    console.error("SYNTRA test runner: " + error.message);
    process.exitCode = 1;
  });
  child.on("close", code => {
    process.exitCode = code === 0 ? 0 : 1;
  });
}

main().catch(error => {
  console.error("SYNTRA test runner: " + (error instanceof Error ? error.message : String(error)));
  process.exitCode = 1;
});

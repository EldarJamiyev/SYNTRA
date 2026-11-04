import assert from "node:assert/strict";
import fs from "node:fs/promises";
import test from "node:test";

const launcherPath = new URL("../scripts/start-syntra.ps1", import.meta.url);

test("Windows startup verifies the expected SYNTRA page after starting Vite", async () => {
  const launcher = await fs.readFile(launcherPath, "utf8");
  assert.match(launcher, /function Wait-Endpoint\(\[string\]\$Uri, \[int\]\$TimeoutSeconds = 45, \[string\]\$ExpectedContent = ""\)/);
  assert.match(launcher, /if \(Test-Endpoint \$Uri \$ExpectedContent\) \{ return \}/);
  assert.match(launcher, /Wait-Endpoint \$HealthUrl 45 \$ExpectedContent/);
  assert.match(launcher, /"SYNTRA Chat"/);
});

test("Windows startup failures report HTTP response and the process listening on the failed port", async () => {
  const launcher = await fs.readFile(launcherPath, "utf8");
  assert.match(launcher, /function Get-EndpointDiagnostic/);
  assert.match(launcher, /Last response: HTTP/);
  assert.match(launcher, /Get-NetTCPConnection -State Listen -LocalPort/);
  assert.match(launcher, /Get-CimInstance Win32_Process -Filter/);
  assert.match(launcher, /Port owner:/);
  assert.match(launcher, /Check the Vite service window for its startup error/);
});

test("Windows startup prefers Ollama but falls back to the native runtime when no model is ready", async () => {
  const launcher = await fs.readFile(launcherPath, "utf8");
  assert.match(launcher, /else\s*\{\s*"auto"\s*\}/);
  assert.match(launcher, /\$env:SYNTRA_MODEL_PROVIDER = \$providerMode/);
  assert.match(launcher, /@\("auto", "native", "ollama", "gemini"\) -notcontains \$providerMode/);
  assert.match(launcher, /SYNTRA_GEMINI_API_KEY/);
  assert.match(launcher, /SYNTRA_MODEL_PROVIDER must be auto, native, ollama, or gemini/);
  assert.match(launcher, /\$providerMode -eq "ollama"/);
  assert.match(launcher, /Continuing with SYNTRA's native local runtime/);
  assert.match(launcher, /55% physical-RAM selection limit/);
  assert.match(launcher, /no eligible local model is installed/);
});

test("Vite startup keeps console output visible in its own service window", async () => {
  const launcher = await fs.readFile(launcherPath, "utf8");
  assert.match(launcher, /npm run dev -- --clearScreen false/);
  assert.match(launcher, /The Vite service window will show the frontend startup log/);
});

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

test("Windows startup prepares and always starts SYNTRA's single native AI runtime", async () => {
  const launcher = await fs.readFile(launcherPath, "utf8");
  assert.doesNotMatch(launcher, /SYNTRA_MODEL_PROVIDER|SYNTRA_LOCAL_MODEL|SYNTRA_EMBED_MODEL|SYNTRA_REVIEW_MODEL|11434/i);
  assert.doesNotMatch(launcher, /UsePretrainedModel|launcher-helpers/i);
  assert.match(launcher, /Preparing the isolated native AI environment/);
  assert.match(launcher, /npm install --package-lock-only --ignore-scripts --no-audit/);
  assert.match(launcher, /npm ci/);
  assert.match(launcher, /Ensure-Service 3002 "http:\/\/127\.0\.0\.1:3002\/health" "ai:server"/);
  assert.match(launcher, /Python 3\.12 or newer/);
  const server = await fs.readFile(new URL("../local/server.mjs", import.meta.url), "utf8");
  assert.doesNotMatch(server, /model-provider|detectLocalModel|handleLocalModelRequest|SYNTRA_MODEL_PROVIDER/i);
  await assert.rejects(() => fs.access(new URL("../local/model-provider.mjs", import.meta.url)));
});

test("Vite startup keeps console output visible in its own service window", async () => {
  const launcher = await fs.readFile(launcherPath, "utf8");
  assert.match(launcher, /npm run dev -- --clearScreen false/);
  assert.match(launcher, /The Vite service window will show the frontend startup log/);
});


test("benchmark CLI has a native-only local inference bridge", async () => {
  const benchmark = await fs.readFile(new URL("../scripts/benchmark-agent.mjs", import.meta.url), "utf8");
  assert.match(benchmark, /127\.0\.0\.1/);
  assert.match(benchmark, /\/model\/status|createModelRequest/);
  assert.match(benchmark, /redirect: "error"/);
  assert.doesNotMatch(benchmark, /model-provider|detectLocalModel|handleLocalModelRequest|SYNTRA_REVIEW_MODEL|11434/i);
});

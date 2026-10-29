import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import net from "node:net";
import { setTimeout as delay } from "node:timers/promises";
import test from "node:test";
import { fileURLToPath } from "node:url";

const projectRoot = fileURLToPath(new URL("../", import.meta.url));

async function reservePort() {
  const server = net.createServer();
  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });
  const address = server.address();
  if (!address || typeof address === "string") throw new Error("Could not reserve a test port.");
  const port = address.port;
  await new Promise((resolve, reject) => server.close(error => error ? reject(error) : resolve()));
  return port;
}

async function waitForHealth(port, child, getLogs) {
  const deadline = Date.now() + 8_000;
  let lastError = null;
  while (Date.now() < deadline) {
    if (child.exitCode !== null) {
      throw new Error("Local server exited before health became available.\n" + getLogs());
    }
    try {
      const response = await fetch("http://127.0.0.1:" + port + "/local-api/health", {
        signal: AbortSignal.timeout(750)
      });
      if (response.ok) return;
      lastError = new Error("Health returned HTTP " + response.status);
    } catch (error) {
      lastError = error;
    }
    await delay(60);
  }
  throw new Error("Local server did not become healthy: " + String(lastError) + "\n" + getLogs());
}

async function stopChild(child) {
  if (child.exitCode !== null) return;
  const exited = new Promise(resolve => child.once("exit", resolve));
  child.kill("SIGTERM");
  const result = await Promise.race([
    exited.then(() => "exited"),
    delay(2_000).then(() => "timeout")
  ]);
  if (result === "timeout" && child.exitCode === null) {
    child.kill("SIGKILL");
    await Promise.race([exited, delay(1_000)]);
  }
}

test("local API health, origin protections, and chat validation work over HTTP", async () => {
  const port = await reservePort();
  const child = spawn(process.execPath, ["local/server.mjs"], {
    cwd: projectRoot,
    env: { ...process.env, SYNTRA_LOCAL_PORT: String(port), SYNTRA_MODEL_PROVIDER: "native" },
    stdio: ["ignore", "pipe", "pipe"],
    windowsHide: true
  });
  let logs = "";
  const collect = chunk => {
    logs += chunk.toString("utf8");
    if (logs.length > 12_000) logs = logs.slice(-12_000);
  };
  child.stdout.on("data", collect);
  child.stderr.on("data", collect);

  try {
    await waitForHealth(port, child, () => logs);
    const base = "http://127.0.0.1:" + port;

    const healthResponse = await fetch(base + "/local-api/health");
    assert.equal(healthResponse.status, 200);
    const health = await healthResponse.json();
    assert.equal(health.status, "ok");
    assert.equal(health.host, "127.0.0.1");
    assert.ok(health.capabilities.includes("local-chat"));
    assert.ok(health.capabilities.includes("local-network-diagnostics"));

    const deniedOrigin = await fetch(base + "/local-api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json", Origin: "https://untrusted.example" },
      body: JSON.stringify({ message: "Do not start a task." })
    });
    assert.equal(deniedOrigin.status, 403);
    assert.match((await deniedOrigin.json()).error, /Origin is not allowed/);

    const deniedCrossSite = await fetch(base + "/local-api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json", "Sec-Fetch-Site": "cross-site" },
      body: JSON.stringify({ message: "Do not start a task." })
    });
    assert.equal(deniedCrossSite.status, 403);
    assert.match((await deniedCrossSite.json()).error, /Cross-site requests/);

    const invalidJson = await fetch(base + "/local-api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json", Origin: "http://127.0.0.1:3000" },
      body: "{"
    });
    assert.equal(invalidJson.status, 400);
    assert.match((await invalidJson.json()).error, /valid JSON/);

    const invalidMessage = await fetch(base + "/local-api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json", Origin: "http://127.0.0.1:3000" },
      body: JSON.stringify({ message: "   " })
    });
    assert.equal(invalidMessage.status, 400);
    assert.match((await invalidMessage.json()).error, /Message must contain/);

    const diagnosticResponse = await fetch(base + "/local-api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json", Origin: "http://127.0.0.1:3000" },
      body: JSON.stringify({ message: "/dns 127.0.0.1" })
    });
    assert.equal(diagnosticResponse.status, 200);
    const diagnosticJob = await diagnosticResponse.json();
    assert.ok(diagnosticJob.jobId);

    let diagnosticStatus = null;
    const diagnosticDeadline = Date.now() + 5_000;
    while (Date.now() < diagnosticDeadline) {
      const statusResponse = await fetch(base + "/local-api/chat/status", {
        method: "POST",
        headers: { "Content-Type": "application/json", Origin: "http://127.0.0.1:3000" },
        body: JSON.stringify({ jobId: diagnosticJob.jobId })
      });
      assert.equal(statusResponse.status, 200);
      diagnosticStatus = await statusResponse.json();
      if (diagnosticStatus.status !== "running") break;
      await delay(40);
    }
    assert.equal(diagnosticStatus?.status, "completed");
    assert.equal(diagnosticStatus?.diagnosticStatus, "resolved");
    assert.match(diagnosticStatus?.reply || "", /DNS lookup: 127\.0\.0\.1/);


    const helpStartResponse = await fetch(base + "/local-api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json", Origin: "http://127.0.0.1:3000" },
      body: JSON.stringify({ message: "/help" })
    });
    assert.equal(helpStartResponse.status, 200);
    const helpJob = await helpStartResponse.json();
    let helpStatus = null;
    const helpDeadline = Date.now() + 5_000;
    while (Date.now() < helpDeadline) {
      const statusResponse = await fetch(base + "/local-api/chat/status", {
        method: "POST",
        headers: { "Content-Type": "application/json", Origin: "http://127.0.0.1:3000" },
        body: JSON.stringify({ jobId: helpJob.jobId })
      });
      helpStatus = await statusResponse.json();
      if (helpStatus.status !== "running") break;
      await delay(40);
    }
    assert.equal(helpStatus?.status, "completed");
    assert.match(helpStatus?.reply || "", /SYNTRA help/);
    assert.match(helpStatus?.reply || "", /\\/dns example\\.com/);

    const unknownRoute = await fetch(base + "/not-a-route");
    assert.equal(unknownRoute.status, 404);
    assert.match((await unknownRoute.json()).error, /Route not found/);
  } finally {
    await stopChild(child);
  }
});

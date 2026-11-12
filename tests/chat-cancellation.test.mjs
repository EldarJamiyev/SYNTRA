import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { createServer } from "node:http";
import net from "node:net";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { setTimeout as delay } from "node:timers/promises";
import test from "node:test";
import { fileURLToPath } from "node:url";

const projectRoot = path.resolve(fileURLToPath(new URL("../", import.meta.url)));

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

async function waitForHealth(port, child) {
  const deadline = Date.now() + 8000;
  let lastError = null;
  while (Date.now() < deadline) {
    if (child.exitCode !== null) throw new Error("SYNTRA local server exited before becoming healthy.");
    try {
      const response = await fetch("http://127.0.0.1:" + port + "/local-api/health", {
        signal: AbortSignal.timeout(500)
      });
      if (response.ok) return;
    } catch {
      lastError = error;
    }
    await delay(40);
  }
  throw new Error("SYNTRA local server did not become healthy: " +
    (lastError instanceof Error ? lastError.message : "timeout"));
}

async function stopChild(child) {
  if (!child || child.exitCode !== null || child.signalCode !== null) return;
  await new Promise(resolve => {
    const timer = setTimeout(() => {
      try { child.kill("SIGKILL"); } catch {}
      resolve();
    }, 2500);
    timer.unref?.();
    child.once("exit", () => {
      clearTimeout(timer);
      resolve();
    });
    child.kill("SIGTERM");
  });
}

test("cancelling ordinary chat aborts the native inference HTTP stream and remains terminal", { timeout: 15000 }, async () => {
  const localPort = await reservePort();
  let aiPort = await reservePort();
  while (aiPort === localPort) aiPort = await reservePort();
  const workspace = await fs.mkdtemp(path.join(os.tmpdir(), "syntra-chat-cancel-"));

  let generationStartedResolve;
  let generationClosedResolve;
  const generationStarted = new Promise(resolve => { generationStartedResolve = resolve; });
  const generationClosed = new Promise(resolve => { generationClosedResolve = resolve; });
  let generationResponseClosed = false;

  const aiServer = createServer(async (request, response) => {
    try {
      for await (const _chunk of request) { /* consume the request body before replying */ }
      if (request.method === "GET" && request.url === "/model/status") {
        response.writeHead(200, { "Content-Type": "application/json" });
        response.end(JSON.stringify({
          status: "ok",
          checkpoint_exists: true,
          context_tokens: 256,
          tokenizer_vocab_size: 258,
          trained_steps: 1
        }));
        return;
      }
      if (request.method === "POST" && request.url === "/model/tokenize") {
        response.writeHead(200, { "Content-Type": "application/json" });
        response.end(JSON.stringify({ token_count: 12 }));
        return;
      }
      if (request.method === "POST" && request.url === "/model/generate") {
        response.on("close", () => {
          if (!response.writableEnded) {
            generationResponseClosed = true;
            generationClosedResolve();
          }
        });
        response.writeHead(200, { "Content-Type": "application/x-ndjson; charset=utf-8" });
        response.write(JSON.stringify({ delta: "partial answer", generated_tokens: 2, done: false }) + "\n");
        generationStartedResolve();
        return;
      }
      response.writeHead(404, { "Content-Type": "application/json" });
      response.end(JSON.stringify({ error: "Unexpected test request." }));
    } catch (error) {
      if (!response.destroyed) {
        response.writeHead(500, { "Content-Type": "application/json" });
        response.end(JSON.stringify({ error: "Unexpected failure in the test inference server." }));
      }
    }
  });

  await new Promise((resolve, reject) => {
    aiServer.once("error", reject);
    aiServer.listen(aiPort, "127.0.0.1", resolve);
  });

  const localChild = spawn(process.execPath, [path.join(projectRoot, "local/server.mjs")], {
    cwd: projectRoot,
    env: {
      ...process.env,
      SYNTRA_WORKSPACE: workspace,
      SYNTRA_LOCAL_PORT: String(localPort),
      SYNTRA_AI_PORT: String(aiPort),
      SYNTRA_API_TOKEN: "syntra-chat-cancellation-test-token"
    },
    stdio: "ignore",
    windowsHide: true
  });

  try {
    await waitForHealth(localPort, localChild);
    const base = "http://127.0.0.1:" + localPort;
    const sessionResponse = await fetch(base + "/local-api/session", {
      headers: { Origin: "http://127.0.0.1:3000" }
    });
    assert.equal(sessionResponse.status, 200);
    const { token } = await sessionResponse.json();
    const apiFetch = (url, options = {}) => fetch(url, {
      ...options,
      headers: {
        ...(options.headers || {}),
        Authorization: "Bearer " + token,
        Origin: "http://127.0.0.1:3000"
      }
    });

    const startResponse = await apiFetch(base + "/local-api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: "Explain how a tokenizer works in a language model." })
    });
    assert.equal(startResponse.status, 200);
    const started = await startResponse.json();
    assert.ok(started.jobId);

    await Promise.race([
      generationStarted,
      delay(5000).then(() => { throw new Error("Native inference generation did not start."); })
    ]);

    const cancelResponse = await apiFetch(base + "/local-api/chat/cancel", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ jobId: started.jobId })
    });
    assert.equal(cancelResponse.status, 200);
    const cancellation = await cancelResponse.json();
    assert.equal(cancellation.ok, true);
    assert.equal(cancellation.status, "cancelling");

    let status = null;
    const deadline = Date.now() + 4000;
    while (Date.now() < deadline) {
      const response = await apiFetch(base + "/local-api/chat/status", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ jobId: started.jobId })
      });
      assert.equal(response.status, 200);
      status = await response.json();
      if (status.status !== "running") break;
      await delay(25);
    }
    assert.equal(status?.status, "cancelled");
    assert.equal(status?.phase, "cancelled");
    assert.match(status?.reply || "", /Task cancelled/);

    await Promise.race([
      generationClosed,
      delay(3000).then(() => { throw new Error("Cancellation did not close the in-flight inference response."); })
    ]);
    assert.equal(generationResponseClosed, true,
      "the local AI HTTP response must be closed by the chat AbortSignal");
  } finally {
    await stopChild(localChild);
    await new Promise(resolve => aiServer.close(resolve));
    await fs.rm(workspace, { recursive: true, force: true });
  }
});

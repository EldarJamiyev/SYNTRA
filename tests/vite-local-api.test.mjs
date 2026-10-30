import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { EventEmitter } from "node:events";
import fs from "node:fs/promises";
import net from "node:net";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { syntraLocalApiPlugin } from "../scripts/vite-local-api.mjs";

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

async function waitUntilOffline(url) {
  const deadline = Date.now() + 5000;
  while (Date.now() < deadline) {
    try {
      await fetch(url, { signal: AbortSignal.timeout(250) });
    } catch {
      return;
    }
    await new Promise(resolve => setTimeout(resolve, 50));
  }
  throw new Error("The Vite-owned local API process did not stop during server shutdown.");
}

test("Vite starts the local API before serving the frontend and stops the child on shutdown", async () => {
  const port = await reservePort();
  const aiPort = await reservePort();
  const workspace = await fs.mkdtemp(path.join(os.tmpdir(), "syntra-vite-workspace-"));
  const previous = {
    localPort: process.env.SYNTRA_LOCAL_PORT,
    aiPort: process.env.SYNTRA_AI_PORT,
    workspace: process.env.SYNTRA_WORKSPACE,
    autostart: process.env.SYNTRA_AUTOSTART_LOCAL_API
  };
  process.env.SYNTRA_LOCAL_PORT = String(port);
  process.env.SYNTRA_AI_PORT = String(aiPort);
  process.env.SYNTRA_WORKSPACE = workspace;
  delete process.env.SYNTRA_AUTOSTART_LOCAL_API;

  const httpServer = new EventEmitter();
  const watcher = new EventEmitter();
  const messages = [];
  const viteServer = {
    httpServer,
    watcher,
    config: { logger: { info: message => messages.push(message), warn: message => messages.push(message) } }
  };

  try {
    const plugin = syntraLocalApiPlugin({ root: projectRoot, stdio: "ignore", timeoutMs: 10000 });
    await plugin.configureServer(viteServer);

    const health = await fetch("http://127.0.0.1:" + port + "/local-api/health");
    assert.equal(health.status, 200);
    assert.equal((await health.json()).status, "ok");

    const status = await fetch("http://127.0.0.1:" + port + "/local-api/workspace/status");
    assert.equal(status.status, 200);
    assert.equal((await status.json()).root, await fs.realpath(workspace));
    assert.ok(messages.some(message => /Local workspace API is ready/.test(message)));

    httpServer.emit("close");
    await waitUntilOffline("http://127.0.0.1:" + port + "/local-api/health");
  } finally {
    httpServer.emit("close");
    watcher.emit("close");
    if (previous.localPort === undefined) delete process.env.SYNTRA_LOCAL_PORT;
    else process.env.SYNTRA_LOCAL_PORT = previous.localPort;
    if (previous.aiPort === undefined) delete process.env.SYNTRA_AI_PORT;
    else process.env.SYNTRA_AI_PORT = previous.aiPort;
    if (previous.workspace === undefined) delete process.env.SYNTRA_WORKSPACE;
    else process.env.SYNTRA_WORKSPACE = previous.workspace;
    if (previous.autostart === undefined) delete process.env.SYNTRA_AUTOSTART_LOCAL_API;
    else process.env.SYNTRA_AUTOSTART_LOCAL_API = previous.autostart;
    await fs.rm(workspace, { recursive: true, force: true });
  }
});

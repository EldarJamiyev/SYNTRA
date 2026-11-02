import assert from "node:assert/strict";
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
  const workspace = await fs.mkdtemp(path.join(os.tmpdir(), "syntra-vite-workspace-"));
  let ownedServer = null;
  let lastStartupError = null;
  try {
    // Other test workers start temporary SYNTRA servers concurrently. A port
    // returned by reservePort can be claimed between close() and child startup.
    // Retry when the plugin finds a healthy server that belongs to another test.
    for (let attempt = 0; attempt < 8 && !ownedServer; attempt += 1) {
      const port = await reservePort();
      const aiPort = await reservePort();
      const env = {
        ...process.env,
        SYNTRA_LOCAL_PORT: String(port),
        SYNTRA_AI_PORT: String(aiPort),
        SYNTRA_WORKSPACE: workspace
      };
      delete env.SYNTRA_AUTOSTART_LOCAL_API;

      const httpServer = new EventEmitter();
      const watcher = new EventEmitter();
      const messages = [];
      const viteServer = {
        httpServer,
        watcher,
        config: { logger: { info: message => messages.push(message), warn: message => messages.push(message) } }
      };

      try {
        const plugin = syntraLocalApiPlugin({ root: projectRoot, env, stdio: "ignore", timeoutMs: 10000 });
        await plugin.configureServer(viteServer);

        const base = "http://127.0.0.1:" + port;
        const health = await fetch(base + "/local-api/health");
        assert.equal(health.status, 200);
        assert.equal((await health.json()).status, "ok");

        const sessionResponse = await fetch(base + "/local-api/session", {
          headers: { Origin: "http://127.0.0.1:3000" }
        });
        assert.equal(sessionResponse.status, 200);
        const token = (await sessionResponse.json()).token;
        const status = await fetch(base + "/local-api/workspace/status", {
          headers: { Authorization: "Bearer " + token }
        });
        assert.equal(status.status, 200);
        const statusBody = await status.json();
        const startedOwnChild = messages.some(message => /Local workspace API is ready/.test(message));

        if (statusBody.root !== await fs.realpath(workspace) || !startedOwnChild) {
          httpServer.emit("close");
          watcher.emit("close");
          lastStartupError = new Error("A concurrent test already owned reserved port " + port + "; retrying with another port.");
          continue;
        }

        assert.ok(messages.some(message => /Local workspace API is ready/.test(message)));
        ownedServer = { base, httpServer, watcher };
      } catch (error) {
        httpServer.emit("close");
        watcher.emit("close");
        lastStartupError = error;
      }
    }

    assert.ok(ownedServer, lastStartupError?.message || "Could not acquire an isolated local API test port.");
    ownedServer.httpServer.emit("close");
    await waitUntilOffline(ownedServer.base + "/local-api/health");
  } finally {
    if (ownedServer) {
      ownedServer.httpServer.emit("close");
      ownedServer.watcher.emit("close");
    }
    await fs.rm(workspace, { recursive: true, force: true });
  }
});

import { spawn } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { setTimeout as delay } from "node:timers/promises";

const DEFAULT_ROOT = path.resolve(fileURLToPath(new URL("../", import.meta.url)));

function apiPort() {
  const value = Number(process.env.SYNTRA_LOCAL_PORT || 3001);
  if (!Number.isInteger(value) || value < 1 || value > 65535) {
    throw new Error("SYNTRA_LOCAL_PORT must be a valid TCP port.");
  }
  return value;
}

async function isHealthy(url) {
  try {
    const response = await fetch(url, { signal: AbortSignal.timeout(700) });
    if (!response.ok) return false;
    const body = await response.json();
    return body?.status === "ok" && body?.service === "SYNTRA local chat and coding agent";
  } catch {
    return false;
  }
}

function stopChild(child) {
  if (!child || child.exitCode !== null || child.killed) return;
  try {
    child.kill("SIGTERM");
  } catch {
    // The child may already have exited while Vite was shutting down.
  }
}

async function startOrReuseApi({ root, port, stdio, timeoutMs }) {
  const healthUrl = "http://127.0.0.1:" + port + "/local-api/health";
  if (await isHealthy(healthUrl)) return { child: null, reused: true };

  const child = spawn(process.execPath, [path.join(root, "local", "server.mjs")], {
    cwd: root,
    env: process.env,
    stdio,
    windowsHide: true
  });

  let spawnError = null;
  let output = "";
  child.once("error", error => { spawnError = error; });
  for (const stream of [child.stdout, child.stderr]) {
    stream?.on("data", chunk => {
      output += chunk.toString("utf8");
      if (output.length > 8000) output = output.slice(-8000);
    });
  }

  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (spawnError) {
      stopChild(child);
      throw new Error("Could not start SYNTRA's local API: " + spawnError.message);
    }
    if (child.exitCode !== null) {
      throw new Error(
        "SYNTRA's local API exited before becoming healthy (code " + child.exitCode + ").\n" +
        (output || "Check Node.js and local/server.mjs.")
      );
    }
    if (await isHealthy(healthUrl)) return { child, reused: false };
    await delay(120);
  }

  stopChild(child);
  throw new Error(
    "SYNTRA's local API did not become healthy at " + healthUrl + ".\n" +
    (output || "Check whether port " + port + " is occupied or the workspace cannot be opened.")
  );
}

/**
 * Starts the Node workspace API whenever Vite's development server starts.
 * The native Python model remains a separately managed, optional service.
 */
export function syntraLocalApiPlugin(options = {}) {
  const root = path.resolve(options.root || DEFAULT_ROOT);
  const stdio = options.stdio || "inherit";
  const timeoutMs = options.timeoutMs || 20000;

  return {
    name: "syntra-local-api",
    apply: "serve",
    async configureServer(server) {
      if (process.env.SYNTRA_AUTOSTART_LOCAL_API === "0") {
        server.config.logger.warn(
          "[SYNTRA] Automatic local API startup is disabled (SYNTRA_AUTOSTART_LOCAL_API=0)."
        );
        return;
      }

      const port = apiPort();
      const result = await startOrReuseApi({ root, port, stdio, timeoutMs });
      server.config.logger.info(
        result.reused
          ? "[SYNTRA] Reusing the healthy local API on port " + port + "."
          : "[SYNTRA] Local workspace API is ready on http://127.0.0.1:" + port + "."
      );

      if (result.child) {
        const shutdown = () => stopChild(result.child);
        server.httpServer?.once("close", shutdown);
        server.watcher?.once("close", shutdown);
      }
    }
  };
}

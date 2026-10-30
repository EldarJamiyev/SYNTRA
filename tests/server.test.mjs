import assert from "node:assert/strict";
import { spawn } from "node:child_process";
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
  const aiPort = await reservePort();
  const child = spawn(process.execPath, ["local/server.mjs"], {
    cwd: projectRoot,
    env: {
      ...process.env,
      SYNTRA_LOCAL_PORT: String(port),
      SYNTRA_AI_PORT: String(aiPort),
      SYNTRA_MODEL_PROVIDER: "native"
    },
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
    assert.ok(health.capabilities.includes("persistent-proposal-review"));

    const workspaceStatusResponse = await fetch(base + "/local-api/workspace/status");
    assert.equal(workspaceStatusResponse.status, 200);
    const workspaceStatus = await workspaceStatusResponse.json();
    assert.equal(workspaceStatus.root, projectRoot);
    assert.deepEqual(workspaceStatus.pendingProposals, []);
    assert.deepEqual(workspaceStatus.staleProposals, []);

    const switchWorkspaceResponse = await fetch(base + "/local-api/workspace/switch", {
      method: "POST",
      headers: { "Content-Type": "application/json", Origin: "http://127.0.0.1:3000" },
      body: JSON.stringify({ root: projectRoot })
    });
    assert.equal(switchWorkspaceResponse.status, 200);
    assert.equal((await switchWorkspaceResponse.json()).root, projectRoot);

    const invalidWorkspaceResponse = await fetch(base + "/local-api/workspace/switch", {
      method: "POST",
      headers: { "Content-Type": "application/json", Origin: "http://127.0.0.1:3000" },
      body: JSON.stringify({ root: path.join(projectRoot, "missing-workspace") })
    });
    assert.equal(invalidWorkspaceResponse.status, 400);
    assert.match((await invalidWorkspaceResponse.json()).error, /workspace|directory|folder/i);

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
    assert.ok((helpStatus?.reply || "").includes("/dns example.com"));
    assert.ok((helpStatus?.reply || "").includes("/proposals"));
    assert.ok((helpStatus?.reply || "").includes("/discard"));

    const naturalHelpResponse = await fetch(base + "/local-api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json", Origin: "http://127.0.0.1:3000" },
      body: JSON.stringify({ message: "What can you do?" })
    });
    assert.equal(naturalHelpResponse.status, 200);
    const naturalHelpJob = await naturalHelpResponse.json();
    let naturalHelpStatus = null;
    const naturalHelpDeadline = Date.now() + 5_000;
    while (Date.now() < naturalHelpDeadline) {
      const response = await fetch(base + "/local-api/chat/status", {
        method: "POST",
        headers: { "Content-Type": "application/json", Origin: "http://127.0.0.1:3000" },
        body: JSON.stringify({ jobId: naturalHelpJob.jobId })
      });
      naturalHelpStatus = await response.json();
      if (naturalHelpStatus.status !== "running") break;
      await delay(40);
    }
    assert.equal(naturalHelpStatus?.status, "completed");
    assert.match(naturalHelpStatus?.reply || "", /SYNTRA help/);
    assert.ok((naturalHelpStatus?.reply || "").includes("/status"));

    const statusResponse = await fetch(base + "/local-api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json", Origin: "http://127.0.0.1:3000" },
      body: JSON.stringify({ message: "/status" })
    });
    assert.equal(statusResponse.status, 200);
    const statusJob = await statusResponse.json();
    let statusResult = null;
    const statusDeadline = Date.now() + 5_000;
    while (Date.now() < statusDeadline) {
      const response = await fetch(base + "/local-api/chat/status", {
        method: "POST",
        headers: { "Content-Type": "application/json", Origin: "http://127.0.0.1:3000" },
        body: JSON.stringify({ jobId: statusJob.jobId })
      });
      statusResult = await response.json();
      if (statusResult.status !== "running") break;
      await delay(40);
    }
    assert.equal(statusResult?.status, "completed");
    assert.equal(statusResult?.modelStatus, "unavailable");
    assert.match(statusResult?.reply || "", /Workspace service: running/);
    assert.match(statusResult?.reply || "", /Network diagnostics: available without a model/);

    const unknownRoute = await fetch(base + "/not-a-route");
    assert.equal(unknownRoute.status, 404);
    assert.match((await unknownRoute.json()).error, /Route not found/);
  } finally {
    await stopChild(child);
  }
});


async function startLocalServer(workspaceRoot) {
  const port = await reservePort();
  const aiPort = await reservePort();
  const child = spawn(process.execPath, ["local/server.mjs"], {
    cwd: projectRoot,
    env: {
      ...process.env,
      SYNTRA_LOCAL_PORT: String(port),
      SYNTRA_AI_PORT: String(aiPort),
      SYNTRA_WORKSPACE: workspaceRoot,
      SYNTRA_MODEL_PROVIDER: "native"
    },
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
    return { base: "http://127.0.0.1:" + port, child, getLogs: () => logs };
  } catch (error) {
    await stopChild(child);
    throw new Error((error instanceof Error ? error.message : String(error)) + "\n" + logs);
  }
}

async function runChatCommand(base, message) {
  const headers = { "Content-Type": "application/json", Origin: "http://127.0.0.1:3000" };
  const start = await fetch(base + "/local-api/chat", {
    method: "POST",
    headers,
    body: JSON.stringify({ message })
  });
  assert.equal(start.status, 200);
  const job = await start.json();
  const deadline = Date.now() + 5_000;
  let status = null;
  while (Date.now() < deadline) {
    const response = await fetch(base + "/local-api/chat/status", {
      method: "POST",
      headers,
      body: JSON.stringify({ jobId: job.jobId })
    });
    assert.equal(response.status, 200);
    status = await response.json();
    if (status.status !== "running") break;
    await delay(40);
  }
  assert.equal(status?.status, "completed", "Chat command did not complete.");
  return status;
}

test("local tool catalogue and API enforce permissions and record execution", async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "syntra-tool-api-"));
  await fs.mkdir(path.join(root, "src"), { recursive: true });
  await fs.writeFile(path.join(root, "src", "index.js"), "export const value = 1;\n", "utf8");
  let service = null;
  try {
    service = await startLocalServer(root);
    const catalogueResponse = await fetch(service.base + "/local-api/tools");
    assert.equal(catalogueResponse.status, 200);
    const catalogue = await catalogueResponse.json();
    assert.equal(catalogue.tools.length, 12);
    assert.ok(catalogue.tools.some(tool => tool.name === "workspace.search"));
    assert.ok(catalogue.tools.some(tool => tool.name === "workspace.proposals.apply-many" && tool.approvalRequired));
    assert.ok(catalogue.tools.some(tool => tool.name === "git.diff-summary" && tool.permission === "read_only"));
    assert.ok(catalogue.tools.some(tool => tool.name === "workspace.check" && tool.approvalRequired));

    const headers = { "Content-Type": "application/json", Origin: "http://127.0.0.1:3000" };
    const inspectResponse = await fetch(service.base + "/local-api/tools/execute", {
      method: "POST", headers,
      body: JSON.stringify({ name: "workspace.inspect", root, input: {} })
    });
    assert.equal(inspectResponse.status, 200);
    const inspectRun = await inspectResponse.json();
    assert.equal(inspectRun.ok, true);
    assert.equal(inspectRun.name, "workspace.inspect");
    assert.equal(inspectRun.auditRecorded, true);

    const deniedResponse = await fetch(service.base + "/local-api/tools/execute", {
      method: "POST", headers,
      body: JSON.stringify({ name: "workspace.check", root, input: { task: "test" } })
    });
    const denied = await deniedResponse.json();
    assert.equal(denied.ok, false);
    assert.equal(denied.status, "approval_required");

    const deniedOrigin = await fetch(service.base + "/local-api/tools/execute", {
      method: "POST",
      headers: { "Content-Type": "application/json", Origin: "https://attacker.invalid" },
      body: JSON.stringify({ name: "workspace.inspect", root, input: {} })
    });
    assert.equal(deniedOrigin.status, 403);

    const events = (await fs.readFile(path.join(root, ".syntra", "actions.jsonl"), "utf8"))
      .split(/\r?\n/).filter(Boolean).map(line => JSON.parse(line));
    assert.ok(events.some(event => event.tool === "workspace.inspect" && event.status === "completed"));
    assert.ok(events.some(event => event.tool === "workspace.check" && event.status === "approval_required"));
  } finally {
    if (service?.child) await stopChild(service.child);
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("saved workspace proposals survive a real local service restart", async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "syntra-restart-workspace-"));
  await fs.mkdir(path.join(root, "src"), { recursive: true });
  const canonicalRoot = await fs.realpath(root);
  const sourcePath = path.join(root, "src", "app.js");
  const original = "export const version = 1;\n";
  const proposed = "export const version = 2;\n";
  await fs.writeFile(sourcePath, original, "utf8");

  let service = null;
  try {
    service = await startLocalServer(root);
    const staged = await fetch(service.base + "/local-api/workspace/propose", {
      method: "POST",
      headers: { "Content-Type": "application/json", Origin: "http://127.0.0.1:3000" },
      body: JSON.stringify({ root, path: "src/app.js", content: proposed })
    });
    assert.equal(staged.status, 200);
    const proposal = await staged.json();
    assert.ok(proposal.id);
    assert.equal(await fs.readFile(sourcePath, "utf8"), original, "staging must not modify the source file");

    await stopChild(service.child);
    service = await startLocalServer(root);

    const restoredResponse = await fetch(service.base + "/local-api/workspace/status");
    assert.equal(restoredResponse.status, 200);
    const restored = await restoredResponse.json();
    assert.equal(restored.root, canonicalRoot);
    assert.equal(restored.pendingProposals.length, 1);
    assert.equal(restored.pendingProposals[0].id, proposal.id);

    const review = await runChatCommand(service.base, "/proposals");
    assert.match(review.reply, /Saved workspace proposals/);
    assert.ok(review.reply.includes("src/app.js"));

    const applied = await runChatCommand(service.base, "/apply");
    assert.match(applied.reply, /Applied the explicitly requested proposals/);
    assert.equal(await fs.readFile(sourcePath, "utf8"), proposed);

    const afterApply = await fetch(service.base + "/local-api/workspace/status");
    const finalStatus = await afterApply.json();
    assert.deepEqual(finalStatus.pendingProposals, []);
    assert.deepEqual(finalStatus.staleProposals, []);
  } finally {
    if (service?.child) await stopChild(service.child);
    await fs.rm(root, { recursive: true, force: true });
  }
});

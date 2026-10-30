import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { spawn } from "node:child_process";
import fs from "node:fs/promises";
import { createServer } from "node:net";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import { validateStructuredPatch } from "../local/agent.mjs";
import {
  createAgentRun,
  getAgentRun,
  listAgentRuns,
  recoverInterruptedAgentRuns,
  requestAgentRunCancellation,
  updateAgentRun
} from "../local/workspace.mjs";

async function createWorkspace() {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "syntra-agent-runs-"));
  await fs.mkdir(path.join(root, "src"), { recursive: true });
  await fs.writeFile(path.join(root, "src", "index.js"), "export const value = 1;\n");
  return root;
}

test("agent run records persist the task plan, context, checkpoint, and event history", async () => {
  const root = await createWorkspace();
  try {
    const created = await createAgentRun(root, "Refactor the local workspace service safely.");
    assert.match(created.id, /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i);
    assert.equal(created.status, "running");
    assert.equal(created.checkpoint.stage, "created");

    const updated = await updateAgentRun(root, created.id, {
      phase: "patch-validated",
      iteration: 2,
      message: "Validated an exact-anchor patch.",
      planSteps: [
        { id: "discover", title: "Discover architecture", status: "completed", detail: "Read 12 files." },
        { id: "patch", title: "Create patch", status: "completed", detail: "Two source edits." }
      ],
      selectedContext: [{ path: "src/index.js", score: 3.5, sha256: "a".repeat(64) }],
      attempts: [{ iteration: 2, status: "patch_validated", promptTokens: 120, outputTokens: 80, changedFiles: ["src/index.js"] }],
      toolCalls: [{ runId: created.id, name: "workspace.inspect", status: "completed", permission: "read_only", durationMs: 15 }],
      changedFiles: [{ path: "src/index.js", existedBefore: true, baseHash: "b".repeat(64), afterHash: "c".repeat(64), reason: "Update a value." }],
      recoveryPatch: {
        summary: "Update the exported value.",
        edits: [{ path: "src/index.js", find: "value = 1", replace: "value = 2", reason: "Use the new value." }]
      },
      checkpoint: { stage: "patch_validated", iteration: 2 },
      event: { phase: "patch-validated", message: "Patch checkpoint saved.", iteration: 2 }
    });
    const restored = await getAgentRun(root, created.id);
    assert.equal(restored.phase, "patch-validated");
    assert.equal(restored.iteration, 2);
    assert.equal(restored.planSteps.length, 2);
    assert.equal(restored.selectedContext[0].path, "src/index.js");
    assert.equal(restored.toolCalls[0].name, "workspace.inspect");
    assert.equal(restored.recoveryPatch.edits[0].replace, "value = 2");
    assert.equal(restored.checkpoint.stage, "patch_validated");
    assert.equal(restored.events.at(-1).message, "Patch checkpoint saved.");
    await updateAgentRun(root, created.id, {
      model: {
        provider: "ollama-local", name: "qwen-local",
        contextWindowTokens: 8192, contextBudgetTokens: 5570, outputBudgetTokens: 1024
      }
    });
    assert.equal((await getAgentRun(root, created.id)).model.name, "qwen-local");
    assert.ok(Date.parse(updated.updatedAt) >= Date.parse(created.createdAt));
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("unfinished agent runs are marked interrupted and can be listed after restart", async () => {
  const root = await createWorkspace();
  try {
    const first = await createAgentRun(root, "Update the API response schema.");
    const second = await createAgentRun(root, "Add validation to the request handler.");
    await updateAgentRun(root, second.id, {
      phase: "verification",
      checkpoint: { stage: "candidate_checked", iteration: 1 }
    });
    await updateAgentRun(root, first.id, { status: "completed", phase: "completed" });

    const recovered = await recoverInterruptedAgentRuns(root);
    assert.equal(recovered.length, 1);
    assert.equal(recovered[0].id, second.id);
    assert.equal(recovered[0].status, "interrupted");
    assert.equal(recovered[0].checkpoint.stage, "candidate_checked");
    assert.match(recovered[0].message, /service stopped/);

    const runs = await listAgentRuns(root, 10);
    assert.equal(runs.length, 2);
    assert.equal(runs.find(run => run.id === first.id)?.status, "completed");
    assert.equal(runs.find(run => run.id === second.id)?.status, "interrupted");
    assert.equal((await recoverInterruptedAgentRuns(root)).length, 0);
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("cancellation requests are durable and terminal runs remain unchanged", async () => {
  const root = await createWorkspace();
  try {
    const run = await createAgentRun(root, "Check cancellation propagation.");
    const requested = await requestAgentRunCancellation(root, run.id);
    assert.equal(requested.cancelRequested, true);
    assert.equal(requested.phase, "cancelling");
    assert.equal((await getAgentRun(root, run.id)).cancelRequested, true);

    const completed = await updateAgentRun(root, run.id, { status: "completed" });
    const noOp = await requestAgentRunCancellation(root, run.id);
    assert.equal(noOp.status, "completed");
    assert.equal(noOp.cancelRequested, completed.cancelRequested);
    await assert.rejects(
      () => updateAgentRun(root, run.id, { status: "running" }),
      /cannot transition from completed to running/
    );
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("agent run records reject invalid transitions and over-sized recovery checkpoints", async () => {
  const root = await createWorkspace();
  try {
    const run = await createAgentRun(root, "Validate durable checkpoint bounds.");
    await assert.rejects(
      () => updateAgentRun(root, run.id, { status: "unknown" }),
      /status is invalid/
    );
    await assert.rejects(
      () => updateAgentRun(root, run.id, {
        recoveryPatch: {
          summary: "too large",
          edits: [{ path: "src/index.js", find: "x", replace: "x".repeat(110_000), reason: "test" }]
        }
      }),
      /100 KB/
    );
    const unchanged = await getAgentRun(root, run.id);
    assert.equal(unchanged.status, "running");
    assert.equal(unchanged.recoveryPatch, null);
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("agent run state directories refuse symbolic-link redirection", async () => {
  const root = await createWorkspace();
  const outside = await fs.mkdtemp(path.join(os.tmpdir(), "syntra-agent-runs-outside-"));
  try {
    await fs.mkdir(path.join(root, ".syntra"), { recursive: true });
    await fs.symlink(outside, path.join(root, ".syntra", "runs"), process.platform === "win32" ? "junction" : "dir");
    await assert.rejects(
      () => createAgentRun(root, "Do not write through the state directory link."),
      /runs state directory|symbolic link/
    );
    assert.deepEqual(await fs.readdir(outside), []);
  } finally {
    await fs.rm(root, { recursive: true, force: true });
    await fs.rm(outside, { recursive: true, force: true });
  }
});

test("resuming a saved patch rejects source files changed since its checkpoint", async () => {
  const root = await createWorkspace();
  try {
    const original = "export const value = 1;\\n";
    const baseHash = createHash("sha256").update(original).digest("hex");
    const patch = {
      summary: "Update the exported value.",
      edits: [{ path: "src/index.js", find: "value = 1", replace: "value = 2", reason: "Update value." }]
    };
    const valid = await validateStructuredPatch(root, patch, { "src/index.js": baseHash });
    assert.equal(valid.changes[0].beforeHash, baseHash);

    await fs.writeFile(path.join(root, "src", "index.js"), original + "export const extra = true;\\n");
    await assert.rejects(
      () => validateStructuredPatch(root, patch, { "src/index.js": baseHash }),
      /changed after the checkpoint/
    );
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

async function allocatePort() {
  const probe = createServer();
  await new Promise((resolve, reject) => {
    probe.once("error", reject);
    probe.listen(0, "127.0.0.1", resolve);
  });
  const address = probe.address();
  if (!address || typeof address === "string") throw new Error("Could not allocate a local test port.");
  await new Promise((resolve, reject) => probe.close(error => error ? reject(error) : resolve()));
  return address.port;
}

async function startLocalService(root, port) {
  const serverPath = fileURLToPath(new URL("../local/server.mjs", import.meta.url));
  const child = spawn(process.execPath, [serverPath], {
    cwd: path.dirname(path.dirname(serverPath)),
    env: {
      ...process.env,
      SYNTRA_WORKSPACE: root,
      SYNTRA_LOCAL_PORT: String(port),
      SYNTRA_AI_PORT: String(port + 1)
    },
    stdio: "ignore"
  });
  const baseUrl = "http://127.0.0.1:" + port;
  let lastError = null;
  for (let attempt = 0; attempt < 80; attempt += 1) {
    if (child.exitCode !== null) throw new Error("SYNTRA service exited during startup.");
    try {
      const response = await fetch(baseUrl + "/local-api/health", { signal: AbortSignal.timeout(250) });
      if (response.ok) return { child, baseUrl };
    } catch (error) {
      lastError = error;
    }
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  child.kill();
  throw new Error("SYNTRA service did not start: " + (lastError instanceof Error ? lastError.message : "timeout"));
}

async function stopLocalService(child) {
  if (!child || child.exitCode !== null || child.signalCode !== null) return;
  await new Promise(resolve => {
    const timer = setTimeout(() => {
      try { child.kill("SIGKILL"); } catch {}
      resolve();
    }, 3000);
    timer.unref?.();
    child.once("exit", () => {
      clearTimeout(timer);
      resolve();
    });
    child.kill("SIGTERM");
  });
}

test("run history preserves live tasks and a new service process recovers them", { timeout: 20000 }, async () => {
  const root = await createWorkspace();
  const port = await allocatePort();
  let first = null;
  let second = null;
  try {
    first = await startLocalService(root, port);
    const run = await createAgentRun(root, "Exercise durable recovery across an actual service restart.");
    const liveResponse = await fetch(first.baseUrl + "/local-api/agent/runs");
    assert.equal(liveResponse.status, 200);
    const liveRuns = await liveResponse.json();
    assert.equal(liveRuns.runs.find(item => item.id === run.id)?.status, "running");

    await stopLocalService(first.child);
    second = await startLocalService(root, port);
    const recoveredResponse = await fetch(second.baseUrl + "/local-api/agent/runs");
    assert.equal(recoveredResponse.status, 200);
    const recoveredRuns = await recoveredResponse.json();
    assert.equal(recoveredRuns.runs.find(item => item.id === run.id)?.status, "interrupted");
  } finally {
    await stopLocalService(first?.child);
    await stopLocalService(second?.child);
    await fs.rm(root, { recursive: true, force: true });
  }
});

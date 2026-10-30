import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
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

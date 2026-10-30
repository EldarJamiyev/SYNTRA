import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import {
  executeWorkspaceTool,
  listWorkspaceTools
} from "../local/tool-runtime.mjs";

async function createProject() {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "syntra-tools-test-"));
  await fs.mkdir(path.join(root, "src"), { recursive: true });
  await fs.writeFile(path.join(root, "src", "value.js"), [
    "export const first = 1;",
    "export const second = 2;",
    "export const third = 3;",
    ""
  ].join("\n"), "utf8");
  await fs.writeFile(path.join(root, "package.json"), JSON.stringify({
    name: "tool-runtime-fixture",
    version: "1.0.0",
    scripts: { test: "node -e \"require('fs').writeFileSync('check-ran.txt', 'yes')\"" }
  }, null, 2), "utf8");
  return root;
}

async function readAudit(root) {
  return (await fs.readFile(path.join(root, ".syntra", "actions.jsonl"), "utf8"))
    .split(/\r?\n/)
    .filter(Boolean)
    .map(line => JSON.parse(line));
}

test("tool registry exposes structured schemas and explicit permission metadata", () => {
  const tools = listWorkspaceTools();
  const byName = new Map(tools.map(item => [item.name, item]));
  assert.equal(tools.length, 12);
  assert.equal(byName.get("workspace.inspect").approvalRequired, false);
  assert.equal(byName.get("workspace.search").permission, "read_only");
  assert.equal(byName.get("workspace.proposals.stage").permission, "local_state_write");
  assert.equal(byName.get("workspace.proposals.apply").approvalRequired, true);
  assert.equal(byName.get("workspace.proposals.apply-many").approvalRequired, true);
  assert.equal(byName.get("workspace.proposals.apply").permission, "source_write");
  assert.equal(byName.get("workspace.proposals.discard").approvalRequired, true);
  assert.equal(byName.get("workspace.check").approvalRequired, true);
  assert.equal(byName.get("workspace.check").permission, "execute");
  assert.equal(byName.get("git.status").approvalRequired, false);
  assert.equal(byName.get("git.diff-summary").permission, "read_only");
  assert.ok(tools.every(item => item.inputSchema.additionalProperties === false));
  assert.ok(!tools.some(item => /shell|terminal|powershell/i.test(item.name)));
});

test("inspection, search, and line-range reads return structured output and durable audit events", async () => {
  const root = await createProject();
  try {
    const inspect = await executeWorkspaceTool({ root, name: "workspace.inspect" });
    assert.equal(inspect.ok, true);
    assert.equal(inspect.status, "completed");
    assert.equal(inspect.output.name, "tool-runtime-fixture");

    const search = await executeWorkspaceTool({
      root, name: "workspace.search", input: { query: "second", limit: 5 }
    });
    assert.equal(search.ok, true);
    assert.equal(search.output.results[0].path, "src/value.js");

    const read = await executeWorkspaceTool({
      root, name: "workspace.read",
      input: { path: "src/value.js", startLine: 2, endLine: 3 }
    });
    assert.equal(read.ok, true);
    assert.equal(read.output.startLine, 2);
    assert.equal(read.output.endLine, 3);
    assert.equal(read.output.content, "export const second = 2;\nexport const third = 3;");
    assert.equal(read.output.truncated, true);

    const events = await readAudit(root);
    const toolEvents = events.filter(event => event.action === "tool_runtime_event");
    assert.equal(toolEvents.length, 6, "each call has a start and completion audit event");
    assert.equal(new Set(toolEvents.map(event => event.runId)).size, 3);
    assert.ok(toolEvents.every(event => ["started", "completed"].includes(event.status)));
    assert.ok(toolEvents.every(event => event.permission === "read_only"));
    assert.ok(events.every(event => !JSON.stringify(event).includes("export const second")));
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("execution and source-writing tools are blocked until explicit approval", async () => {
  const root = await createProject();
  const sourcePath = path.join(root, "src", "value.js");
  try {
    const deniedCheck = await executeWorkspaceTool({
      root, name: "workspace.check", input: { task: "test" }
    });
    assert.equal(deniedCheck.ok, false);
    assert.equal(deniedCheck.status, "approval_required");
    await assert.rejects(() => fs.stat(path.join(root, "check-ran.txt")), { code: "ENOENT" });

    const approvedCheck = await executeWorkspaceTool({
      root, name: "workspace.check", input: { task: "test" }, approval: true
    });
    assert.equal(approvedCheck.ok, true);
    assert.equal(approvedCheck.output.passed, true);
    assert.equal(await fs.readFile(path.join(root, "check-ran.txt"), "utf8"), "yes");

    const original = await fs.readFile(sourcePath, "utf8");
    const staged = await executeWorkspaceTool({
      root,
      name: "workspace.proposals.stage",
      input: {
        path: "src/value.js",
        content: "export const safe = true;\n"
      }
    });
    assert.equal(staged.ok, true);
    assert.equal(await fs.readFile(sourcePath, "utf8"), original, "staging cannot modify project source");

    const deniedApply = await executeWorkspaceTool({
      root,
      name: "workspace.proposals.apply",
      input: { proposalId: staged.output.id }
    });
    assert.equal(deniedApply.status, "approval_required");
    assert.equal(await fs.readFile(sourcePath, "utf8"), original);

    const applied = await executeWorkspaceTool({
      root,
      name: "workspace.proposals.apply",
      input: { proposalId: staged.output.id },
      approval: true
    });
    assert.equal(applied.ok, true);
    assert.equal(await fs.readFile(sourcePath, "utf8"), "export const safe = true;\n");

    const second = await executeWorkspaceTool({
      root, name: "workspace.proposals.stage",
      input: { path: "src/value.js", content: "export const safe = false;\n" }
    });
    const deniedDiscard = await executeWorkspaceTool({
      root, name: "workspace.proposals.discard",
      input: { proposalIds: [second.output.id] }
    });
    assert.equal(deniedDiscard.status, "approval_required");

    const discarded = await executeWorkspaceTool({
      root, name: "workspace.proposals.discard",
      input: { proposalIds: [second.output.id] }, approval: true
    });
    assert.equal(discarded.ok, true);
    assert.equal(discarded.output.discarded.length, 1);
    assert.equal(await fs.readFile(sourcePath, "utf8"), "export const safe = true;\n");

    const events = await readAudit(root);
    const approvalEvents = events.filter(event => event.action === "tool_runtime_event" && event.status === "approval_required");
    assert.equal(approvalEvents.length, 3);
    assert.ok(events.some(event => event.tool === "workspace.check" && event.status === "completed" && event.approved));
    assert.ok(events.every(event => !JSON.stringify(event).includes("export const safe = false")));
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("multi-file apply is approval-gated and remains transactional", async () => {
  const root = await createProject();
  try {
    const first = await executeWorkspaceTool({
      root, name: "workspace.proposals.stage",
      input: { path: "src/value.js", content: "export const first = 'updated';\n" }
    });
    const second = await executeWorkspaceTool({
      root, name: "workspace.proposals.stage",
      input: { path: "src/added.js", content: "export const added = true;\n" }
    });
    const before = await fs.readFile(path.join(root, "src", "value.js"), "utf8");
    const denied = await executeWorkspaceTool({
      root, name: "workspace.proposals.apply-many",
      input: { proposalIds: [first.output.id, second.output.id] }
    });
    assert.equal(denied.status, "approval_required");
    assert.equal(await fs.readFile(path.join(root, "src", "value.js"), "utf8"), before);
    await assert.rejects(() => fs.stat(path.join(root, "src", "added.js")), { code: "ENOENT" });

    const applied = await executeWorkspaceTool({
      root, name: "workspace.proposals.apply-many",
      input: { proposalIds: [first.output.id, second.output.id] },
      approval: true
    });
    assert.equal(applied.ok, true);
    assert.equal(applied.output.applied.length, 2);
    assert.equal(await fs.readFile(path.join(root, "src", "value.js"), "utf8"), "export const first = 'updated';\n");
    assert.equal(await fs.readFile(path.join(root, "src", "added.js"), "utf8"), "export const added = true;\n");
    const events = await readAudit(root);
    assert.ok(events.some(event => event.tool === "workspace.proposals.apply-many" && event.status === "completed" && event.approved));
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("tool calls against one workspace serialize long-running checks and reads", async () => {
  const root = await createProject();
  try {
    const manifestPath = path.join(root, "package.json");
    const manifest = JSON.parse(await fs.readFile(manifestPath, "utf8"));
    manifest.scripts.test = "node -e \"require('fs').writeFileSync('check-started.txt', 'yes'); setTimeout(() => {}, 180)\"";
    await fs.writeFile(manifestPath, JSON.stringify(manifest, null, 2), "utf8");

    const checkPromise = executeWorkspaceTool({
      root, name: "workspace.check", input: { task: "test" }, approval: true
    });

    const deadline = Date.now() + 3000;
    let started = false;
    while (Date.now() < deadline) {
      started = await fs.access(path.join(root, "check-started.txt")).then(() => true).catch(() => false);
      if (started) break;
      await new Promise(resolve => setTimeout(resolve, 15));
    }
    assert.equal(started, true, "the approved project check should begin before the second tool call");

    const searchPromise = executeWorkspaceTool({
      root, name: "workspace.search", input: { query: "second" }
    });
    const [check, search] = await Promise.all([checkPromise, searchPromise]);
    assert.equal(check.ok, true);
    assert.equal(check.output.passed, true);
    assert.equal(search.ok, true);

    const events = (await readAudit(root))
      .filter(event => event.action === "tool_runtime_event");
    assert.deepEqual(events.map(event => [event.tool, event.status]), [
      ["workspace.check", "started"],
      ["workspace.check", "completed"],
      ["workspace.search", "started"],
      ["workspace.search", "completed"]
    ]);
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("tool input schemas reject unknown fields and unsafe file paths", async () => {
  const root = await createProject();
  try {
    await assert.rejects(
      () => executeWorkspaceTool({ root, name: "workspace.search", input: { query: "value", command: "whoami" } }),
      /Unknown input field/
    );
    await assert.rejects(
      () => executeWorkspaceTool({
        root, name: "workspace.read",
        input: { path: "src/value.js", startLine: 1, endLine: 10_000_001 }
      }),
      /allowed integer range/
    );
    const secret = await executeWorkspaceTool({
      root, name: "workspace.read", input: { path: ".env" }
    });
    assert.equal(secret.ok, false);
    assert.equal(secret.status, "failed");
    assert.match(secret.error, /secret|credential/i);
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("tool runtime fails closed when the audit state path is a symbolic link", async () => {
  const root = await createProject();
  const outside = await fs.mkdtemp(path.join(os.tmpdir(), "syntra-tools-audit-outside-"));
  try {
    await fs.symlink(outside, path.join(root, ".syntra"), process.platform === "win32" ? "junction" : "dir");
    const result = await executeWorkspaceTool({
      root, name: "workspace.proposals.stage",
      input: { path: "src/value.js", content: "export const redirected = true;\n" }
    });
    assert.equal(result.ok, false);
    assert.equal(result.status, "audit_failed");
    assert.deepEqual(await fs.readdir(outside), [], "the tool must not write through an audit symlink");
    assert.equal(await fs.readFile(path.join(root, "src", "value.js"), "utf8"), [
      "export const first = 1;",
      "export const second = 2;",
      "export const third = 3;",
      ""
    ].join("\n"));
  } finally {
    await fs.rm(root, { recursive: true, force: true });
    await fs.rm(outside, { recursive: true, force: true });
  }
});

import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";

import {
  applyFileProposal,
  createProjectPlan,
  inspectWorkspace,
  proposeFileChange,
  readWorkspaceFile,
  searchWorkspace
} from "../local/workspace.mjs";

async function createTempProject() {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "syntra-workspace-"));
  await fs.mkdir(path.join(root, "src"), { recursive: true });
  await fs.mkdir(path.join(root, "tests"), { recursive: true });
  await fs.writeFile(path.join(root, "package.json"), JSON.stringify({
    name: "sample-project",
    scripts: { build: "echo build", test: "node --test", lint: "echo lint" },
    dependencies: { react: "^18.0.0", vite: "^6.0.0" }
  }, null, 2));
  await fs.writeFile(path.join(root, "src", "network.ts"), "export function resolveGateway(host: string) { return host; }\n", "utf8");
  await fs.writeFile(path.join(root, "tests", "network.test.mjs"), "import test from 'node:test'; test('ok', () => {});\n", "utf8");
  return root;
}

test("inspect workspace reports project metadata and source files", async () => {
  const root = await createTempProject();
  const info = await inspectWorkspace(root);
  assert.equal(info.name, "sample-project");
  assert.ok(info.fileCount >= 3);
  assert.ok(info.frameworks.includes("React"));
  assert.ok(info.scripts.build);
});

test("search returns matching paths and line numbers", async () => {
  const root = await createTempProject();
  const results = await searchWorkspace(root, "resolveGateway");
  assert.equal(results.results[0].path, "src/network.ts");
  assert.equal(results.results[0].matches[0].line, 1);
});

test("read exposes content hash and refuses secret files", async () => {
  const root = await createTempProject();
  const read = await readWorkspaceFile(root, "src/network.ts");
  assert.match(read.sha256, /^[a-f0-9]{64}$/);
  await fs.writeFile(path.join(root, ".env"), "SECRET=do-not-read", "utf8");
  await fs.writeFile(path.join(root, ".env.example"), "APP_PORT=3000\\nSECRET=replace-me", "utf8");
  await assert.rejects(() => readWorkspaceFile(root, ".env"), /blocks access/);
  const example = await readWorkspaceFile(root, ".env.example");
  assert.match(example.content, /APP_PORT=3000/);
});

test("proposal requires confirmation and protects concurrent edits", async () => {
  const root = await createTempProject();
  const proposal = await proposeFileChange(root, "src/network.ts", "export const version = 2;\n");
  await assert.rejects(() => applyFileProposal(root, proposal.id, false), /explicit confirmation/);
  await fs.writeFile(path.join(root, "src", "network.ts"), "changed outside proposal\n", "utf8");
  await assert.rejects(() => applyFileProposal(root, proposal.id, true), /changed after the proposal/);
});

test("confirmed proposal applies and keeps a backup", async () => {
  const root = await createTempProject();
  const original = await fs.readFile(path.join(root, "src", "network.ts"), "utf8");
  const proposal = await proposeFileChange(root, "src/network.ts", "export const version = 3;\n");
  const result = await applyFileProposal(root, proposal.id, true);
  assert.equal(result.applied, true);
  assert.equal(await fs.readFile(path.join(root, "src", "network.ts"), "utf8"), "export const version = 3;\n");
  assert.equal(result.backupCreated, true);
  const syntraDir = path.join(root, ".syntra", "backups");
  const entries = await fs.readdir(syntraDir);
  assert.equal(entries.length, 1);
  const backupFiles = await fs.readdir(path.join(syntraDir, entries[0], "src"));
  assert.ok(backupFiles.includes("network.ts"));
  assert.equal(await fs.readFile(path.join(syntraDir, entries[0], "src", "network.ts"), "utf8"), original);
});

test("project plan creates staged workflow", async () => {
  const root = await createTempProject();
  const plan = await createProjectPlan(root, "Build an infrastructure dashboard with tested network diagnostics.");
  assert.equal(plan.planningMode, "deterministic-template");
  assert.equal(plan.tasks.length, 6);
  assert.ok(plan.tasks.some(task => task.phase === "verify"));
  const saved = JSON.parse(await fs.readFile(path.join(root, ".syntra", "plans", "current.json"), "utf8"));
  assert.equal(saved.id, plan.id);
});

test("workspace root cannot be a filesystem root", async () => {
  const root = path.parse(process.cwd()).root;
  await assert.rejects(() => inspectWorkspace(root), /filesystem root/);
});

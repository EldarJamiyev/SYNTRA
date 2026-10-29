import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";

import {
  applyFileProposal,
  applyFileProposals,
  chooseCheck,
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



test("workspace state refuses symbolic links that could redirect audit files outside the project", async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "syntra-state-link-"));
  const outside = await fs.mkdtemp(path.join(os.tmpdir(), "syntra-outside-state-"));
  try {
    await fs.mkdir(path.join(root, "src"), { recursive: true });
    await fs.writeFile(path.join(root, "src", "inside.ts"), "export const safe = true;\n");
    await fs.symlink(outside, path.join(root, ".syntra"), process.platform === "win32" ? "junction" : "dir");
    await assert.rejects(
      () => proposeFileChange(root, "src/inside.ts", "export const safe = false;\n"),
      /must not be a symbolic link/
    );
    await assert.rejects(
      () => createProjectPlan(root, "Build a useful and tested local diagnostics interface."),
      /must not be a symbolic link/
    );
    assert.deepEqual(await fs.readdir(outside), [], "no audit, proposal, or plan files may be written through the link");
  } finally {
    await fs.rm(root, { recursive: true, force: true });
    await fs.rm(outside, { recursive: true, force: true });
  }
});


test("workspace proposal and audit subpaths cannot be redirected by symlinks", async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "syntra-nested-state-"));
  const outside = await fs.mkdtemp(path.join(os.tmpdir(), "syntra-nested-outside-"));
  try {
    await fs.mkdir(path.join(root, ".syntra"), { recursive: true });
    await fs.mkdir(path.join(root, "src"), { recursive: true });
    await fs.writeFile(path.join(root, "src", "network.ts"), "export const network = 'local';\n");
    await fs.symlink(outside, path.join(root, ".syntra", "proposals"), process.platform === "win32" ? "junction" : "dir");
    await assert.rejects(
      () => proposeFileChange(root, "src/network.ts", "export const network = 'changed';\n"),
      /\\.syntra\\/proposals state directory must not be a symbolic link/
    );
    assert.deepEqual(await fs.readdir(outside), []);

    await fs.unlink(path.join(root, ".syntra", "proposals"));
    const outsideAudit = path.join(outside, "outside-audit.jsonl");
    await fs.writeFile(outsideAudit, "keep unchanged\n", "utf8");
    await fs.symlink(outsideAudit, path.join(root, ".syntra", "actions.jsonl"), "file");
    await assert.rejects(
      () => createProjectPlan(root, "Build a tested local network diagnostics tool."),
      /actions\\.jsonl audit log must be a regular file/
    );
    assert.equal(await fs.readFile(outsideAudit, "utf8"), "keep unchanged\n");
  } finally {
    await fs.rm(root, { recursive: true, force: true });
    await fs.rm(outside, { recursive: true, force: true });
  }
});

test("single-file apply rejects proposal files whose saved content no longer matches its hash", async () => {
  const root = await createTempProject();
  try {
    const proposal = await proposeFileChange(root, "src/network.ts", "export const version = 9;\n");
    const savedPath = path.join(root, ".syntra", "proposals", proposal.id + ".json");
    const saved = JSON.parse(await fs.readFile(savedPath, "utf8"));
    saved.content = "export const version = 999;\n";
    await fs.writeFile(savedPath, JSON.stringify(saved), "utf8");
    await assert.rejects(
      () => applyFileProposal(root, proposal.id, true),
      /content-integrity check/
    );
    assert.match((await readWorkspaceFile(root, "src/network.ts")).content, /return host/);
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("multi-file proposal transaction applies together and creates a transaction backup", async () => {
  const root = await createTempProject();
  const originalNetwork = await fs.readFile(path.join(root, "src", "network.ts"), "utf8");
  await fs.writeFile(path.join(root, "src", "printer.ts"), "export const printer = 'offline';\n", "utf8");
  const networkProposal = await proposeFileChange(root, "src/network.ts", "export const gateway = '10.0.0.1';\n");
  const printerProposal = await proposeFileChange(root, "src/printer.ts", "export const printer = 'online';\n");
  const transaction = await applyFileProposals(root, [networkProposal.id, printerProposal.id], true);
  assert.equal(transaction.applied.length, 2);
  assert.equal(transaction.applied.every(item => item.applied), true);
  assert.equal(await fs.readFile(path.join(root, "src", "network.ts"), "utf8"), "export const gateway = '10.0.0.1';\n");
  assert.equal(await fs.readFile(path.join(root, "src", "printer.ts"), "utf8"), "export const printer = 'online';\n");
  assert.equal(await fs.readFile(path.join(root, transaction.backupDirectory, "src", "network.ts"), "utf8"), originalNetwork);
  const audit = await fs.readFile(path.join(root, ".syntra", "actions.jsonl"), "utf8");
  assert.match(audit, /apply_file_proposals_transaction/);
});

test("multi-file proposal transaction rejects stale member before modifying any file", async () => {
  const root = await createTempProject();
  const originalNetwork = await fs.readFile(path.join(root, "src", "network.ts"), "utf8");
  await fs.writeFile(path.join(root, "src", "printer.ts"), "export const printer = 'offline';\n", "utf8");
  const networkProposal = await proposeFileChange(root, "src/network.ts", "export const gateway = '10.0.0.1';\n");
  const printerProposal = await proposeFileChange(root, "src/printer.ts", "export const printer = 'online';\n");
  await fs.writeFile(path.join(root, "src", "printer.ts"), "external update\n", "utf8");
  await assert.rejects(
    () => applyFileProposals(root, [networkProposal.id, printerProposal.id], true),
    /source changed after the proposal was staged/
  );
  assert.equal(await fs.readFile(path.join(root, "src", "network.ts"), "utf8"), originalNetwork,
    "the valid first proposal must not be applied when a later proposal is stale");
  assert.equal(await fs.readFile(path.join(root, "src", "printer.ts"), "utf8"), "external update\n");
});

test("multi-file transaction rejects duplicate IDs and requires explicit confirmation", async () => {
  const root = await createTempProject();
  const proposal = await proposeFileChange(root, "src/network.ts", "export const version = 4;\n");
  await assert.rejects(() => applyFileProposals(root, [proposal.id], false), /explicit confirmation/);
  await assert.rejects(() => applyFileProposals(root, [proposal.id, proposal.id], true), /only once/);
});


test("check selection uses the matching native test runner, not Node for arbitrary tests directories", async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "syntra-runner-"));
  try {
    await fs.writeFile(path.join(root, "Cargo.toml"), '[package]\nname = "fixture"\nversion = "0.1.0"\n');
    await fs.mkdir(path.join(root, "tests"), { recursive: true });
    await fs.writeFile(path.join(root, "tests", "integration.rs"), '#[test]\nfn works() {}\n');
    const rustCheck = await chooseCheck(root, "test", null);
    assert.equal(rustCheck.kind, "command");
    assert.equal(rustCheck.executable, "cargo");
    assert.deepEqual(rustCheck.args, ["test"]);

    await fs.rm(path.join(root, "Cargo.toml"));
    await fs.writeFile(path.join(root, "go.mod"), "module example.com/fixture\n\ngo 1.22\n");
    await fs.writeFile(path.join(root, "tests", "integration_test.go"), "package fixture\n");
    const goCheck = await chooseCheck(root, "test", null);
    assert.equal(goCheck.executable, "go");
    assert.deepEqual(goCheck.args, ["test", "./..."]);
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("Python projects select unittest or pytest only when Python tests exist", async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "syntra-python-runner-"));
  try {
    await fs.writeFile(path.join(root, "requirements.txt"), "numpy>=1.24\n");
    await fs.mkdir(path.join(root, "tests", "unit"), { recursive: true });
    await fs.writeFile(path.join(root, "tests", "unit", "test_sample.py"), "import unittest\n");
    const unit = await chooseCheck(root, "test", null);
    assert.equal(unit.kind, "python-unittest");
    assert.deepEqual(unit.args, ["-m", "unittest", "discover", "-s", "tests/unit", "-v"]);

    await fs.writeFile(path.join(root, "pytest.ini"), "[pytest]\ntestpaths = tests\n");
    const pytest = await chooseCheck(root, "test", null);
    assert.equal(pytest.kind, "python-pytest");
    assert.deepEqual(pytest.args, ["-m", "pytest"]);
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});


test("dotnet projects are discovered and need an actual test project for test verification", async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "syntra-dotnet-runner-"));
  try {
    await fs.mkdir(path.join(root, "tests"), { recursive: true });
    await fs.writeFile(path.join(root, "app.csproj"), "<Project Sdk=\"Microsoft.NET.Sdk\"></Project>\n");
    await fs.writeFile(path.join(root, "tests", "sample.cs"), "public class Sample {}\n");
    const appOnly = await chooseCheck(root, "test", null).catch(error => error);
    assert.equal(appOnly instanceof Error, true);
    assert.match(appOnly.message, /No supported test runner and matching test files/);

    await fs.writeFile(path.join(root, "tests", "Sample.Tests.csproj"),
      "<Project Sdk=\"Microsoft.NET.Sdk\"><ItemGroup><PackageReference Include=\"Microsoft.NET.Test.Sdk\" Version=\"17.0.0\" /></ItemGroup></Project>\n");
    const testProject = await chooseCheck(root, "test", null);
    assert.equal(testProject.executable, "dotnet");
    assert.deepEqual(testProject.args, ["test", "tests/Sample.Tests.csproj"]);

    const inventory = await inspectWorkspace(root);
    assert.equal(inventory.manifests.dotnetProject, true);
    assert.ok(inventory.frameworks.includes(".NET"));
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("empty tests folders cannot be reported as a passing test suite", async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "syntra-no-tests-"));
  try {
    await fs.mkdir(path.join(root, "tests"));
    await assert.rejects(() => chooseCheck(root, "test", null), /No supported test runner and matching test files/);
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
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

test("AI-generated proposal rejects a source file that changed after generation", async () => {
  const root = await createTempProject();
  const loaded = await readWorkspaceFile(root, "src/network.ts");
  const proposalContent = loaded.content + "\n// generated continuation\n";
  await fs.writeFile(path.join(root, "src", "network.ts"), loaded.content + "// concurrent change\n", "utf8");
  await assert.rejects(
    () => proposeFileChange(root, "src/network.ts", proposalContent, loaded.sha256),
    /changed after SYNTRA read it for generation/
  );
});

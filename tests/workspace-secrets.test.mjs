import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import { updateWorkspaceExclusions } from "../local/workspace.mjs";
import { scanWorkspaceSecrets } from "../local/secret-scanner.mjs";
import { executeWorkspaceTool, listWorkspaceTools } from "../local/tool-runtime.mjs";

async function createWorkspace() {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "syntra-secret-scan-"));
  await fs.mkdir(path.join(root, "src"), { recursive: true });
  await fs.mkdir(path.join(root, "fixtures"), { recursive: true });
  return root;
}

test("secret scanner detects common credential patterns without returning secret material", async () => {
  const root = await createWorkspace();
  const githubToken = "ghp_" + "a".repeat(30);
  const stripeKey = "sk_live_" + "b".repeat(24);
  const databaseUrl = "postgres://app:" + "c".repeat(22) + "@db.internal/app";
  try {
    await fs.writeFile(path.join(root, "src", "config.ts"), [
      "const githubToken = '" + githubToken + "';",
      "const stripeSecret = '" + stripeKey + "';",
      "const database = '" + databaseUrl + "';",
      "const configured = { password: 'process.env.DB_PASSWORD' };",
      "const example = { api_key: 'your_api_key_here' };",
      ""
    ].join("\n"), "utf8");

    const result = await scanWorkspaceSecrets(root);
    assert.equal(result.scannedFiles, 1);
    assert.equal(result.findingCount, 3);
    assert.deepEqual(result.findings.map(item => item.type).sort(), [
      "database_url_credentials", "github_token", "stripe_live_secret"
    ]);
    assert.ok(result.findings.every(item => item.path === "src/config.ts"));
    assert.ok(result.findings.every(item => item.line >= 1));
    const serialized = JSON.stringify(result);
    assert.equal(serialized.includes(githubToken), false);
    assert.equal(serialized.includes(stripeKey), false);
    assert.equal(serialized.includes(databaseUrl), false);
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("secret scanner respects workspace exclusions", async () => {
  const root = await createWorkspace();
  const token = "ghp_" + "z".repeat(30);
  try {
    await fs.writeFile(path.join(root, "src", "safe.ts"), "export const enabled = true;\n", "utf8");
    await fs.writeFile(path.join(root, "fixtures", "private.ts"), "const token = '" + token + "';\n", "utf8");
    await updateWorkspaceExclusions(root, { patterns: ["fixtures/**"] });

    const result = await scanWorkspaceSecrets(root);
    assert.equal(result.scannedFiles, 1);
    assert.equal(result.findingCount, 0);
    assert.equal(JSON.stringify(result).includes(token), false);
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("secret scan is available as a read-only audited tool and audit data never contains values", async () => {
  const root = await createWorkspace();
  const token = "ghp_" + "m".repeat(30);
  try {
    await fs.writeFile(path.join(root, "src", "secrets.ts"), "const token = '" + token + "';\n", "utf8");
    const definition = listWorkspaceTools().find(tool => tool.name === "workspace.secrets.scan");
    assert.equal(definition.permission, "read_only");
    assert.equal(definition.approvalRequired, false);

    const result = await executeWorkspaceTool({ root, name: "workspace.secrets.scan" });
    assert.equal(result.ok, true);
    assert.equal(result.output.findingCount, 1);
    assert.equal(result.output.findings[0].type, "github_token");

    const audit = await fs.readFile(path.join(root, ".syntra", "actions.jsonl"), "utf8");
    assert.equal(audit.includes(token), false);
    assert.equal(audit.includes("secrets.ts"), false);
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("scanner gives an explicit empty result for a workspace with no supported matches", async () => {
  const root = await createWorkspace();
  try {
    await fs.writeFile(path.join(root, "src", "safe.ts"), "export const value = 42;\n", "utf8");
    const result = await scanWorkspaceSecrets(root);
    assert.equal(result.findingCount, 0);
    assert.equal(result.findings.length, 0);
    assert.match(result.notice, /best-effort/);
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

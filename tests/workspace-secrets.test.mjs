import assert from "node:assert/strict";
import { createHash } from "node:crypto";
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


test("secret scanner detects additional provider tokens and private-key formats without returning values", async () => {
  const root = await createWorkspace();
  const samples = [
    "GOCSPX-" + "a".repeat(28),
    "hf_" + "b".repeat(34),
    "npm_" + "c".repeat(36),
    "pypi-" + "d".repeat(55),
    "123456789:" + "e".repeat(32),
    "SG." + "f".repeat(20) + "." + "g".repeat(24),
    "SK" + "a".repeat(32),
    "hvs." + "b".repeat(28),
    "key-" + "c".repeat(32),
    "sk-ant-api03-" + "d".repeat(28),
    "AbCdEfGhIjKlMnOpQrStUvWx.123456.Abcdefghijklmnopqrstuvwxyz123456"
  ];
  try {
    const source = [
      ...samples.map((value, index) => "const sample" + index + " = '" + value + "';"),
      "-----BEGIN ENCRYPTED PRIVATE KEY-----",
      "-----BEGIN PGP PRIVATE KEY BLOCK-----",
      ""
    ].join("\n");
    await fs.writeFile(path.join(root, "src", "provider-tokens.ts"), source, "utf8");

    const result = await scanWorkspaceSecrets(root);
    const types = new Set(result.findings.map(item => item.type));
    for (const type of [
      "google_oauth_client_secret",
      "huggingface_token",
      "npm_token",
      "pypi_token",
      "telegram_bot_token",
      "sendgrid_api_key",
      "twilio_api_key",
      "hashicorp_vault_token",
      "mailgun_api_key",
      "anthropic_api_key",
      "discord_bot_token",
      "private_key_header"
    ]) {
      assert.ok(types.has(type), "expected detection rule: " + type);
    }
    assert.equal(types.has("openai_api_key"), false, "Anthropic sk-ant tokens must not also match the OpenAI detector.");
    const serialized = JSON.stringify(result);
    for (const value of samples) assert.equal(serialized.includes(value), false);
    assert.equal(serialized.includes("ENCRYPTED PRIVATE KEY"), false);
    assert.equal(serialized.includes("PGP PRIVATE KEY"), false);
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("secret allowlist suppresses only reviewed detector types for the exact file hash", async () => {
  const root = await createWorkspace();
  const githubToken = "ghp_" + "a".repeat(30);
  const stripeKey = "sk_live_" + "b".repeat(24);
  const source = [
    "const githubToken = '" + githubToken + "';",
    "const stripeSecret = '" + stripeKey + "';",
    ""
  ].join("\n");
  const filePath = "src/config.ts";
  const fileSha256 = createHash("sha256").update(source, "utf8").digest("hex");
  const expiresAt = new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString().slice(0, 10);
  const allowlist = {
    schema: "syntra-secret-allowlist-v1",
    entries: [{
      path: filePath,
      sha256: fileSha256,
      ruleTypes: ["github_token"],
      approved: true,
      reason: "Reviewed public fixture data; this exact token is intentionally non-secret test content.",
      expiresAt
    }]
  };
  try {
    await fs.writeFile(path.join(root, filePath), source, "utf8");
    await fs.mkdir(path.join(root, ".syntra"), { recursive: true });
    await fs.writeFile(
      path.join(root, ".syntra", "secret-allowlist.json"),
      JSON.stringify(allowlist, null, 2),
      "utf8"
    );

    const result = await scanWorkspaceSecrets(root);
    assert.equal(result.allowlistStatus, "loaded");
    assert.equal(result.allowlistEntries, 1);
    assert.equal(result.allowlistedFiles, 1);
    assert.equal(result.allowlistedFindings, 1);
    assert.equal(result.staleAllowlistEntries, 0);
    assert.equal(result.findingCount, 1);
    assert.equal(result.findings[0].type, "stripe_live_secret");
    assert.equal(result.findings[0].fileSha256, fileSha256);
    const serialized = JSON.stringify(result);
    assert.equal(serialized.includes(githubToken), false);
    assert.equal(serialized.includes(stripeKey), false);
    assert.equal(serialized.includes(allowlist.entries[0].reason), false);

    // Any file edit invalidates the allowlist fingerprint, so all matches return.
    await fs.writeFile(path.join(root, filePath), source + "// reviewed hash no longer matches\n", "utf8");
    const changed = await scanWorkspaceSecrets(root);
    assert.equal(changed.allowlistStatus, "loaded");
    assert.equal(changed.allowlistedFindings, 0);
    assert.equal(changed.staleAllowlistEntries, 1);
    assert.equal(changed.findingCount, 2);
    assert.deepEqual(new Set(changed.findings.map(item => item.type)), new Set([
      "github_token", "stripe_live_secret"
    ]));
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("invalid, unapproved, expired, or malformed allowlists fail closed", async () => {
  const root = await createWorkspace();
  const token = "ghp_" + "q".repeat(30);
  try {
    await fs.writeFile(path.join(root, "src", "config.ts"), "const token = '" + token + "';\n", "utf8");
    await fs.mkdir(path.join(root, ".syntra"), { recursive: true });
    const allowlistPath = path.join(root, ".syntra", "secret-allowlist.json");
    const invalidEntry = {
      path: "src/config.ts",
      sha256: "0".repeat(64),
      ruleTypes: ["github_token"],
      approved: false,
      reason: "This entry was not explicitly approved by a reviewer.",
      expiresAt: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString().slice(0, 10)
    };
    await fs.writeFile(allowlistPath, JSON.stringify({
      schema: "syntra-secret-allowlist-v1",
      entries: [invalidEntry]
    }), "utf8");

    const invalid = await scanWorkspaceSecrets(root);
    assert.equal(invalid.allowlistStatus, "invalid");
    assert.equal(invalid.allowlistedFindings, 0);
    assert.equal(invalid.findingCount, 1);
    assert.match(invalid.allowlistWarning, /explicit approval/);

    const expired = {
      ...invalidEntry,
      approved: true,
      expiresAt: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString().slice(0, 10)
    };
    await fs.writeFile(allowlistPath, JSON.stringify({
      schema: "syntra-secret-allowlist-v1",
      entries: [expired]
    }), "utf8");
    const expiredResult = await scanWorkspaceSecrets(root);
    assert.equal(expiredResult.allowlistStatus, "loaded");
    assert.equal(expiredResult.allowlistExpiredEntries, 1);
    assert.equal(expiredResult.allowlistedFindings, 0);
    assert.equal(expiredResult.findingCount, 1);

    await fs.writeFile(allowlistPath, "{not-json", "utf8");
    const malformed = await scanWorkspaceSecrets(root);
    assert.equal(malformed.allowlistStatus, "invalid");
    assert.equal(malformed.allowlistedFindings, 0);
    assert.equal(malformed.findingCount, 1);
    assert.equal(JSON.stringify(malformed).includes(token), false);
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
    assert.match(result.notice, /best-effort/i);
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("secret allowlist changes require explicit audited approval and can be revoked", async () => {
  const root = await createWorkspace();
  const token = "ghp_" + "r".repeat(30);
  const source = "const token = '" + token + "';\n";
  const filePath = "src/config.ts";
  const fileSha256 = createHash("sha256").update(source, "utf8").digest("hex");
  const expiresAt = new Date(Date.now() + 14 * 24 * 60 * 60 * 1000).toISOString().slice(0, 10);
  const input = {
    path: filePath,
    sha256: fileSha256,
    ruleTypes: ["github_token"],
    reason: "Reviewed local integration-test fixture; this value is intentionally non-production test data.",
    expiresAt
  };
  try {
    await fs.writeFile(path.join(root, filePath), source, "utf8");
    const setDefinition = listWorkspaceTools().find(tool => tool.name === "workspace.secrets.allowlist.set");
    const removeDefinition = listWorkspaceTools().find(tool => tool.name === "workspace.secrets.allowlist.remove");
    assert.equal(setDefinition.permission, "local_state_write");
    assert.equal(setDefinition.approvalRequired, true);
    assert.equal(removeDefinition.permission, "destructive_state_write");
    assert.equal(removeDefinition.approvalRequired, true);

    const denied = await executeWorkspaceTool({
      root,
      name: "workspace.secrets.allowlist.set",
      input
    });
    assert.equal(denied.ok, false);
    assert.equal(denied.status, "approval_required");
    await assert.rejects(fs.stat(path.join(root, ".syntra", "secret-allowlist.json")), { code: "ENOENT" });

    const approved = await executeWorkspaceTool({
      root,
      name: "workspace.secrets.allowlist.set",
      input,
      approval: true
    });
    assert.equal(approved.ok, true);
    assert.deepEqual(approved.output.ruleTypes, ["github_token"]);
    assert.equal(approved.output.expiresAt, expiresAt);
    assert.equal(approved.output.fileSha256, fileSha256);

    const scan = await scanWorkspaceSecrets(root);
    assert.equal(scan.findingCount, 0);
    assert.equal(scan.allowlistedFindings, 1);
    assert.equal(scan.allowlistEntries, 1);
    assert.equal(scan.allowlistEntryDetails[0].applied, true);

    const allowlistPath = path.join(root, ".syntra", "secret-allowlist.json");
    const persisted = await fs.readFile(allowlistPath, "utf8");
    assert.equal(persisted.includes(token), false);
    assert.equal(persisted.includes(input.reason), true);
    if (process.platform !== "win32") {
      const stat = await fs.stat(allowlistPath);
      assert.equal(stat.mode & 0o077, 0, "allowlist state must not be group/world readable");
    }

    const deniedRemove = await executeWorkspaceTool({
      root,
      name: "workspace.secrets.allowlist.remove",
      input: { path: filePath, sha256: fileSha256 }
    });
    assert.equal(deniedRemove.ok, false);
    assert.equal(deniedRemove.status, "approval_required");
    assert.equal((await scanWorkspaceSecrets(root)).findingCount, 0);

    const removed = await executeWorkspaceTool({
      root,
      name: "workspace.secrets.allowlist.remove",
      input: { path: filePath, sha256: fileSha256 },
      approval: true
    });
    assert.equal(removed.ok, true);
    assert.equal(removed.output.revoked, true);
    assert.equal((await scanWorkspaceSecrets(root)).findingCount, 1);

    const audit = await fs.readFile(path.join(root, ".syntra", "actions.jsonl"), "utf8");
    assert.equal(audit.includes(token), false);
    assert.equal(audit.includes(filePath), false);
    assert.equal(audit.includes(input.reason), false);
    assert.match(audit, /workspace\.secrets\.allowlist\.set/);
    assert.match(audit, /workspace\.secrets\.allowlist\.remove/);
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("secret allowlist refuses changed file hashes and selected rules that do not match", async () => {
  const root = await createWorkspace();
  const token = "ghp_" + "s".repeat(30);
  const original = "const token = '" + token + "';\n";
  const filePath = "src/config.ts";
  const originalHash = createHash("sha256").update(original, "utf8").digest("hex");
  const input = {
    path: filePath,
    sha256: originalHash,
    ruleTypes: ["github_token"],
    reason: "Reviewed an exact non-production fixture for a narrowly scoped test exception.",
    expiresAt: new Date(Date.now() + 14 * 24 * 60 * 60 * 1000).toISOString().slice(0, 10)
  };
  try {
    await fs.writeFile(path.join(root, filePath), original + "// modified after the scan\n", "utf8");
    const changed = await executeWorkspaceTool({
      root,
      name: "workspace.secrets.allowlist.set",
      input,
      approval: true
    });
    assert.equal(changed.ok, false);
    assert.match(changed.error, /SHA-256 changed/);
    assert.equal((await scanWorkspaceSecrets(root)).findingCount, 1);

    const current = await fs.readFile(path.join(root, filePath), "utf8");
    const currentHash = createHash("sha256").update(current, "utf8").digest("hex");
    const wrongRule = await executeWorkspaceTool({
      root,
      name: "workspace.secrets.allowlist.set",
      input: { ...input, sha256: currentHash, ruleTypes: ["stripe_live_secret"] },
      approval: true
    });
    assert.equal(wrongRule.ok, false);
    assert.match(wrongRule.error, /do not currently match/);
    assert.equal((await scanWorkspaceSecrets(root)).findingCount, 1);
    await assert.rejects(fs.stat(path.join(root, ".syntra", "secret-allowlist.json")), { code: "ENOENT" });
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("secret allowlist update refuses symlinked or malformed local state", async () => {
  const root = await createWorkspace();
  const token = "ghp_" + "t".repeat(30);
  const source = "const token = '" + token + "';\n";
  const filePath = "src/config.ts";
  const sha256 = createHash("sha256").update(source, "utf8").digest("hex");
  const input = {
    path: filePath,
    sha256,
    ruleTypes: ["github_token"],
    reason: "Reviewed a test-only fixture with a clear reason and narrow scope.",
    expiresAt: new Date(Date.now() + 14 * 24 * 60 * 60 * 1000).toISOString().slice(0, 10)
  };
  try {
    await fs.writeFile(path.join(root, filePath), source, "utf8");
    await fs.mkdir(path.join(root, ".syntra"), { recursive: true });
    const outside = path.join(root, "outside.json");
    await fs.writeFile(outside, JSON.stringify({ schema: "syntra-secret-allowlist-v1", entries: [] }), "utf8");
    const target = path.join(root, ".syntra", "secret-allowlist.json");
    try {
      await fs.symlink(outside, target);
    } catch (error) {
      if (error && ["EPERM", "EACCES", "ENOTSUP"].includes(error.code)) {
        await fs.rm(path.join(root, ".syntra"), { recursive: true, force: true });
        await fs.mkdir(path.join(root, ".syntra"), { recursive: true });
        await fs.writeFile(target, "{malformed", "utf8");
      } else {
        throw error;
      }
    }
    const result = await executeWorkspaceTool({
      root,
      name: "workspace.secrets.allowlist.set",
      input,
      approval: true
    });
    assert.equal(result.ok, false);
    assert.equal((await scanWorkspaceSecrets(root)).findingCount, 1);
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});


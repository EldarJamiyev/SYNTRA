import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import {
  getWorkspaceIndexStatus,
  refreshWorkspaceIndex,
  searchWorkspaceSymbols
} from "../local/workspace-index.mjs";

async function createProject() {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "syntra-index-"));
  await fs.mkdir(path.join(root, "src"), { recursive: true });
  await fs.writeFile(path.join(root, "src", "service.ts"), [
    "export interface GatewayOptions { host: string }",
    "export async function resolveGateway(host: string): Promise<string> { return host; }",
    "export const createGateway = (name: string) => ({ name });",
    "export class GatewayClient {}",
    "export type GatewayState = 'ready' | 'offline';",
    ""
  ].join("\n"), "utf8");
  await fs.writeFile(path.join(root, "src", "worker.py"), [
    "class JobRunner:",
    "    async def execute_job(self, job_id):",
    "        return job_id",
    ""
  ].join("\n"), "utf8");
  await fs.writeFile(path.join(root, "src", "probe.go"), [
    "package probe",
    "func ResolveProbe(host string) error { return nil }",
    "type ProbeOptions struct {}",
    ""
  ].join("\n"), "utf8");
  await fs.writeFile(path.join(root, "src", "lib.rs"), [
    "pub struct RouteTable {}",
    "pub async fn resolve_route() {}",
    "pub trait RouteProvider {}",
    ""
  ].join("\n"), "utf8");
  await fs.writeFile(path.join(root, ".env"), "SYNTRA_SECRET=never-index-this\n", "utf8");
  return root;
}

test("workspace symbol index extracts bounded declarations without persisting source text", async () => {
  const root = await createProject();
  try {
    const statusBefore = await getWorkspaceIndexStatus(root);
    assert.equal(statusBefore.available, false);

    const refreshed = await refreshWorkspaceIndex(root);
    assert.equal(refreshed.available, true);
    assert.equal(refreshed.parsedFiles, 4);
    assert.equal(refreshed.reusedFiles, 0);
    assert.equal(refreshed.fileCount, 4);
    assert.ok(refreshed.symbolCount >= 9);

    const gateway = await searchWorkspaceSymbols(root, "resolveGateway");
    assert.equal(gateway.results.length, 1);
    assert.equal(gateway.results[0].name, "resolveGateway");
    assert.equal(gateway.results[0].kind, "function");
    assert.equal(gateway.results[0].path, "src/service.ts");
    assert.equal(gateway.results[0].line, 2);

    const python = await searchWorkspaceSymbols(root, "JobRunner");
    assert.equal(python.results[0].path, "src/worker.py");
    const go = await searchWorkspaceSymbols(root, "ProbeOptions");
    assert.equal(go.results[0].path, "src/probe.go");
    const rust = await searchWorkspaceSymbols(root, "RouteProvider");
    assert.equal(rust.results[0].path, "src/lib.rs");

    const stored = await fs.readFile(path.join(root, ".syntra", "workspace-index-v1.json"), "utf8");
    assert.equal(stored.includes("SYNTRA_SECRET"), false);
    assert.equal(stored.includes("return host"), false);
    assert.equal(stored.includes("GatewayOptions { host"), false);
    const status = await getWorkspaceIndexStatus(root);
    assert.equal(status.available, true);
    assert.equal(status.symbolCount, refreshed.symbolCount);
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("refresh reuses unchanged parses, detects same-size edits, and prunes deleted source files", async () => {
  const root = await createProject();
  try {
    const first = await refreshWorkspaceIndex(root);
    const second = await refreshWorkspaceIndex(root);
    assert.equal(second.parsedFiles, 0);
    assert.equal(second.reusedFiles, first.fileCount);

    const sourcePath = path.join(root, "src", "service.ts");
    const original = await fs.readFile(sourcePath, "utf8");
    const changed = original.replace("resolveGateway", "refreshGateway");
    assert.equal(Buffer.byteLength(changed), Buffer.byteLength(original));
    await fs.writeFile(sourcePath, changed, "utf8");

    const staleSearch = await searchWorkspaceSymbols(root, "resolveGateway");
    assert.equal(staleSearch.results.length, 0);
    assert.equal(staleSearch.staleFiles, 1);

    await fs.rm(path.join(root, "src", "worker.py"));
    const third = await refreshWorkspaceIndex(root);
    assert.equal(third.parsedFiles, 1);
    assert.equal(third.reusedFiles, 2);
    assert.equal(third.removedFiles, 1);

    const oldName = await searchWorkspaceSymbols(root, "resolveGateway");
    assert.equal(oldName.results.length, 0);
    const newName = await searchWorkspaceSymbols(root, "refreshGateway");
    assert.equal(newName.results[0].name, "refreshGateway");
    assert.equal(newName.results[0].path, "src/service.ts");
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("symbol search validates query bounds and ignores unavailable indexes", async () => {
  const root = await createProject();
  try {
    const unavailable = await searchWorkspaceSymbols(root, "Gateway", 5);
    assert.equal(unavailable.available, false);
    await assert.rejects(() => searchWorkspaceSymbols(root, "x", 5), /2-160 characters/);
    await assert.rejects(() => searchWorkspaceSymbols(root, "Gateway", 21), /integer from 1 to 20/);
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("workspace index refuses a symbolic-link index file", async t => {
  const root = await createProject();
  const outside = path.join(os.tmpdir(), "syntra-index-outside-" + process.pid + ".json");
  try {
    await refreshWorkspaceIndex(root);
    await fs.writeFile(outside, JSON.stringify({ schemaVersion: 1 }), "utf8");
    await fs.unlink(path.join(root, ".syntra", "workspace-index-v1.json"));
    try {
      await fs.symlink(outside, path.join(root, ".syntra", "workspace-index-v1.json"), "file");
    } catch (error) {
      if (["EPERM", "EACCES", "ENOTSUP", "UNKNOWN"].includes(error?.code)) {
        t.skip("The current operating system does not permit creating file symlinks.");
        return;
      }
      throw error;
    }
    await assert.rejects(() => getWorkspaceIndexStatus(root), /regular file, not a symbolic link/);
    await assert.rejects(() => refreshWorkspaceIndex(root), /regular file, not a symbolic link/);
  } finally {
    await fs.rm(root, { recursive: true, force: true });
    await fs.rm(outside, { force: true });
  }
});

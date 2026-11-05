import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import {
  getWorkspaceIndexStatus,
  refreshWorkspaceIndex,
  searchWorkspaceSemantic,
  searchWorkspaceSymbols
} from "../local/workspace-index.mjs";

async function createProject() {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "syntra-index-"));
  await fs.mkdir(path.join(root, "src"), { recursive: true });
  await fs.writeFile(path.join(root, "src", "service.ts"), [
    "export interface GatewayOptions { host: string }",
    "export async function resolveGateway(host: string): Promise<string> { return host; }",
    "export const createGateway = (name: string) => ({ name });",
    "export async function",
    "  resolveMultilineGateway(",
    "    host: string",
    "  ): Promise<string> { return host; }",
    "export const createMultilineGateway = (",
    "  name: string",
    ") => ({ name });",
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

    const multiline = await searchWorkspaceSymbols(root, "resolveMultilineGateway");
    assert.equal(multiline.results[0].path, "src/service.ts");
    assert.equal(multiline.results[0].line, 4);
    const multilineArrow = await searchWorkspaceSymbols(root, "createMultilineGateway");
    assert.equal(multilineArrow.results[0].path, "src/service.ts");
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

    const stored = await fs.readFile(path.join(root, ".syntra", "workspace-index-v3.json"), "utf8");
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


test("workspace graph indexes local imports, documentation links, and test ownership", async () => {
  const root = await createProject();
  try {
    await fs.mkdir(path.join(root, "docs"), { recursive: true });
    await fs.mkdir(path.join(root, "tests"), { recursive: true });
    await fs.writeFile(path.join(root, "src", "service.ts"), [
      "export function resolveGateway(host: string) { return host; }",
      "export class GatewayService {}",
      ""
    ].join("\n"), "utf8");
    await fs.writeFile(path.join(root, "src", "consumer.ts"), [
      'import { resolveGateway } from "./service";',
      "export function useGateway() { return resolveGateway('localhost'); }",
      ""
    ].join("\n"), "utf8");
    await fs.writeFile(path.join(root, "tests", "service.test.ts"), [
      'import { resolveGateway } from "../src/service";',
      "export function testResolveGateway() { return resolveGateway('localhost'); }",
      ""
    ].join("\n"), "utf8");
    await fs.writeFile(path.join(root, "docs", "architecture.md"), [
      "# Architecture",
      "",
      "The [gateway service](../src/service.ts) is used by the [consumer](../src/consumer.ts).",
      ""
    ].join("\n"), "utf8");

    const refreshed = await refreshWorkspaceIndex(root);
    assert.equal(refreshed.documentationFileCount, 1);
    assert.equal(refreshed.testFileCount, 1);
    assert.ok(refreshed.importEdgeCount >= 2);
    assert.ok(refreshed.callEdgeCount >= 2);
    assert.equal(refreshed.documentationReferenceCount, 2);
    assert.equal(refreshed.testOwnershipCount, 1);

    const service = await searchWorkspaceSymbols(root, "resolveGateway");
    const implementation = service.results.find(item =>
      item.path === "src/service.ts" && item.name === "resolveGateway"
    );
    assert.ok(implementation);
    assert.equal(implementation.imports.length, 0);
    assert.ok(implementation.importedBy.includes("src/consumer.ts"));
    assert.ok(implementation.importedBy.includes("tests/service.test.ts"));
    assert.ok(implementation.calledBy.includes("src/consumer.ts"));
    assert.ok(implementation.calledBy.includes("tests/service.test.ts"));
    assert.ok(implementation.referencedByDocs.includes("docs/architecture.md"));
    assert.ok(implementation.ownedTests.includes("tests/service.test.ts"));

    const consumer = await searchWorkspaceSymbols(root, "useGateway");
    assert.ok(consumer.results[0].imports.includes("src/service.ts"));
    assert.ok(consumer.results[0].calls.includes("src/service.ts"));

    const testSymbol = await searchWorkspaceSymbols(root, "testResolveGateway");
    assert.equal(testSymbol.results[0].role, "test");
    assert.ok(testSymbol.results[0].testOwners.includes("src/service.ts"));

    const status = await getWorkspaceIndexStatus(root);
    assert.equal(status.documentationFileCount, 1);
    assert.equal(status.testFileCount, 1);
    assert.equal(status.importEdgeCount, refreshed.importEdgeCount);
    assert.equal(status.callEdgeCount, refreshed.callEdgeCount);
    assert.equal(status.documentationReferenceCount, 2);
    assert.equal(status.testOwnershipCount, 1);

    const stored = await fs.readFile(path.join(root, ".syntra", "workspace-index-v3.json"), "utf8");
    assert.equal(stored.includes("resolveGateway(host: string)"), false);
    assert.equal(stored.includes("used by the"), false);
    assert.equal(stored.includes("The [gateway service]"), false);
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("Go module imports and Rust crate module declarations resolve to local files", async () => {
  const root = await createProject();
  try {
    await fs.mkdir(path.join(root, "internal", "store"), { recursive: true });
    await fs.mkdir(path.join(root, "src"), { recursive: true });
    await fs.writeFile(path.join(root, "go.mod"), "module example.com/syntra-fixture\n", "utf8");
    await fs.writeFile(path.join(root, "cmd.go"), [
      "package main",
      'import "example.com/syntra-fixture/internal/store"',
      "func main() {}",
      ""
    ].join("\n"), "utf8");
    await fs.writeFile(path.join(root, "internal", "store", "store.go"), [
      "package store",
      "func OpenStore() {}",
      ""
    ].join("\n"), "utf8");
    await fs.writeFile(path.join(root, "src", "lib.rs"), [
      "mod networking;",
      "use crate::networking::GatewayClient;",
      ""
    ].join("\n"), "utf8");
    await fs.mkdir(path.join(root, "pkg"), { recursive: true });
    await fs.mkdir(path.join(root, "tests"), { recursive: true });
    await fs.writeFile(path.join(root, "pkg", "engine.py"), [
      "def run_engine():",
      "    return True",
      ""
    ].join("\n"), "utf8");
    await fs.writeFile(path.join(root, "tests", "test_engine.py"), [
      "from pkg.engine import run_engine",
      "def test_run_engine():",
      "    return run_engine()",
      ""
    ].join("\n"), "utf8");
    await fs.writeFile(path.join(root, "src", "networking.rs"), [
      "pub struct GatewayClient;",
      ""
    ].join("\n"), "utf8");

    await refreshWorkspaceIndex(root);
    const goEntry = await searchWorkspaceSymbols(root, "OpenStore");
    assert.ok(goEntry.results[0].importedBy.includes("cmd.go"));
    const rustEntry = await searchWorkspaceSymbols(root, "GatewayClient");
    assert.ok(rustEntry.results[0].importedBy.includes("src/lib.rs"));
    assert.ok(rustEntry.results[0].imports.length === 0);
    const pythonEntry = await searchWorkspaceSymbols(root, "run_engine");
    assert.ok(pythonEntry.results.some(item =>
      item.path === "pkg/engine.py" && item.importedBy.includes("tests/test_engine.py")
        && item.ownedTests.includes("tests/test_engine.py")
    ));
    const pythonTest = await searchWorkspaceSymbols(root, "test_run_engine");
    assert.ok(pythonTest.results.some(item => item.testOwners.includes("pkg/engine.py")));
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
    await fs.unlink(path.join(root, ".syntra", "workspace-index-v3.json"));
    try {
      await fs.symlink(outside, path.join(root, ".syntra", "workspace-index-v3.json"), "file");
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


test("semantic source search retrieves deep code chunks, reuses local vectors, and stores no source text", async () => {
  const root = await createProject();
  try {
    const lines = Array.from({ length: 150 }, (_, index) =>
      "// unrelated bootstrap configuration note " + index + " for the application runtime"
    );
    lines.push(
      "export function normalizeWebhookEvent(payload: unknown) {",
      "  // Convert incoming webhook payloads into the canonical event envelope before dispatch.",
      "  return { type: 'canonical', payload };",
      "}"
    );
    await fs.writeFile(path.join(root, "src", "router.ts"), lines.join("\n") + "\n", "utf8");
    await refreshWorkspaceIndex(root);

    let requests = 0;
    let activeCheckpointIdentity = "a".repeat(64);
    const fetchImpl = async (url, options) => {
      requests += 1;
      assert.equal(url, "http://127.0.0.1:3002/model/embed");
      assert.equal(options.redirect, "error");
      const body = JSON.parse(options.body);
      assert.equal(body.device, "auto");
      const embeddings = body.texts.map(input =>
        /canonical event envelope/i.test(input)
          ? [1, 0, 0, 0, 0, 0, 0, 0]
          : [0, 1, 0, 0, 0, 0, 0, 0]
      );
      return new Response(JSON.stringify({ embeddings, checkpoint_identity: activeCheckpointIdentity }), {
        status: 200, headers: { "Content-Type": "application/json" }
      });
    };

    const first = await searchWorkspaceSemantic(
      root,
      "Where are incoming webhook payloads normalized into a canonical event envelope before dispatch?",
      8,
      { embedModel: "syntra-native", fetchImpl }
    );
    assert.equal(first.retrievalMode, "semantic");
    assert.ok(first.semanticChunkCoverage > first.semanticCoverage,
      "the index should store multiple chunk vectors for multi-chunk source files");
    const router = first.results.find(item => item.path === "src/router.ts");
    assert.ok(router, JSON.stringify(first.results));
    assert.ok(router.lineStart > 1, "the matching result must point beyond the file prefix");
    assert.ok(router.lineEnd >= router.lineStart);
    assert.match(router.excerpt, /canonical event envelope/);

    const cachePath = path.join(root, ".syntra", "workspace-semantic-v2.json");
    const stored = await fs.readFile(cachePath, "utf8");
    const cache = JSON.parse(stored);
    assert.equal(cache.schemaVersion, 2);
    assert.equal(cache.model, "syntra-native:" + "a".repeat(64));
    assert.equal(first.checkpointIdentity, "a".repeat(64));
    assert.ok(cache.entries.length > first.semanticCoverage);
    assert.ok(cache.entries.every(entry =>
      typeof entry.path === "string"
      && Number.isInteger(entry.chunkIndex)
      && /^[a-f0-9]{64}$/.test(entry.textHash)
      && Array.isArray(entry.embedding)
    ));
    assert.equal(stored.includes("canonical event envelope"), false);
    assert.equal(stored.includes("embeddingText"), false);

    const callsBeforeCachedSearch = requests;
    const second = await searchWorkspaceSemantic(
      root,
      "Find the code that prepares webhook events for dispatch.",
      8,
      { embedModel: "syntra-native", fetchImpl }
    );
    assert.equal(second.retrievalMode, "semantic");
    assert.equal(requests, callsBeforeCachedSearch + 1,
      "unchanged source chunks should be reused; only the new query needs embedding");

    activeCheckpointIdentity = "c".repeat(64);
    const callsBeforeCheckpointChange = requests;
    const refreshedCheckpoint = await searchWorkspaceSemantic(
      root,
      "Where are webhook payloads normalized?",
      8,
      { embedModel: "syntra-native", fetchImpl }
    );
    assert.equal(refreshedCheckpoint.checkpointIdentity, activeCheckpointIdentity);
    assert.ok(requests >= callsBeforeCheckpointChange + 2,
      "a different native checkpoint must regenerate cached source vectors as well as the query vector");
    const refreshedCache = JSON.parse(await fs.readFile(cachePath, "utf8"));
    assert.equal(refreshedCache.model, "syntra-native:" + activeCheckpointIdentity);

    await fs.appendFile(path.join(root, "src", "router.ts"), "\n// changed after indexing\n", "utf8");
    const stale = await searchWorkspaceSemantic(
      root,
      "Where is the canonical webhook event normalized?",
      8,
      { embedModel: "syntra-native", fetchImpl }
    );
    assert.ok(stale.staleFiles >= 1);
    assert.equal(stale.results.some(item => item.path === "src/router.ts"), false,
      "source excerpts must not be returned when the indexed file hash is stale");
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("semantic source search falls back cleanly when native checkpoint embeddings are unavailable", async () => {
  const root = await createProject();
  try {
    await refreshWorkspaceIndex(root);
    const result = await searchWorkspaceSemantic(root, "Gateway client routing", 6, {
      embedModel: "syntra-native",
      fetchImpl: async () => { throw new Error("connection refused"); }
    });
    assert.equal(result.retrievalMode, "lexical_fallback");
    assert.match(result.semanticError, /connection refused/);
    assert.ok(result.results.length > 0);
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import {
  refreshWorkspaceSemanticIndex,
  searchWorkspaceSemantic,
  getSemanticIndexStatus
} from "../local/semantic-index.mjs";

async function makeWorkspace() {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "syntra-local-bm25-test-"));
  await fs.mkdir(path.join(root, "src"), { recursive: true });
  await fs.writeFile(path.join(root, "src", "network.mjs"), [
    "export function resolveGateway(address) {",
    "  // Resolve the default network gateway for an IPv4 client.",
    "  // Gateway configuration is used by network diagnostics and routing.",
    "  return address;",
    "}",
    ""
  ].join("\n"), "utf8");
  await fs.writeFile(path.join(root, "src", "database.mjs"), [
    "export function createDatabaseConnection(options) {",
    "  // Configure the database pool and database credentials.",
    "  return options;",
    "}",
    ""
  ].join("\n"), "utf8");
  return root;
}

test("local BM25 indexing and search work without an embedding service or network call", async () => {
  const root = await makeWorkspace();
  let attemptedNetworkRequest = false;
  const forbiddenFetch = async () => {
    attemptedNetworkRequest = true;
    throw new Error("Network access must not be attempted.");
  };
  try {
    const built = await refreshWorkspaceSemanticIndex(root, { fetchImpl: forbiddenFetch });
    assert.equal(built.available, true);
    assert.equal(built.indexMethod, "syntra-local-bm25-v1");
    assert.ok(built.chunkCount >= 2);
    assert.equal(attemptedNetworkRequest, false);

    const status = await getSemanticIndexStatus(root);
    assert.equal(status.available, true);
    assert.equal(status.indexMethod, "syntra-local-bm25-v1");

    const result = await searchWorkspaceSemantic(root, "default network gateway configuration", 5);
    assert.equal(result.available, true);
    assert.equal(result.indexMethod, "syntra-local-bm25-v1");
    assert.equal(result.results[0].path, "src/network.mjs");
    assert.match(result.results[0].excerpt, /resolveGateway/);

    const database = await searchWorkspaceSemantic(root, "database connection pool", 5);
    assert.equal(database.results[0].path, "src/database.mjs");

    await fs.writeFile(path.join(root, "src", "network.mjs"), "export const changed = true;\n", "utf8");
    const stale = await searchWorkspaceSemantic(root, "default network gateway configuration", 5);
    assert.equal(stale.results.length, 0);
    assert.ok(stale.staleResults > 0);
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

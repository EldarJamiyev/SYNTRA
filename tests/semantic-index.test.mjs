import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { refreshWorkspaceSemanticIndex, searchWorkspaceSemantic, getSemanticIndexStatus, splitIntoChunks } from "../local/semantic-index.mjs";

async function makeWorkspace() {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "syntra-semantic-test-"));
  await fs.mkdir(path.join(root, "src"), { recursive: true });
  await fs.writeFile(path.join(root, "src", "network.mjs"), [
    "export function resolveGateway(address) {",
    "  // Resolve the default network gateway for an IPv4 client.",
    "  return address;",
    "}",
    ""
  ].join("\n"), "utf8");
  return root;
}
test("semantic retrieval accepts only the native SYNTRA checkpoint as its embedding model", async () => {
  const root = await makeWorkspace();
  try {
    await assert.rejects(
      () => refreshWorkspaceSemanticIndex(root, { model: "external-model" }),
      /only its native checkpoint embeddings/
    );
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

function fakeEmbeddings(url, options) {
  assert.equal(url, "http://127.0.0.1:3002/model/embed");
  assert.equal(options.redirect, "error", "native embedding requests must not follow redirects");
  const body = JSON.parse(options.body);
  assert.equal(body.device, "auto");
  assert.equal(body.max_length, 256);
  const input = body.texts;
  return Promise.resolve(new Response(JSON.stringify({
    checkpoint_identity: fakeEmbeddings.checkpointIdentity || "d".repeat(64),
    embeddings: input.map(text => {
      const value = String(text).toLowerCase();
      return [value.includes("gateway") || value.includes("network") ? 1 : 0,
        value.includes("database") ? 1 : 0, 0.25, 0.5, 0.75, 0.1, 0.2, 0.3];
    })
  }), { status: 200 }));
}
test("semantic index invalidates vectors when the native checkpoint changes", async () => {
  const root = await makeWorkspace();
  try {
    fakeEmbeddings.checkpointIdentity = "d".repeat(64);
    const options = { fetchImpl: fakeEmbeddings, model: "syntra-native" };
    const built = await refreshWorkspaceSemanticIndex(root, options);
    assert.equal(built.available, true);
    assert.equal(built.checkpointIdentity, fakeEmbeddings.checkpointIdentity);
    assert.ok(built.chunkCount > 0);
    assert.equal((await getSemanticIndexStatus(root)).checkpointIdentity, fakeEmbeddings.checkpointIdentity);
    const result = await searchWorkspaceSemantic(root, "find gateway configuration", 5, options);
    assert.equal(result.available, true);
    assert.equal(result.results[0].path, "src/network.mjs");
    assert.match(result.results[0].excerpt, /resolveGateway/);

    fakeEmbeddings.checkpointIdentity = "e".repeat(64);
    const changedCheckpoint = await searchWorkspaceSemantic(root, "find gateway configuration", 5, options);
    assert.equal(changedCheckpoint.available, false);
    assert.equal(changedCheckpoint.reason, "native_checkpoint_changed_refresh_required");
    const refreshed = await refreshWorkspaceSemanticIndex(root, options);
    assert.equal(refreshed.checkpointIdentity, fakeEmbeddings.checkpointIdentity);
    assert.equal(refreshed.reusedChunks, 0, "a new checkpoint must not reuse old native vectors");

    await fs.writeFile(path.join(root, "src", "network.mjs"), "export const changed = true;\n", "utf8");
    const stale = await searchWorkspaceSemantic(root, "find gateway configuration", 5, options);
    assert.equal(stale.results.length, 0);
    assert.ok(stale.staleResults > 0);
  } finally {
    delete fakeEmbeddings.checkpointIdentity;
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("splitIntoChunks aligns chunks to declaration boundaries when source exceeds one window", () => {
  const lines = [
    ...Array.from({ length: 18 }, (_, i) => `const line${i} = ${i}; // setup preamble line`),
    "export function computeChecksum(payload) {",
    ...Array.from({ length: 24 }, (_, i) => `  const step${i} = payload + ${i};`),
    "  return payload;",
    "}"
  ].join("\n");
  const chunks = splitIntoChunks("src/checksum.mjs", lines);
  assert.ok(chunks.length >= 2);
  assert.equal(chunks[0].endLine, 18);
  assert.match(chunks[1].text, /export function computeChecksum/);
});


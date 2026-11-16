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
  assert.equal(options.redirect, "error", "native tokenizer/embedding requests must not follow redirects");
  const body = JSON.parse(options.body);
  assert.equal(body.max_length, 256);
  if (url === "http://127.0.0.1:3002/model/tokenize") {
    assert.ok(Array.isArray(body.texts));
    return Promise.resolve(new Response(JSON.stringify({
      checkpoint_identity: fakeEmbeddings.checkpointIdentity || "d".repeat(64),
      token_counts: body.texts.map(text => Array.from(String(text)).length),
      max_input_tokens: fakeEmbeddings.maxInputTokens || 256
    }), { status: 200 }));
  }
  assert.equal(url, "http://127.0.0.1:3002/model/embed");
  assert.equal(body.device, "auto");
  const input = body.texts;
  const tokenCounts = input.map(text => Array.from(String(text)).length);
  const maximum = fakeEmbeddings.maxInputTokens || 256;
  if (tokenCounts.some(count => count > maximum)) {
    const index = tokenCounts.findIndex(count => count > maximum);
    return Promise.resolve(new Response(JSON.stringify({
      error: "Text at index " + index + " requires " + tokenCounts[index] +
        " tokenizer tokens, exceeding the embedding limit of " + maximum + "; input was not truncated."
    }), { status: 400 }));
  }
  return Promise.resolve(new Response(JSON.stringify({
    checkpoint_identity: fakeEmbeddings.checkpointIdentity || "d".repeat(64),
    input_token_counts: tokenCounts,
    max_input_tokens: fakeEmbeddings.maxInputTokens || 256,
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

test("splitIntoChunks preserves complete very long source lines without character truncation", () => {
  const sourceLine = "x".repeat(6000);
  const chunks = splitIntoChunks("src/generated.mjs", sourceLine);
  assert.equal(chunks.map(chunk => chunk.text).join(""), sourceLine);
  assert.equal(chunks.length, 3);
  assert.equal(chunks[0].startColumn, 1);
  assert.equal(chunks[0].endColumn, 2401);
  assert.equal(chunks[1].startColumn, 2401);
  assert.equal(chunks[1].endColumn, 4801);
  assert.equal(chunks[2].startColumn, 4801);
  assert.equal(chunks[2].endColumn, 6001);
});

test("semantic indexing splits long source lines to exact tokenizer offsets before embedding", async () => {
  const root = await makeWorkspace();
  const originalLine = "const source = '" + "x".repeat(420) + " gateway configuration lives here " + "y".repeat(420) + "';";
  try {
    await fs.writeFile(path.join(root, "src", "long-line.mjs"), originalLine + "\n", "utf8");
    fakeEmbeddings.checkpointIdentity = "f".repeat(64);
    fakeEmbeddings.maxInputTokens = 128;
    const options = { fetchImpl: fakeEmbeddings, model: "syntra-native" };
    const built = await refreshWorkspaceSemanticIndex(root, options);
    assert.equal(built.available, true);
    assert.equal(built.maxInputTokens, 128);
    assert.ok(built.splitForContextChunks > 0);

    const reused = await refreshWorkspaceSemanticIndex(root, options);
    assert.equal(reused.reusedChunks, built.chunkCount);
    assert.equal(reused.splitForContextChunks, 0);
    assert.equal(reused.maxInputTokens, 128,
      "a fully reused index must retain the selected checkpoint's actual context limit");

    const result = await searchWorkspaceSemantic(root, "gateway configuration", 8, options);
    const match = result.results.find(item => item.path === "src/long-line.mjs");
    assert.ok(match, "the late part of the long source line must have its own embedding");
    assert.ok(match.tokenCount <= 128);
    assert.ok(Number.isInteger(match.startColumn));
    assert.ok(Number.isInteger(match.endColumn));
    assert.match(match.excerpt, /gateway configuration/);
    const expected = Array.from(originalLine).slice(match.startColumn - 1, match.endColumn - 1).join("");
    assert.equal(match.excerpt, expected, "retrieved source columns must match the exact embedded fragment");

    const status = await getSemanticIndexStatus(root);
    assert.equal(status.schemaVersion, 2);
    assert.equal(status.maxInputTokens, 128);
  } finally {
    delete fakeEmbeddings.checkpointIdentity;
    delete fakeEmbeddings.maxInputTokens;
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("semantic query text is rejected instead of being silently clipped to model context", async () => {
  const root = await makeWorkspace();
  try {
    fakeEmbeddings.checkpointIdentity = "a".repeat(64);
    fakeEmbeddings.maxInputTokens = 64;
    const options = { fetchImpl: fakeEmbeddings, model: "syntra-native" };
    await refreshWorkspaceSemanticIndex(root, options);
    await assert.rejects(
      () => searchWorkspaceSemantic(root, "q".repeat(65), 5, options),
      /requires 65 tokenizer tokens.*input was not truncated/
    );
  } finally {
    delete fakeEmbeddings.checkpointIdentity;
    delete fakeEmbeddings.maxInputTokens;
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


import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { refreshWorkspaceSemanticIndex, searchWorkspaceSemantic, getSemanticIndexStatus } from "../local/semantic-index.mjs";

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
function fakeEmbeddings(url, options) {
  assert.equal(url, "http://127.0.0.1:11434/api/embed");
  const input = JSON.parse(options.body).input;
  return Promise.resolve(new Response(JSON.stringify({
    embeddings: input.map(text => {
      const value = String(text).toLowerCase();
      return [value.includes("gateway") || value.includes("network") ? 1 : 0,
        value.includes("database") ? 1 : 0, 0.25, 0.5, 0.75, 0.1, 0.2, 0.3];
    })
  }), { status: 200 }));
}
test("semantic index stores vectors and revalidates source hashes before returning excerpts", async () => {
  const root = await makeWorkspace();
  try {
    const options = { fetchImpl: fakeEmbeddings, model: "test-embedding" };
    const built = await refreshWorkspaceSemanticIndex(root, options);
    assert.equal(built.available, true);
    assert.ok(built.chunkCount > 0);
    assert.equal((await getSemanticIndexStatus(root)).available, true);
    const result = await searchWorkspaceSemantic(root, "find gateway configuration", 5, options);
    assert.equal(result.available, true);
    assert.equal(result.results[0].path, "src/network.mjs");
    assert.match(result.results[0].excerpt, /resolveGateway/);
    await fs.writeFile(path.join(root, "src", "network.mjs"), "export const changed = true;\n", "utf8");
    const stale = await searchWorkspaceSemantic(root, "find gateway configuration", 5, options);
    assert.equal(stale.results.length, 0);
    assert.ok(stale.staleResults > 0);
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

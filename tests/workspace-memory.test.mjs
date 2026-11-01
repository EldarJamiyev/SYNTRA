import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import {
  addWorkspaceMemory,
  deleteWorkspaceMemory,
  listWorkspaceMemory,
  searchWorkspaceMemory,
  updateWorkspaceMemory
} from "../local/workspace-memory.mjs";

async function createWorkspace() {
  return fs.mkdtemp(path.join(os.tmpdir(), "syntra-project-memory-"));
}

test("project memory is explicitly user-authored, local to a workspace, and searchable", async () => {
  const firstRoot = await createWorkspace();
  const secondRoot = await createWorkspace();
  try {
    const empty = await listWorkspaceMemory(firstRoot);
    assert.equal(empty.total, 0);
    assert.deepEqual(empty.notes, []);

    const created = await addWorkspaceMemory(firstRoot, {
      title: "API error handling",
      content: "All API handlers must return a stable error envelope and include a request identifier.",
      category: "convention",
      tags: ["api", "backend", "errors"]
    });
    assert.equal(created.storedLocally, true);
    assert.equal(created.note.owner, "user");
    assert.equal(created.note.category, "convention");
    assert.deepEqual(created.note.tags, ["api", "backend", "errors"]);
    assert.equal(created.total, 1);

    const results = await searchWorkspaceMemory(firstRoot, "request identifier");
    assert.equal(results.results.length, 1);
    assert.equal(results.results[0].id, created.note.id);
    assert.match(results.results[0].content, /stable error envelope/);
    assert.equal((await listWorkspaceMemory(secondRoot)).total, 0);

    const stored = await fs.readFile(path.join(firstRoot, ".syntra", "project-memory-v1.json"), "utf8");
    assert.match(stored, /API error handling/);
    assert.match(stored, /"owner": "user"/);
  } finally {
    await fs.rm(firstRoot, { recursive: true, force: true });
    await fs.rm(secondRoot, { recursive: true, force: true });
  }
});

test("semantic project-memory retrieval uses local embeddings and caches only user-authored notes", async () => {
  const root = await createWorkspace();
  try {
    const network = await addWorkspaceMemory(root, {
      title: "Network access convention",
      content: "Resolve service names through the internal DNS resolver and use a bounded lookup timeout.",
      category: "convention",
      tags: ["network", "dns"]
    });
    await addWorkspaceMemory(root, {
      title: "Interface theme",
      content: "Keep the interface typography and spacing consistent across panels.",
      category: "decision",
      tags: ["frontend", "design"]
    });
    let requests = 0;
    const fetchImpl = async (url, options) => {
      requests += 1;
      assert.equal(url, "http://127.0.0.1:11434/api/embed");
      const body = JSON.parse(options.body);
      assert.equal(body.model, "test-embedding-model");
      assert.equal(body.truncate, false);
      const embeddings = body.input.map(input => {
        const text = input.toLowerCase();
        return /dns|network access|service name|name resolution|lookup timeout/.test(text)
          ? [1, 0, 0, 0, 0, 0, 0, 0]
          : [0, 1, 0, 0, 0, 0, 0, 0];
      });
      return new Response(JSON.stringify({ embeddings }), {
        status: 200, headers: { "Content-Type": "application/json" }
      });
    };
    const first = await searchWorkspaceMemory(root, "name resolution policy", 6, {
      embedModel: "test-embedding-model", fetchImpl
    });
    assert.equal(first.retrievalMode, "hybrid_semantic");
    assert.equal(first.embeddingModel, "test-embedding-model");
    assert.equal(first.results[0].id, network.note.id);
    assert.equal(first.semanticCoverage, 2);
    assert.equal(requests, 1);

    const second = await searchWorkspaceMemory(root, "dns resolver convention", 6, {
      embedModel: "test-embedding-model", fetchImpl
    });
    assert.equal(second.retrievalMode, "hybrid_semantic");
    assert.equal(second.results[0].id, network.note.id);
    assert.equal(requests, 2, "the second search should reuse cached note vectors and embed only the query");

    const cache = JSON.parse(await fs.readFile(
      path.join(root, ".syntra", "project-memory-embeddings-v1.json"), "utf8"
    ));
    assert.equal(cache.schemaVersion, 1);
    assert.equal(cache.model, "test-embedding-model");
    assert.equal(cache.root, root);
    assert.equal(cache.entries.length, 2);
    assert.ok(cache.entries.every(entry => !("content" in entry)));
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("semantic project-memory search falls back to BM25 when the local embedding model is unavailable", async () => {
  const root = await createWorkspace();
  try {
    await addWorkspaceMemory(root, {
      title: "Error response",
      content: "Return stable API error codes and request identifiers.",
      category: "convention"
    });
    const results = await searchWorkspaceMemory(root, "request identifiers", 6, {
      embedModel: "missing-local-embedding-model",
      fetchImpl: async () => { throw new Error("connection refused"); }
    });
    assert.equal(results.retrievalMode, "bm25_fallback");
    assert.match(results.semanticError, /connection refused/);
    assert.equal(results.results.length, 1);
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("notes support optimistic updates and deletion without losing other notes", async () => {
  const root = await createWorkspace();
  try {
    const first = await addWorkspaceMemory(root, {
      title: "Runtime",
      content: "Use the local Node service for workspace tools.",
      category: "architecture"
    });
    const second = await addWorkspaceMemory(root, {
      title: "Tests",
      content: "Run Node tests before opening a pull request.",
      category: "testing",
      tags: ["ci"]
    });

    const updated = await updateWorkspaceMemory(root, {
      id: first.note.id,
      expectedUpdatedAt: first.note.updatedAt,
      title: "Local runtime",
      content: "Use the local Node API service for workspace tools and memory.",
      category: "decision",
      tags: ["runtime", "local"]
    });
    assert.equal(updated.note.title, "Local runtime");
    assert.equal(updated.note.category, "decision");
    assert.deepEqual(updated.note.tags, ["runtime", "local"]);
    await assert.rejects(() => updateWorkspaceMemory(root, {
      id: first.note.id,
      expectedUpdatedAt: first.note.updatedAt,
      title: "Stale edit"
    }), /changed after it was opened/);

    await assert.rejects(() => deleteWorkspaceMemory(root, {
      id: first.note.id, expectedUpdatedAt: first.note.updatedAt
    }), /changed after it was opened/);

    const removed = await deleteWorkspaceMemory(root, {
      id: first.note.id, expectedUpdatedAt: updated.note.updatedAt
    });
    assert.equal(removed.deleted, true);
    assert.equal(removed.total, 1);
    const remaining = await listWorkspaceMemory(root);
    assert.deepEqual(remaining.notes.map(note => note.id), [second.note.id]);
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("project memory rejects likely credentials and enforces bounded note input", async () => {
  const root = await createWorkspace();
  try {
    await assert.rejects(() => addWorkspaceMemory(root, {
      title: "Production config",
      content: "API_KEY=sk-abcdefghijklmnopqrstuvwx",
      category: "other"
    }), /may contain a credential or secret/);
    await assert.rejects(() => addWorkspaceMemory(root, {
      title: "Database",
      content: "postgres://app:correct-horse-battery-staple@db.internal/app",
      category: "architecture"
    }), /may contain a credential or secret/);
    await assert.rejects(() => addWorkspaceMemory(root, {
      title: "x".repeat(121), content: "A note", category: "other"
    }), /1-120 characters/);
    await assert.rejects(() => addWorkspaceMemory(root, {
      title: "A note", content: " ", category: "other"
    }), /1-2400 characters/);
    await assert.rejects(() => addWorkspaceMemory(root, {
      title: "A note", content: "Safe content", category: "unsupported"
    }), /supported project-memory category/);
    await assert.rejects(() => addWorkspaceMemory(root, {
      title: "A note", content: "Safe content", category: "other", tags: ["bad/tag"]
    }), /Each note tag/);
    assert.equal((await listWorkspaceMemory(root)).total, 0);
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("invalid workspace memory is never silently overwritten", async () => {
  const root = await createWorkspace();
  try {
    await fs.mkdir(path.join(root, ".syntra"), { recursive: true });
    await fs.writeFile(path.join(root, ".syntra", "project-memory-v1.json"), JSON.stringify({
      schemaVersion: 99, root, updatedAt: new Date().toISOString(), notes: []
    }), "utf8");
    await assert.rejects(() => listWorkspaceMemory(root), /invalid or belongs to a different workspace/);
    await assert.rejects(() => addWorkspaceMemory(root, {
      title: "New", content: "Do not overwrite invalid store", category: "other"
    }), /invalid or belongs to a different workspace/);
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("project memory refuses symbolic-link stores", async t => {
  const root = await createWorkspace();
  const outside = path.join(os.tmpdir(), "syntra-memory-outside-" + process.pid + ".json");
  try {
    await fs.mkdir(path.join(root, ".syntra"), { recursive: true });
    await fs.writeFile(outside, JSON.stringify({
      schemaVersion: 1, root, updatedAt: new Date().toISOString(), notes: []
    }), "utf8");
    try {
      await fs.symlink(outside, path.join(root, ".syntra", "project-memory-v1.json"), "file");
    } catch (error) {
      if (["EPERM", "EACCES", "ENOTSUP", "UNKNOWN"].includes(error?.code)) {
        t.skip("The current operating system does not permit creating file symlinks.");
        return;
      }
      throw error;
    }
    await assert.rejects(() => listWorkspaceMemory(root), /regular file, not a symbolic link/);
    await assert.rejects(() => addWorkspaceMemory(root, {
      title: "No overwrite", content: "Must reject symlink store", category: "other"
    }), /regular file, not a symbolic link/);
  } finally {
    await fs.rm(root, { recursive: true, force: true });
    await fs.rm(outside, { force: true });
  }
});

import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import {
  getWorkspaceExclusions,
  listWorkspaceIndexCandidates,
  readWorkspaceFile,
  searchWorkspace,
  updateWorkspaceExclusions
} from "../local/workspace.mjs";
import {
  refreshWorkspaceIndex,
  searchWorkspaceSymbols
} from "../local/workspace-index.mjs";

async function createProject() {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "syntra-exclusions-"));
  await fs.mkdir(path.join(root, "src", "generated"), { recursive: true });
  await fs.mkdir(path.join(root, "docs", "internal"), { recursive: true });
  await fs.mkdir(path.join(root, "fixtures"), { recursive: true });
  await fs.writeFile(path.join(root, "src", "app.ts"), "export function visibleFeature() { return true; }\n", "utf8");
  await fs.writeFile(path.join(root, "src", "generated", "schema.ts"), "export function generatedSchema() { return true; }\n", "utf8");
  await fs.writeFile(path.join(root, "src", "legacy.generated.ts"), "export function legacyGenerated() { return true; }\n", "utf8");
  await fs.writeFile(path.join(root, "docs", "internal", "plan.md"), "private architecture note\n", "utf8");
  await fs.writeFile(path.join(root, "fixtures", "sample.ts"), "export function fixtureOnly() { return true; }\n", "utf8");
  return root;
}

test("workspace exclusions apply to discovery, full-text search, direct reads, and symbol indexing", async () => {
  const root = await createProject();
  try {
    const initial = await getWorkspaceExclusions(root);
    assert.deepEqual(initial.patterns, []);
    assert.equal(initial.updatedAt, null);

    const saved = await updateWorkspaceExclusions(root, {
      patterns: ["src/generated/**", "*.generated.ts", "docs/internal/**"],
      expectedUpdatedAt: null
    });
    assert.equal(saved.patterns.length, 3);
    assert.ok(saved.updatedAt);

    const candidates = await listWorkspaceIndexCandidates(root);
    assert.deepEqual(candidates.files.map(file => file.path).sort(), ["fixtures/sample.ts", "src/app.ts"]);
    const search = await searchWorkspace(root, "private architecture note");
    assert.equal(search.results.some(item => item.path === "docs/internal/plan.md"), false);
    await assert.rejects(
      () => readWorkspaceFile(root, "src/generated/schema.ts"),
      /excludes this path/
    );
    await assert.rejects(
      () => readWorkspaceFile(root, "docs/internal/plan.md"),
      /excludes this path/
    );

    await refreshWorkspaceIndex(root);
    const generated = await searchWorkspaceSymbols(root, "generatedSchema");
    const visible = await searchWorkspaceSymbols(root, "visibleFeature");
    assert.equal(generated.results.length, 0);
    assert.equal(visible.results[0].path, "src/app.ts");

    const reloaded = await getWorkspaceExclusions(root);
    assert.deepEqual(reloaded.patterns, saved.patterns);
    assert.equal(reloaded.updatedAt, saved.updatedAt);
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("changing exclusions invalidates concurrent edits and malformed patterns are rejected", async () => {
  const root = await createProject();
  try {
    const saved = await updateWorkspaceExclusions(root, { patterns: ["docs/**"], expectedUpdatedAt: null });
    await assert.rejects(
      () => updateWorkspaceExclusions(root, { patterns: ["fixtures/**"], expectedUpdatedAt: null }),
      /changed after they were loaded/
    );
    for (const pattern of ["../outside", "/absolute/path", "C:/secret", "src\\private", ".syntra/**", "!src/private/**"]) {
      await assert.rejects(
        () => updateWorkspaceExclusions(root, { patterns: [pattern] }),
        /Workspace exclusions|relative glob patterns|cannot contain|reserved/
      );
    }
    const cleared = await updateWorkspaceExclusions(root, {
      patterns: [],
      expectedUpdatedAt: saved.updatedAt
    });
    assert.deepEqual(cleared.patterns, []);
    assert.equal((await getWorkspaceExclusions(root)).patterns.length, 0);
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("invalid exclusion state is never silently overwritten", async () => {
  const root = await createProject();
  try {
    await fs.mkdir(path.join(root, ".syntra"), { recursive: true });
    await fs.writeFile(path.join(root, ".syntra", "workspace-exclusions-v1.json"), "{not-json", "utf8");
    await assert.rejects(() => getWorkspaceExclusions(root), /invalid JSON/);
    await assert.rejects(
      () => updateWorkspaceExclusions(root, { patterns: [] }),
      /invalid JSON/
    );
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

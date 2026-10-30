import assert from "node:assert/strict";
import test from "node:test";
import {
  WORKSPACE_PATH_KEY,
  loadWorkspacePath,
  normalizeWorkspacePath,
  saveWorkspacePath
} from "../src/workspace-settings.js";

function memoryStorage(seed = null) {
  let value = seed;
  return {
    getItem(key) { assert.equal(key, WORKSPACE_PATH_KEY); return value; },
    setItem(key, next) { assert.equal(key, WORKSPACE_PATH_KEY); value = next; }
  };
}

test("selected workspace paths are stored and restored locally", () => {
  const storage = memoryStorage();
  assert.equal(saveWorkspacePath("C:\\Projects\\SYNTRA", storage), true);
  assert.equal(loadWorkspacePath(storage), "C:\\Projects\\SYNTRA");
});

test("invalid or oversized workspace paths are rejected", () => {
  assert.equal(normalizeWorkspacePath(null), null);
  assert.equal(normalizeWorkspacePath("  "), null);
  assert.equal(normalizeWorkspacePath("C:\\folder\nnext"), null);
  assert.equal(normalizeWorkspacePath("x".repeat(4097)), null);
});

test("local storage failures do not prevent chat startup", () => {
  const blocked = {
    getItem() { throw new Error("storage denied"); },
    setItem() { throw new Error("quota exceeded"); }
  };
  assert.equal(loadWorkspacePath(blocked), null);
  assert.equal(saveWorkspacePath("C:\\Projects", blocked), false);
});

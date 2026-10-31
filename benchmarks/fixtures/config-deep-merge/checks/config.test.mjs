import assert from "node:assert/strict";
import test from "node:test";
import { mergeConfig } from "../src/config.mjs";
test("mergeConfig recursively preserves unspecified nested defaults", () => {
  const base = { api: { host: "localhost", port: 3000 }, enabled: true };
  const override = { api: { port: 8080 } };
  assert.deepEqual(mergeConfig(base, override), { api: { host: "localhost", port: 8080 }, enabled: true });
});
test("mergeConfig replaces arrays instead of merging their entries", () => {
  assert.deepEqual(mergeConfig({ flags: ["a", "b"] }, { flags: ["c"] }), { flags: ["c"] });
});
test("mergeConfig does not mutate either input", () => {
  const base = { nested: { keep: 1, replace: 2 } };
  const override = { nested: { replace: 3 } };
  mergeConfig(base, override);
  assert.deepEqual(base, { nested: { keep: 1, replace: 2 } });
  assert.deepEqual(override, { nested: { replace: 3 } });
});

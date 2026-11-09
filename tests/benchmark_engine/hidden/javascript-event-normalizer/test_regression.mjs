import assert from "node:assert/strict";
import path from "node:path";
import { pathToFileURL } from "node:url";

const workspace = process.env.SYNTRA_BENCHMARK_WORKSPACE;
assert.ok(workspace, "The isolated fixture workspace must be available to hidden tests.");
const moduleUrl = pathToFileURL(path.join(workspace, "src", "normalize-event.mjs")).href;
const { normalizeEvent } = await import(moduleUrl);

assert.deepEqual(normalizeEvent({
  type: "ready",
  timestamp: 0,
  payload: { source: "local" },
}), {
  type: "ready",
  timestamp: 0,
  payload: { source: "local" },
});
assert.equal(normalizeEvent(null), null);

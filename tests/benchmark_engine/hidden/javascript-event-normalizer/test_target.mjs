import assert from "node:assert/strict";
import path from "node:path";
import { pathToFileURL } from "node:url";

const workspace = process.env.SYNTRA_BENCHMARK_WORKSPACE;
assert.ok(workspace, "The isolated fixture workspace must be available to hidden tests.");
const moduleUrl = pathToFileURL(path.join(workspace, "src", "normalize-event.mjs")).href;
const { normalizeEvent } = await import(moduleUrl);

const payload = { id: 7, nested: { retained: true } };
const valid = normalizeEvent(JSON.stringify({
  type: "  DEVICE.STATE_CHANGED  ",
  timestamp: "2026-11-08T10:00:00.000Z",
  payload,
}));
assert.deepEqual(valid, {
  type: "device.state_changed",
  timestamp: Date.parse("2026-11-08T10:00:00.000Z"),
  payload,
});
assert.notEqual(valid.payload, payload, "The normalized payload must be copied, not aliased.");
assert.notEqual(valid.payload.nested, payload.nested, "Nested payload data must also be copied.");

assert.equal(normalizeEvent("{malformed"), null, "Malformed JSON must not escape as an exception.");
assert.equal(normalizeEvent([]), null, "Array roots are not event records.");
assert.equal(normalizeEvent({ type: "   ", timestamp: 1, payload: {} }), null);
assert.equal(normalizeEvent({ type: "ready", timestamp: -1, payload: {} }), null);
assert.equal(normalizeEvent({ type: "ready", timestamp: true, payload: {} }), null);
assert.equal(normalizeEvent({ type: "ready", timestamp: 1, payload: [] }), null);

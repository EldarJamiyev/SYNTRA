import assert from "node:assert/strict";
import path from "node:path";
import { pathToFileURL } from "node:url";

const workspace = process.env.SYNTRA_BENCHMARK_WORKSPACE;
assert.ok(workspace, "The isolated fixture workspace must be available to hidden tests.");
const moduleUrl = pathToFileURL(path.join(workspace, "src", "map-concurrent.mjs")).href;
const { mapConcurrent } = await import(moduleUrl);
const delay = milliseconds => new Promise(resolve => setTimeout(resolve, milliseconds));

const values = [0, 1, 2, 3, 4, 5, 6, 7];
let active = 0;
let maxActive = 0;
const ordered = await mapConcurrent(values, async (value, index) => {
  active += 1;
  maxActive = Math.max(maxActive, active);
  await delay((values.length - index) * 2);
  active -= 1;
  return { index, value: value * 10 };
}, { concurrency: 3 });

assert.equal(maxActive, 3, "The mapper must never exceed the configured concurrency.");
assert.deepEqual(
  ordered,
  values.map((value, index) => ({ index, value: value * 10 })),
  "Results must preserve input order even when mapper operations finish out of order.",
);
assert.equal(active, 0, "Every started mapper must finish before mapConcurrent resolves.");

let defaultActive = 0;
let defaultMaxActive = 0;
await mapConcurrent([0, 1, 2, 3, 4, 5], async value => {
  defaultActive += 1;
  defaultMaxActive = Math.max(defaultMaxActive, defaultActive);
  await delay(2);
  defaultActive -= 1;
  return value;
});
assert.equal(defaultMaxActive, 4, "The documented default concurrency must be four.");

const original = [1, 2, 3];
const snapshot = await mapConcurrent(original, async (value, index) => {
  if (index === 0) original.push(4);
  await delay(1);
  return value;
}, { concurrency: 2 });
assert.deepEqual(snapshot, [1, 2, 3], "The input sequence must be snapshotted before asynchronous work begins.");

const synchronous = await mapConcurrent([5, 2, 1], (value, index) => value + index, { concurrency: 1 });
assert.deepEqual(synchronous, [5, 3, 3], "Synchronous mapper results must also be supported.");

const syncFailure = new Error("synchronous worker failure");
await assert.rejects(
  mapConcurrent([0, 1], value => {
    if (value === 0) throw syncFailure;
    return value;
  }, { concurrency: 1 }),
  error => error === syncFailure,
  "Synchronous mapper throws must reject with their original reason.",
);

const failure = new Error("worker failed");
const started = [];
let inFlightSettled = false;
await assert.rejects(
  mapConcurrent([0, 1, 2, 3, 4], async value => {
    started.push(value);
    if (value === 0) {
      await delay(5);
      throw failure;
    }
    await delay(25);
    inFlightSettled = true;
    return value;
  }, { concurrency: 2 }),
  error => error === failure,
  "The first mapper failure must be preserved.",
);
assert.deepEqual(started, [0, 1], "No queued mapper may start after a failure is observed.");
assert.equal(inFlightSettled, true, "The function must wait for already-running mappers to settle before rejecting.");

for (const concurrency of [0, -1, 1.5, 65, NaN, Infinity]) {
  await assert.rejects(
    mapConcurrent([1], value => value, { concurrency }),
    RangeError,
    "Concurrency must be a finite integer between 1 and 64.",
  );
}
for (const options of [null, [], "invalid"]) {
  await assert.rejects(
    mapConcurrent([], value => value, options),
    TypeError,
    "Options must be a non-array object.",
  );
}
await assert.rejects(mapConcurrent(null, value => value), TypeError);
await assert.rejects(mapConcurrent([1], null), TypeError);

import assert from "node:assert/strict";
import path from "node:path";
import { pathToFileURL } from "node:url";

const workspace = process.env.SYNTRA_BENCHMARK_WORKSPACE;
assert.ok(workspace, "The isolated fixture workspace must be available to hidden tests.");
const moduleUrl = pathToFileURL(path.join(workspace, "src", "map-concurrent.mjs")).href;
const { mapConcurrent } = await import(moduleUrl);

assert.deepEqual(
  await mapConcurrent([2, 4, 6], value => value / 2),
  [1, 2, 3],
);
assert.deepEqual(await mapConcurrent([], value => value), []);

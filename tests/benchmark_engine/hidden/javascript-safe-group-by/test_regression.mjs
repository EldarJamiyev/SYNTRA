import assert from "node:assert/strict";
import path from "node:path";
import { pathToFileURL } from "node:url";

const workspace = process.env.SYNTRA_BENCHMARK_WORKSPACE;
assert.ok(workspace, "The isolated fixture workspace must be available to hidden tests.");
const moduleUrl = pathToFileURL(path.join(workspace, "src", "group-by.mjs")).href;
const { groupBy } = await import(moduleUrl);

const grouped = groupBy(["ant", "ape", "bear"], value => value[0]);
assert.deepEqual(grouped.a, ["ant", "ape"]);
assert.deepEqual(grouped.b, ["bear"]);

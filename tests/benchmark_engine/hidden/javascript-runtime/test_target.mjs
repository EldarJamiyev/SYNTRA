import assert from "node:assert/strict";
import path from "node:path";
import { pathToFileURL } from "node:url";

const workspace = process.env.SYNTRA_BENCHMARK_WORKSPACE;
assert.ok(workspace, "SYNTRA_BENCHMARK_WORKSPACE must point to the isolated fixture workspace");
const modulePath = path.join(workspace, "src", "value.mjs");
const valueModule = await import(pathToFileURL(modulePath).href);
assert.equal(valueModule.readValue(), 41, "the runtime must load source from the isolated workspace");

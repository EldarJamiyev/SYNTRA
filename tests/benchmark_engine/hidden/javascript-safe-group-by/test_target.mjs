import assert from "node:assert/strict";
import path from "node:path";
import { pathToFileURL } from "node:url";

const workspace = process.env.SYNTRA_BENCHMARK_WORKSPACE;
assert.ok(workspace, "The isolated fixture workspace must be available to hidden tests.");
const moduleUrl = pathToFileURL(path.join(workspace, "src", "group-by.mjs")).href;
const { groupBy } = await import(moduleUrl);

const calls = [];
const records = [
  { key: "__proto__", value: 1 },
  { key: "constructor", value: 2 },
  { key: "toString", value: 3 },
  { key: "__proto__", value: 4 },
  { key: "", value: 5 },
];
const grouped = groupBy(records, (record, index) => {
  calls.push(index);
  return record.key;
});
assert.deepEqual(calls, [0, 1, 2, 3, 4], "The selector must run once per item with its original index.");
assert.equal(Object.getPrototypeOf(grouped), null, "The output record must not inherit prototype keys.");
assert.deepEqual(grouped["__proto__"], [records[0], records[3]]);
assert.deepEqual(grouped.constructor, [records[1]]);
assert.deepEqual(grouped.toString, [records[2]]);
assert.deepEqual(grouped[""], [records[4]]);
assert.deepEqual(records.map(record => record.value), [1, 2, 3, 4, 5], "The input must remain unchanged.");

for (const invalid of [null, {}, "not-an-array"]) {
  assert.throws(() => groupBy(invalid, () => "key"), TypeError);
}
assert.throws(() => groupBy([], null), TypeError);
assert.throws(() => groupBy([1], () => 1), TypeError, "Group keys must be strings.");
assert.equal(Object.getPrototypeOf(groupBy([], () => "unused")), null);

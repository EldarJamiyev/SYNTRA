import assert from "node:assert/strict";
import test from "node:test";
import { chunkArray } from "../src/array.mjs";
test("splits values into ordered chunks", () => {
  assert.deepEqual(chunkArray([1, 2, 3, 4, 5], 2), [[1, 2], [3, 4], [5]]);
  assert.deepEqual(chunkArray([1, 2, 3], 3), [[1, 2, 3]]);
  assert.deepEqual(chunkArray([], 2), []);
});
test("rejects invalid values and sizes", () => {
  for (const value of [null, {}, "abc"]) assert.deepEqual(chunkArray(value, 2), []);
  for (const size of [0, -1, 1.5, NaN, Infinity, Number.MAX_SAFE_INTEGER + 1]) assert.deepEqual(chunkArray([1, 2], size), []);
});
test("does not mutate the source array", () => {
  const source = [1, 2, 3];
  chunkArray(source, 2);
  assert.deepEqual(source, [1, 2, 3]);
});

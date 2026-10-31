import assert from "node:assert/strict";
import test from "node:test";
import { clamp } from "../src/math.mjs";
test("clamp returns the lower bound for values below the range", () => assert.equal(clamp(-5, 0, 10), 0));
test("clamp returns the upper bound for values above the range", () => assert.equal(clamp(15, 0, 10), 10));
test("clamp preserves values inside the inclusive range", () => {
  assert.equal(clamp(5, 0, 10), 5);
  assert.equal(clamp(0, 0, 10), 0);
  assert.equal(clamp(10, 0, 10), 10);
});

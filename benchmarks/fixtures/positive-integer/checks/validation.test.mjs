import assert from "node:assert/strict";
import test from "node:test";
import { parsePositiveInteger } from "../src/validation.mjs";
test("accepts positive safe integers", () => {
  assert.equal(parsePositiveInteger(12), 12);
  assert.equal(parsePositiveInteger("42"), 42);
});
test("rejects values outside the positive safe integer domain", () => {
  for (const value of [0, -1, 1.5, Infinity, NaN, Number.MAX_SAFE_INTEGER + 1, "", "  ", "1.2", null, {}, true]) {
    assert.equal(parsePositiveInteger(value), null);
  }
});

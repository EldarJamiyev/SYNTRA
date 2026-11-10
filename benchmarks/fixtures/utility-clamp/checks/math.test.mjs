import assert from "node:assert/strict";
import { clamp } from "../src/math.mjs";
assert.equal(clamp(-5, 0, 10), 0);
assert.equal(clamp(15, 0, 10), 10);
assert.equal(clamp(5, 0, 10), 5);
assert.equal(clamp(0, 0, 10), 0);
assert.equal(clamp(10, 0, 10), 10);

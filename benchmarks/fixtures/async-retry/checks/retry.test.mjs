import assert from "node:assert/strict";
import { retry } from "../src/retry.mjs";

let attempts = 0;
const value = await retry(async attempt => {
  attempts += 1;
  if (attempt < 3) throw new Error("temporary");
  return "ready";
}, { maxAttempts: 3 });
assert.equal(value, "ready");
assert.equal(attempts, 3);

attempts = 0;
await assert.rejects(() => retry(async () => {
  attempts += 1;
  throw new Error("permanent");
}, { maxAttempts: 5, shouldRetry: () => false }), /permanent/);
assert.equal(attempts, 1);

attempts = 0;
await assert.rejects(() => retry(async () => {
  attempts += 1;
  throw new Error("always failing");
}, { maxAttempts: 2 }), /always failing/);
assert.equal(attempts, 2);

await assert.rejects(() => retry(null), /function/i);
await assert.rejects(() => retry(async () => 1, { maxAttempts: 0 }), /maxAttempts/i);

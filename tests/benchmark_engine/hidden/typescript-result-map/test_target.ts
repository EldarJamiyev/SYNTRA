import { mapResult } from "../workspace/src/map-result.js";
import type { Result } from "../workspace/src/result.js";

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

const errorValue = { code: "E_REMOTE", detail: { retryable: false } };
const failed: Result<number, typeof errorValue> = { ok: false, error: errorValue };
let failureMapperCalls = 0;
const failedMapped: Result<string, typeof errorValue> = mapResult(failed, value => {
  failureMapperCalls += 1;
  return String(value);
});
assert(!failedMapped.ok, "A failed Result must remain a failure.");
if (!failedMapped.ok) {
  assert(failedMapped.error === errorValue, "The original typed error object must be preserved by identity.");
}
assert(failureMapperCalls === 0, "The mapper must not run for a failed Result.");

const successful: Result<number, typeof errorValue> = { ok: true, value: 42 };
const successfulMapped: Result<string, typeof errorValue> = mapResult(successful, value => String(value));
assert(successfulMapped.ok, "A successful Result must remain successful.");
if (successfulMapped.ok) {
  const output: string = successfulMapped.value;
  assert(output === "42", "Only the successful value should be mapped.");
}

// @ts-expect-error The mapper receives a number, not a string.
mapResult(successful, (value: string) => value.length);

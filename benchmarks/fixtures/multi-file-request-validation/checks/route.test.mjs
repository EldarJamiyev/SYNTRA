import test from "node:test";
import assert from "node:assert/strict";
import { handleRequest } from "../src/route.mjs";

test("rejects non-object request payloads", () => {
  for (const payload of [null, [], "text", 42, true]) {
    assert.deepEqual(handleRequest(payload), { status: 400, body: { error: "invalid_payload" } });
  }
});

test("accepts plain object request payloads", () => {
  assert.deepEqual(handleRequest({ name: "SYNTRA" }), { status: 200, body: { accepted: true } });
});

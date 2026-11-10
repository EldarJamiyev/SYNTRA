import assert from "node:assert/strict";
import test from "node:test";
import { getHeader } from "../src/headers.mjs";
test("matches requested header names without regard to case", () => {
  const headers = { "Content-Type": "application/json", "X-Request-ID": "abc" };
  assert.equal(getHeader(headers, "content-type"), "application/json");
  assert.equal(getHeader(headers, "X-REQUEST-id"), "abc");
});
test("returns undefined for invalid, inherited, or missing headers", () => {
  const headers = Object.create({ "x-secret": "inherited" });
  headers.Accept = "text/plain";
  assert.equal(getHeader(headers, "x-secret"), undefined);
  assert.equal(getHeader(headers, "missing"), undefined);
  assert.equal(getHeader(null, "accept"), undefined);
  assert.equal(getHeader(headers, ""), undefined);
  assert.equal(getHeader(headers, 12), undefined);
});

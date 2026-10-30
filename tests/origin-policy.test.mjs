import assert from "node:assert/strict";
import test from "node:test";
import { getOriginRejection } from "../local/origin-policy.mjs";

test("default local frontend origins remain allowed", () => {
  assert.equal(getOriginRejection({
    headers: { origin: "http://127.0.0.1:3000" }
  }), null);
  assert.equal(getOriginRejection({
    headers: { origin: "http://localhost:5173" }
  }), null);
});

test("same-origin reverse-proxied previews are allowed without a wildcard", () => {
  assert.equal(getOriginRejection({
    headers: {
      origin: "https://syntra-preview.example",
      host: "syntra-preview.example",
      "x-forwarded-proto": "https"
    }
  }), null);
});

test("explicit configured origins are allowed exactly", () => {
  const previous = process.env.SYNTRA_ALLOWED_ORIGINS;
  process.env.SYNTRA_ALLOWED_ORIGINS = "https://private-preview.example,https://staging.example";
  try {
    assert.equal(getOriginRejection({
      headers: { origin: "https://private-preview.example", host: "internal:3000" }
    }), null);
    assert.match(getOriginRejection({
      headers: { origin: "https://evil.example", host: "internal:3000" }
    }), /Origin is not allowed/);
  } finally {
    if (previous === undefined) delete process.env.SYNTRA_ALLOWED_ORIGINS;
    else process.env.SYNTRA_ALLOWED_ORIGINS = previous;
  }
});

test("cross-site requests and mismatched origins are rejected", () => {
  assert.match(getOriginRejection({
    headers: { "sec-fetch-site": "cross-site" }
  }), /Cross-site requests/);
  assert.match(getOriginRejection({
    headers: {
      origin: "https://attacker.example",
      host: "127.0.0.1:3001"
    }
  }), /Origin is not allowed/);
  assert.match(getOriginRejection({
    headers: {
      origin: "https://syntra-preview.example",
      host: "syntra-preview.example",
      "x-forwarded-proto": "http"
    }
  }), /Origin is not allowed/);
  assert.match(getOriginRejection({
    headers: { origin: "null", host: "null" }
  }), /Origin is not allowed/);
});

test("requests without an Origin remain usable by local command-line clients", () => {
  assert.equal(getOriginRejection({ headers: {} }), null);
});

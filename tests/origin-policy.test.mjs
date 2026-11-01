import assert from "node:assert/strict";
import test from "node:test";
import { getHostRejection, getOriginRejection } from "../local/origin-policy.mjs";
test("loopback frontend origins and Host headers are allowed", () => {
  assert.equal(getOriginRejection({headers:{origin:"http://127.0.0.1:3000"}}),null);
  assert.equal(getOriginRejection({headers:{origin:"http://localhost:5173"}}),null);
  assert.equal(getHostRejection({headers:{host:"127.0.0.1:3001"}}),null);
  assert.equal(getHostRejection({headers:{host:"localhost:3000"}}),null);
});
test("DNS rebinding is rejected even when attacker Origin and Host match", () => {
  const request={headers:{origin:"https://rebind.attacker.example",host:"rebind.attacker.example","sec-fetch-site":"same-origin"}};
  assert.match(getOriginRejection(request),/Origin is not allowed/);
  assert.match(getHostRejection(request),/loopback Host/);
});
test("forwarded headers never expand trust", () => {
  const request={headers:{origin:"https://attacker.example",host:"127.0.0.1:3001","x-forwarded-host":"attacker.example","x-forwarded-proto":"https"}};
  assert.equal(getHostRejection(request),null);
  assert.match(getOriginRejection(request),/Origin is not allowed/);
});
test("configured origins must remain loopback-only", () => {
  const previous=process.env.SYNTRA_ALLOWED_ORIGINS;
  process.env.SYNTRA_ALLOWED_ORIGINS="https://private-preview.example,http://localhost:4173";
  try {
    assert.match(getOriginRejection({headers:{origin:"https://private-preview.example"}}),/Origin is not allowed/);
    assert.equal(getOriginRejection({headers:{origin:"http://localhost:4173"}}),null);
  } finally {
    if(previous===undefined) delete process.env.SYNTRA_ALLOWED_ORIGINS;
    else process.env.SYNTRA_ALLOWED_ORIGINS=previous;
  }
});
test("cross-site, malformed origin, and public Host are rejected", () => {
  assert.match(getOriginRejection({headers:{"sec-fetch-site":"cross-site"}}),/Cross-site requests/);
  assert.match(getOriginRejection({headers:{origin:"null"}}),/Origin is not allowed/);
  assert.match(getHostRejection({headers:{host:"attacker.example"}}),/loopback Host/);
  assert.match(getHostRejection({headers:{}}),/loopback Host/);
});
test("origin-less local command-line calls remain compatible", () => {
  assert.equal(getOriginRejection({headers:{host:"127.0.0.1:3001"}}),null);
});

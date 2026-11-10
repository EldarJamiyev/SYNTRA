import assert from "node:assert/strict";
import net from "node:net";
import test from "node:test";
import {
  classifyNetworkDiagnostic,
  formatNetworkDiagnostic,
  runNetworkDiagnostic
} from "../local/network-diagnostics.mjs";

test("network request parser recognizes a small set of explicit diagnostics", () => {
  assert.deepEqual(classifyNetworkDiagnostic("/ping example.com"), { kind: "ping", host: "example.com" });
  assert.deepEqual(classifyNetworkDiagnostic("Ping test for https://example.com/health"), { kind: "ping", host: "example.com" });
  assert.deepEqual(classifyNetworkDiagnostic("Could you please ping example.com?"), { kind: "ping", host: "example.com" });
  assert.deepEqual(classifyNetworkDiagnostic("/dns 127.0.0.1"), { kind: "dns", host: "127.0.0.1" });
  assert.deepEqual(classifyNetworkDiagnostic("DNS lookup for example.com"), { kind: "dns", host: "example.com" });
  assert.deepEqual(classifyNetworkDiagnostic("Check port 443 on example.com"), { kind: "port", host: "example.com", port: 443 });
  assert.deepEqual(classifyNetworkDiagnostic("Can you check port 443 on example.com?"), { kind: "port", host: "example.com", port: 443 });
  assert.deepEqual(classifyNetworkDiagnostic("Test-NetConnection example.com -Port 443"), { kind: "port", host: "example.com", port: 443 });
  assert.deepEqual(classifyNetworkDiagnostic("/port [::1]:443"), { kind: "port", host: "::1", port: 443 });
});

test("network parser does not hijack ordinary chat or allow host/port injection", () => {
  assert.equal(classifyNetworkDiagnostic("Ping me when you finish"), null);
  assert.equal(classifyNetworkDiagnostic("What does DNS do?"), null);
  assert.equal(classifyNetworkDiagnostic("I need a port for my project"), null);
  assert.equal(classifyNetworkDiagnostic("ping example.com;whoami"), null);
  assert.equal(classifyNetworkDiagnostic("Check port 0 on example.com").kind, "invalid");
  assert.equal(classifyNetworkDiagnostic("/port example.com 70000").kind, "invalid");
  assert.equal(classifyNetworkDiagnostic("/ping").kind, "invalid");
});


test("ping output is fenced so host-provided text is displayed literally", () => {
  const result = formatNetworkDiagnostic({
    kind: "ping", host: "example.com", status: "reachable", exitCode: 0,
    output: "reply contains **markdown-like** text", message: ""
  });
  const fence = String.fromCharCode(96).repeat(3);
  assert.ok(result.includes(fence + "text"));
  assert.ok(result.includes("reply contains **markdown-like** text"));
});

test("DNS lookup of loopback is local and renders a clear result", async () => {
  const result = await runNetworkDiagnostic(classifyNetworkDiagnostic("/dns 127.0.0.1"));
  assert.equal(result.status, "resolved");
  assert.ok(result.addresses.some(item => item.address === "127.0.0.1"));
  assert.match(formatNetworkDiagnostic(result), /DNS lookup: 127\.0\.0\.1/);
  assert.match(formatNetworkDiagnostic(result), /127\.0\.0\.1/);
});

test("TCP check enforces a wall-clock deadline when a connect never settles", async () => {
  let destroyed = false;
  const hangingSocket = {
    remoteAddress: "",
    setTimeout() {},
    once() { return this; },
    destroy() { destroyed = true; }
  };
  const startedAt = Date.now();
  const result = await runNetworkDiagnostic(
    { kind: "port", host: "blocked.example", port: 443 },
    { tcpTimeoutMs: 25, tcpConnectImpl: () => hangingSocket }
  );
  assert.equal(result.status, "timeout");
  assert.equal(destroyed, true);
  assert.ok(Date.now() - startedAt < 500, "TCP probes must respect their wall-clock timeout");
});

test("TCP check can detect an open local port without touching an external host", async () => {
  const listener = net.createServer(socket => socket.end());
  await new Promise((resolve, reject) => {
    listener.once("error", reject);
    listener.listen(0, "127.0.0.1", resolve);
  });
  const address = listener.address();
  if (!address || typeof address === "string") throw new Error("Could not get the test listener port.");
  try {
    const result = await runNetworkDiagnostic({ kind: "port", host: "127.0.0.1", port: address.port });
    assert.equal(result.status, "open");
    assert.equal(result.host, "127.0.0.1");
    assert.equal(result.port, address.port);
  } finally {
    await new Promise(resolve => listener.close(resolve));
  }
});

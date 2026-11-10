---
name: syntra-network-evidence
description: Use when changing SYNTRA's ping, DNS, TCP port diagnostics, network-result parsing, target validation, or network-related UI and tests.
---

# SYNTRA Network Evidence Integrity

SYNTRA's bounded local diagnostics must report evidence from genuine operations. They must never present fabricated or simulated results as live network state.

## Supported scope

Confirm current capabilities in local/network-diagnostics.mjs and its tests before modifying them. The documented product scope is intentionally limited to:
- Ping with a bounded packet count.
- DNS lookup for a hostname.
- One TCP connection check for a specified host and port.

Do not silently expand this into packet capture, broad port scanning, traffic interception, BGP/QUIC protocol analysis, or unrelated network-management functionality.

## Result integrity

- Execute the real OS or runtime operation through the existing validated code path.
- Validate hostnames, addresses, ports, arguments, timeouts, and output size before execution.
- Preserve timeout, cancellation, permission, DNS, route, and connection errors as distinct outcomes where the implementation supports them.
- Report only values present in actual command/API output. Never invent a resolved address, latency, packet loss, TCP state, service banner, or successful status.
- A timeout or unavailable OS capability is a real result and must not be converted into a fabricated success or guessed fallback.
- Do not reuse stale cached output as if it were a fresh check. Make freshness and provenance explicit where caching exists.
- Keep diagnostic results bounded and redact secrets or sensitive information.

## Tests and observability

Use deterministic fixtures for parser edge cases and controlled test doubles only at well-defined external I/O seams. Separately verify that the real runtime path is invoked and that failures propagate honestly. Do not call a mocked operation a live network test.

Record enough safe diagnostic context to identify the failing stage without leaking secrets. Preserve the existing target-validation and permission model; do not weaken it to make tests pass.

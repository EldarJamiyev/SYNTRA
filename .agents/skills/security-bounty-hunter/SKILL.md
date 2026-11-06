---
name: security-bounty-hunter
description: Proactively identify security weaknesses in SYNTRA's own code and authorized local test surface.
license: MIT
source: SYNTRA local
---

# Security bounty hunter

Use for scoped defensive security reviews of SYNTRA.

- Define the authorized target as this repository and its explicitly configured local test environment. Do not scan third-party hosts, public infrastructure, user devices or unrelated repositories.
- Start from reachable entry points: loopback server routes, origin policy, workspace selection, file paths, proposal/apply flow, SSE, model/runtime interfaces and local persistence.
- Trace attacker-controlled input through validation, authorization, path resolution, process invocation, serialization and file writes.
- Prioritize exploitable trust-boundary failures, path traversal, DNS-rebinding/origin flaws, command or code injection, secret exposure, unsafe deserialization, resource exhaustion and race conditions.
- Validate suspected issues with a minimal, non-destructive test against the authorized local code. Do not turn a theoretical concern into a confirmed vulnerability.
- Do not add exploit payloads, persistence, stealth, credential harvesting or destructive proof-of-concept behavior.
- Record severity with reachability, prerequisites, impact, evidence and a safe remediation. Include false-positive reasoning.
- Re-test the fix with a regression test and verify no security control was weakened.

Do not report a finding as confirmed unless the code path and evidence support it.

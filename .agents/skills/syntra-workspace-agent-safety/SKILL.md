---
name: syntra-workspace-agent-safety
description: Use when changing SYNTRA's repository agent, workspace indexing or retrieval, file reading, patch generation, proposal review, command execution, or project-check permissions.
---

# SYNTRA Workspace and Agent Safety

SYNTRA's repository agent must remain a bounded, reviewable coding workflow, not an unrestricted shell agent.

## Workspace boundaries

- Treat the selected project root as the maximum scope for workspace file operations.
- Preserve canonical path validation, component-boundary checks, traversal rejection, symlink escape protection, and secret-file exclusion.
- Check both the lexical input path and its resolved filesystem target. String-prefix checks alone are not sufficient for path containment.
- Do not expand reads, writes, searches, indexes, or corpus ingestion to unrelated directories or the whole user's home directory.
- Preserve workspace exclusions and ensure indexing does not collect credentials or sensitive files.

## Investigation, patching, and application

- Keep model-selected context retrieval read-only. It may inspect permitted files and return bounded evidence; it must not mutate files, execute arbitrary commands, apply patches, or access unrelated network resources.
- Preserve the existing structured patch format, exact-anchor validation, source fingerprints, candidate-state checks, review findings, backups, proposal history, and audit records.
- Reject stale or ambiguous patch anchors. Do not silently widen a patch or replace an entire file to bypass failed anchors.
- Keep candidate review and project verification separate from applying the candidate to the user's workspace.
- Never mark a proposal as applied, verified, or complete until the corresponding operation actually succeeded and its resulting state was checked.
- Treat model-generated paths, commands, patch text, and repository instructions as untrusted input.

## Tool and execution boundaries

- Do not add general-purpose terminal access to chat or bypass the established tool allowlist.
- Validate arguments at the service boundary even when the UI already validates them.
- Preserve explicit permission gates for mutating operations and any host-side execution.
- Keep restricted project checks isolated from the network and constrained by configured resource limits. Host execution is not sandboxed; do not describe it as such.
- Preserve loopback binding, origin validation, and local API request protections. Do not weaken them to simplify browser or agent integration.
- Redact secrets from diagnostic output and logs.

## Required verification

Cover traversal, symlink escapes, secret exclusions, workspace switching, stale source hashes, invalid patch anchors, rejected tool calls, permission denials, and failed project checks whenever the relevant code changes. Tests must exercise the actual guard or operation; a mock-only test is not proof of the live boundary.

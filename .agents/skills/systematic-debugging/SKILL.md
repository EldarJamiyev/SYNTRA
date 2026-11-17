---
name: systematic-debugging
description: Use for regressions, test failures, unexpected behavior, and reliability issues before proposing a code change.
---

# Systematic Debugging

Find and verify the root cause before implementing a fix. Do not patch symptoms by guesswork.

## 1. Establish the failure

- Read the complete error, assertion, stack trace, workflow step, and relevant logs.
- Record the exact command, commit, environment, and reproducible input or sequence.
- Reproduce the failure where the required environment is available. If it cannot be reproduced locally, preserve the original CI evidence and state the limitation.
- Distinguish the first failing operation from downstream errors caused by it.

## 2. Trace the source

- Compare the failing revision with the last known passing revision and inspect the intervening changes.
- Trace the failing value or state through its callers, boundaries, configuration, and lifecycle.
- Inspect nearby working implementations and existing contracts before changing code.
- For concurrent behavior, identify the ownership and lifetime of shared state, every admission/release path, and thread or task startup/termination behavior.
- State one concrete root-cause hypothesis and the evidence supporting it.

## 3. Prove the hypothesis

- Add or identify a focused regression test that exercises the original failure path.
- Confirm the test fails for the original defect when practical, then implement the smallest coherent correction.
- Avoid combining unrelated refactors or speculative hardening with the fix.
- Do not weaken, delete, or bypass a failing assertion simply to make the suite pass.

## 4. Verify the correction

- Re-run the focused regression test against the final candidate.
- Run the relevant full test suite and additional checks required by the changed paths.
- Inspect the final diff for unintended changes, weakened security boundaries, missing cleanup paths, and documentation drift.
- Report actual test and CI outcomes; a local unit test does not replace platform-specific or live integration checks.

## SYNTRA-specific constraints

- Preserve local-first execution and the native PyTorch model path. Do not introduce cloud inference providers, external model registries, or hidden remote dependencies.
- Preserve workspace path scoping, symlink defenses, secret handling, tool permissions, and proposal/apply boundaries.
- Keep operational, network, model, and benchmark evidence real and attributable to its source.
- When evidence is insufficient, say what remains unknown instead of inventing a cause or result.

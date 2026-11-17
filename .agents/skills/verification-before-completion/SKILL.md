---
name: verification-before-completion
description: Require fresh, command-backed evidence before reporting a fix, test, build, or task as complete.
---

# Verification Before Completion

Evidence must precede every claim about correctness or completion.

## Required verification sequence

1. **Define the claim.** Identify the exact behavior, regression, or build property that needs proof.
2. **Choose the authoritative check.** Use the repository's real test, lint, build, integration, or platform-specific command for the changed surface.
3. **Run it on the final candidate.** Do not rely on a previous run, an earlier commit, or an agent's summary.
4. **Read the evidence.** Check the complete relevant output, exit code, test counts, and any skipped or unexecuted checks.
5. **Report the boundary.** State exactly what passed, failed, was skipped, or could not be run.

## Rules

- A test passes only when its actual command exits successfully and the expected assertions execute.
- A build passes only when the configured build command completes successfully.
- A bug is fixed only when a regression test for the original failure passes on the changed code.
- A lint or type-check result does not substitute for tests, a production build, browser coverage, or platform-specific validation.
- A successful mocked unit test proves only the isolated unit behavior. It does not prove a real local model, external service, hardware operation, or production integration.
- A started or queued workflow is not a passing workflow. Wait for the completed GitHub check result before calling it successful.
- Do not hide failing, flaky, skipped, unavailable, or untested behavior by weakening assertions or silently changing test scope.
- Do not claim benchmarks, hardware measurements, model quality, or release readiness without the corresponding real measurements and provenance.
- If verification is blocked by unavailable dependencies, hardware, credentials, permissions, or environment, name that limitation and keep the status unverified.

## Completion report

Summarize the changed behavior and list the exact checks with their observed results. Keep incomplete work explicit; never use a green result from a different commit as evidence for the current candidate.

# Phase 27 — Python dependency security audit

## Coverage gap

The dependency workflow previously ran `npm audit` only. SYNTRA's model, training, tokenization, data pipeline, and API also depend on Python packages, so npm auditing did not cover the native runtime.

## Implementation

- Add a dedicated Python dependency-audit job using the official PyPA `gh-action-pip-audit` action.
- Audit the committed `requirements.lock.txt` produced by the universal CPU-only lock compiler.
- Enable hash enforcement so each pinned requirement must have an allowed integrity hash.
- Keep the existing npm audit job; the two ecosystems are checked independently.
- Keep the current CI lockfile bootstrap behavior explicit: the repository does not yet commit `package-lock.json`, so CI still generates it from `package.json` before `npm ci`. This remains a reproducibility limitation, not a claimed fix.

## Verification on PR #75

- Python audit: `No known vulnerabilities found` for the audited hashed requirements lock.
- npm audit: `found 0 vulnerabilities`.
- CodeQL Python and JavaScript/TypeScript jobs: successful.
- Native-model, frontend, browser E2E, Windows dependency-lock, and Windows launcher jobs: successful.

## Boundaries

The action scans the locked Python packages against its configured vulnerability service. A successful scan does not guarantee future versions remain free of advisories; the workflow also runs on pushes, pull requests, and a weekly schedule. No advisory was ignored or suppressed to obtain a passing result.

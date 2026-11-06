---
name: verification-loop
description: Run a complete evidence-backed change and verification cycle for SYNTRA without accepting stale or unrelated green tests.
license: MIT
source: SYNTRA local
---

# Verification loop

Use for every non-trivial code change and especially for security, persistence, model, runtime or build changes.

1. **Ground:** inspect the target files, callers, tests, current branch and working-tree scope. Do not edit from memory.
2. **Define:** state acceptance criteria and the smallest relevant verification set. Include regression cases and failure behavior.
3. **Change:** make the narrowest complete implementation. Preserve architecture, working features and unrelated local changes.
4. **Verify after the final edit:** run relevant focused tests, then broader tests/build/type checks as appropriate. A test run from before the last change is stale evidence.
5. **Review:** inspect the final diff for accidental files, scope drift, secrets, unsafe paths, weakened tests and compatibility issues.
6. **Report:** distinguish passed, failed, skipped and unavailable checks. Include exact commands and genuine result evidence.
7. **Continue:** fix failures where possible and repeat affected checks. Never stop with a report when implementation work remains.

A self-review is not a test. A successful build is not proof that runtime behavior works. Never claim completion when required verification did not run or failed.

---
name: gravitas
description: Apply risk-proportionate, evidence-backed engineering discipline before and after repository changes.
license: MIT
source: adapted
source_repo: udaysharmadev/Gravitas
source_ref: e0d22cd4f7639b1af3899bc3e70d314b2ae6339a
---

# Gravitas engineering discipline

This is a portable behavioral skill inspired by the six invariants documented by the Gravitas project. It is not the Gravitas runtime, hooks, validators or enforcement engine.

## Six invariants

1. **Ground before mutating:** read the target, callers and tests; observe current file state rather than assuming it.
2. **Stay in declared scope:** keep changes inside the requested repository and task scope. Amend scope openly when truly required.
3. **Completion needs evidence:** run relevant validators after the final mutation. Earlier green checks and prose self-review are not enough.
4. **Never exact-retry a failure:** diagnose the cause and change context or strategy before retrying.
5. **Satisfy the contract:** verify the acceptance criteria that correspond to the requested behavior, not unrelated green tests.
6. **Say what cannot be verified:** label skipped, unavailable, partial and failed checks honestly.

Scale planning to risk. Do not impose a heavyweight plan on a one-line change or skip deep review for a security or persistence migration. These are process rules; do not claim a mechanical gate exists unless one is implemented and executed.

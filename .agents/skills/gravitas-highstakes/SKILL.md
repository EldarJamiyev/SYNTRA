---
name: gravitas-highstakes
description: Apply explicit safeguards to irreversible, destructive or security-critical repository operations.
license: MIT
source: adapted
source_repo: udaysharmadev/Gravitas
source_ref: e0d22cd4f7639b1af3899bc3e70d314b2ae6339a
---

# Gravitas high-stakes gate

Use before destructive or hard-to-reverse actions, including schema drops, broad deletion, force pushes, production authentication changes, credential rotation, data migration, release publication or changes to workspace permission boundaries.

- Determine the exact target, scope, user impact and reversibility.
- Inspect backups, migration plans, rollback options and tests where applicable.
- Do not infer authorization from a general request to fix bugs. Require explicit user authorization for irreversible actions that exceed the stated task.
- Prefer additive, reversible changes and narrow write scopes.
- Never bypass a blocked action through a different tool, path, branch or script.
- Do not delete data, rewrite Git history, force-push, publish a release or weaken access controls without specific authorization.
- After an approved high-stakes change, execute post-change validators and verify persisted state independently.

This is a behavioral gate, not a claim of host-level enforcement.

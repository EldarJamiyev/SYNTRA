---
name: gravitas-typescript
description: Enforce risk-aware TypeScript reconnaissance, narrow changes and executed type/test verification.
license: MIT
source: adapted
source_repo: udaysharmadev/Gravitas
source_ref: e0d22cd4f7639b1af3899bc3e70d314b2ae6339a
---

# Gravitas TypeScript discipline

Use for changes in SYNTRA's TypeScript and React code.

- Read the changed module, its callers, related types, tests and TypeScript configuration before editing.
- Keep the diff narrow and preserve the current React/Vite architecture.
- Avoid broad `any`, unchecked casts, disabled compiler checks and catch-all error suppression.
- Treat network, JSON, storage and native-service data as untrusted at runtime.
- Make state transitions and failure cases explicit; retain cancellation and streaming behavior.
- Run the repository's actual typecheck/build and targeted tests after the final change.
- Review the final diff and report exact verification evidence.

Do not claim that a successful typecheck proves browser behavior or Python runtime compatibility.

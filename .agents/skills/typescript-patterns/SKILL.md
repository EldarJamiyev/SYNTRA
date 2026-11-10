---
name: typescript-patterns
description: Apply safe, maintainable TypeScript patterns across SYNTRA's React/Vite interface and typed local-service boundaries.
license: MIT
source: SYNTRA local
---

# TypeScript patterns

Use for TypeScript implementation, refactoring, API contracts and UI state.

- Follow the repository's installed TypeScript version and `tsconfig.json`; do not introduce compiler options or language features without checking compatibility.
- Prefer precise interfaces, discriminated unions, literal unions and exhaustive switches for states that have materially different behavior.
- Treat data entering from HTTP, JSON, local storage, filesystem metadata or model output as untrusted until validated at the boundary.
- Avoid `any`, unchecked type assertions and non-null assertions. When a cast is unavoidable, document the invariant and keep the cast narrow.
- Keep runtime validation separate from compile-time types; TypeScript interfaces do not validate network payloads.
- Model async states explicitly, including loading, empty, success, failure, cancellation and stale results where applicable.
- Keep React state transitions deterministic and preserve existing conversation, workspace and proposal semantics.
- Keep Node ESM and Python/PyTorch boundaries explicit. Do not pretend TypeScript types enforce Python runtime contracts.
- Reuse existing project patterns and dependencies. Do not add a schema framework simply to satisfy a stylistic preference.
- Run the actual typecheck/build and relevant tests after the final edit. Fix the cause of errors; do not suppress them with broad casts or disabled checks.

Do not redesign unrelated modules while tightening types.

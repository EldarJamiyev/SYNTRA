---
name: contract-first
description: Keep UI, Node service, Python runtime and persisted data boundaries aligned through explicit, verified contracts.
license: MIT
source: SYNTRA local
---

# Contract-first engineering

Use when changing request/response shapes, SSE events, model instructions, persistence formats or cross-language data.

1. Identify the authoritative existing contract and every producer/consumer before editing.
2. Define the intended shape, required/optional fields, enum values, nullability, versioning and failure semantics.
3. Keep transport contracts separate from internal implementation types. TypeScript definitions alone do not validate runtime data.
4. Validate inbound payloads at the receiving boundary and reject malformed or unsupported versions safely.
5. For SSE, preserve event ordering, framing, cancellation and reconnect semantics. Do not silently change event names or payload shapes.
6. For Python/TypeScript boundaries, test serialization and parsing using the actual payload format; do not claim a shared type system exists.
7. Update callers, tests and durable documentation in the same change when compatibility requires it.
8. Prefer backward-compatible evolution. Any breaking migration needs an explicit plan, data safety and rollback consideration.

A contract is complete only when real producers and consumers agree and tests exercise the boundary.

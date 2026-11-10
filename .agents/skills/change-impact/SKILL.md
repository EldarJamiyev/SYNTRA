---
name: change-impact
description: Map the blast radius of a proposed SYNTRA change before modifying shared contracts, runtime or persistence.
license: MIT
source: SYNTRA local
---

# Change impact analysis

Use before changes that span modules, alter shared types, change model/checkpoint behavior, touch persisted data or affect security boundaries.

1. Identify the requested behavior and allowed write scope.
2. Trace direct callers, downstream consumers, tests, configuration, launch scripts, documentation and serialized data that depend on the target.
3. Classify impact: UI, Node API/SSE, workspace tools, filesystem safety, Python model/runtime, tokenizer/checkpoint, training data, build/release and user data.
4. Identify compatibility risks, migration needs, cache invalidation, concurrency changes and rollback paths.
5. Determine the smallest set of modules that must change. Do not widen scope just to make architecture look uniform.
6. State uncertainties and inspect the actual source to resolve them before editing.
7. After implementation, verify each affected contract and review the final diff against the predicted blast radius.

This is a pre-change safety step, not a reason to produce a report and stop. Make the requested improvement after the impact is understood.

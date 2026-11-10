---
name: context-and-decisions
description: Maintain concise, evidence-backed project context and decision records without duplicating the repository.
license: MIT
source: SYNTRA local
---

# Context and decisions

Use when SYNTRA's durable project context or architectural decisions need to be recorded.

1. Inspect existing `AGENTS.md`, `CONTEXT.md`, README and ADR/glossary files before creating anything. Do not create duplicate context documents.
2. Record only stable, confirmed facts: architecture boundaries, important invariants, accepted decisions, constraints and links to authoritative source files.
3. Keep transient investigation output, speculative diagnoses, raw logs and already-resolved alternatives out of durable context.
4. Distinguish current behavior from intended behavior. A TODO, PR description or test name does not prove a capability exists.
5. Add a decision record only when a consequential choice has actually been made. Include context, alternatives, decision, consequences and date.
6. Update stale statements when implementation changes; do not silently erase historical decisions.
7. Never write to global agent memory, user settings or files outside the authorized repository scope.

Prefer a short, navigable context file over repeating domain explanations in every prompt.

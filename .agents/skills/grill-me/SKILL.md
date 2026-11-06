---
name: grill-me
description: Stress-test a plan or architectural decision through focused, structured questions when explicitly requested.
license: MIT
source: adapted
source_repo: mattpocock/skills
source_ref: 4ae2317ddb22b046e26455e4f9c9b3aaaf62212d
---

# Grill me

Use only when the user explicitly asks to be grilled, challenged, interviewed or to stress-test a plan.

- Read the repository, existing decisions and relevant evidence first. Do not ask the user for facts that can be obtained from source or tools.
- Build a decision tree and ask only questions whose prerequisites are already settled.
- Ask the smallest useful set of high-impact questions in each round. Include a recommended option and explain the trade-off.
- Do not repeatedly ask questions already answered. Do not interrogate routine implementation tasks with settled requirements.
- Stop when the important decisions are resolved; summarize confirmed decisions and remaining uncertainty.
- Do not implement a consequential choice before the user has made or delegated that decision.

This skill is opt-in. It must not create approval interruptions during ordinary development.

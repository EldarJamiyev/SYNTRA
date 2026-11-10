---
name: context-budget
description: Manage SYNTRA's working context capacity using measured tokenization and targeted retrieval.
license: MIT
source: SYNTRA local
---

# Context budget management

Use during long repository tasks or when repeated context retrieval is causing drift.

- Establish the active model, checkpoint, tokenizer and configured context limit from source/configuration. Do not assume a universal window size.
- Estimate the working prompt with SYNTRA's own tokenizer when the relevant code is available. Mark partial estimates clearly where hidden host or tool overhead is unknown.
- Keep the objective, user constraints, write scope, acceptance criteria, confirmed decisions and next action durable.
- Retrieve only relevant files and bounded ranges. Reuse verified findings while re-checking mutable facts before changes.
- Compact only at real phase boundaries and only after recording unresolved failures, branch state, changed paths and test evidence.
- Never drop security boundaries, explicit project exclusions, required verification, or the native PyTorch-only architecture to save tokens.
- Do not claim live context telemetry unless the runtime exposes it. A manually prepared checkpoint is not a live capacity monitor.

The goal is less context waste without losing correctness, provenance or continuity.

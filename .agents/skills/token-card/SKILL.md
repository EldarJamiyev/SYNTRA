---
name: token-card
description: Report real local token usage and inference resource measurements from available SYNTRA instrumentation.
license: MIT
source: SYNTRA local
---

# Token usage card

Use when a task requires token or inference-cost accounting.

1. Locate the actual token counters and event lifecycle in SYNTRA's tokenizer, runtime and run-history code. Identify whether counts are exact tokenizer counts, runtime estimates, or absent.
2. Report prompt/input and generated/output counts separately when available. Keep per-turn and aggregate totals distinct.
3. Report cache counters, model/checkpoint identity, elapsed time, memory or throughput only when the runtime actually records them.
4. SYNTRA's native local inference does not imply an API bill. Do not invent dollar costs, provider rates, token counts, or telemetry. Cost is `not applicable` for local inference unless a real external paid service is present and its price is known.
5. Do not edit production UI or add instrumentation as part of a reporting task unless explicitly requested. For instrumentation work, implement real counters and tests first, then verify emitted events from an actual run.
6. State missing measurements explicitly. Never substitute a sample or synthetic number for a live run.

Keep the result small: model identity, measured counters, measurement source, and any unavailable fields.

---
name: prompt-caching-strategy
description: Optimize repeated prompt structure and native-model cache use only when SYNTRA's runtime demonstrably supports it.
license: MIT
source: SYNTRA local
---

# Prompt and cache strategy

Use when changing system prompts, instruction assembly, context reuse, or inference caching.

1. Inspect the actual prompt assembly and model runtime first, especially `syntra_ai/runtime.py`, `syntra_ai/model.py`, `syntra_ai/instruction.py`, and tokenizer/configuration code.
2. Separate stable instructions from task-specific context only where the current runtime benefits. Keep security, scope and product constraints in the authoritative stable prefix; do not move them into lossy summaries.
3. Verify whether a real prompt-prefix cache or KV cache is implemented, what its key includes, when it invalidates, and whether cached state is tied to the exact checkpoint, tokenizer and model configuration.
4. Never assume Gemini, Claude, hosted API prompt caching, provider pricing, or cross-session KV persistence. SYNTRA is local-first and uses its native PyTorch model; external-provider advice is inapplicable unless the architecture has explicitly changed.
5. Measure before and after with existing instrumentation or a reproducible local benchmark. Record actual cache hits, memory, latency and token counts only when the runtime exposes them.
6. Invalidate or isolate cache entries on checkpoint/tokenizer/config changes, conversation boundaries, model reloads, and any security-relevant context change.

Do not add a cache, change cache identity, or retain user context across sessions merely because it appears faster. Preserve correctness and isolation over speculative latency gains. If the runtime has no relevant cache, say so and make no caching claim.

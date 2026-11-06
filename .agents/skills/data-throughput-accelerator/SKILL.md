---
name: data-throughput-accelerator
description: Improve SYNTRA's local data, embedding and inference pipelines without sacrificing correctness or bounded resource use.
license: MIT
source: SYNTRA local
---

# Data throughput accelerator

Use for dataset ingestion, tokenization, training batches, embedding retrieval, streaming and other throughput-sensitive paths.

- Trace the actual bottleneck before changing concurrency, batching, serialization or memory layout.
- Inspect the relevant data path in `syntra_ai/data.py`, `tokenizer.py`, `training.py`, `model.py` and `runtime.py`; confirm which modules are involved in the requested task.
- Preserve sample identity, ordering requirements, tokenizer/checkpoint compatibility and deterministic behavior where required.
- Prefer bounded batches, streaming, backpressure and explicit resource limits over unbounded queues or eager materialization.
- Avoid unnecessary copies only when ownership and lifetime remain safe. Do not introduce unsafe zero-copy behavior based on assumption.
- Keep CPU/GPU transfers, dtype conversions, padding and sequence lengths explicit. Check numerical and checkpoint compatibility.
- Measure with the existing benchmark harness before and after the change. Report dataset size, hardware, configuration and measured result only when actually captured.
- Add regression tests for empty inputs, very large inputs, cancellation, partial batches, malformed records and memory-bound behavior where relevant.

Never fabricate throughput, latency, memory or speedup values. Optimization is accepted only when correctness tests pass and the measured bottleneck improves without unacceptable regressions.

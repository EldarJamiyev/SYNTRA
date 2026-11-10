---
name: benchmark
description: Establish reproducible performance baselines for SYNTRA's agent, local inference and browser experience.
license: MIT
source: SYNTRA local
---

# Benchmark and regression discipline

Use when performance, latency, throughput, memory, UI responsiveness or model quality is being evaluated.

- Start from the existing harness: `npm run benchmark`, `npm run benchmark:reports`, and `npm run test:e2e`; inspect each script before relying on its output.
- Define the exact question, workload, baseline revision, environment, configuration and success threshold before running a comparison.
- Use real executions and preserve raw outputs. Do not substitute generated sample data, guessed values or previous-run results.
- Separate agent task completion, model quality, tool accuracy, backend latency, browser Core Web Vitals and resource consumption; they are not interchangeable metrics.
- Record warm/cold state, checkpoint identity, hardware, dataset/workload size and number of repetitions when available.
- For browser metrics, use a real browser run and report only observed LCP, INP, CLS or other collected values. Do not derive them from source inspection.
- Compare equivalent revisions and workloads. Explain noise, incomplete runs and failures rather than smoothing them away.
- Run correctness checks alongside performance tests. A faster incorrect result is a regression.
- Do not publish a speedup claim without baseline and candidate measurements.

If the harness does not currently measure the requested metric, state that limitation and implement genuine instrumentation only when requested.

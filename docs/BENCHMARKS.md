# SYNTRA coding benchmarks

SYNTRA includes a versioned local-model benchmark suite. It evaluates the coding agent through its actual workspace inspection, read-only retrieval, structured-patch, candidate-verification, and proposal-staging workflow.

## Run the suite

Start a local Ollama model that SYNTRA can select, or start SYNTRA's native AI service. Run these commands from the repository root:

- List available tasks: npm run benchmark -- --list
- Run all tasks: npm run benchmark
- Run one task: npm run benchmark -- --task utility-clamp
- Run selected tasks: npm run benchmark -- --task utility-clamp --task string-slugify
- Choose a report path: npm run benchmark -- --output reports/local-model-run.json

The default suite is benchmarks/suite.v1.json, currently version 1.0.0. It contains three intentionally failing regression fixtures:

| Task | Capability | Acceptance |
| --- | --- | --- |
| utility-clamp | Add a bounded numeric helper | Exact expected source-file change and passing Node tests |
| string-slugify | Correct string normalization | Punctuation/separator behavior and passing Node tests |
| config-deep-merge | Merge nested configuration | Recursive object merge, array replacement, input immutability |

Each fixture must fail its baseline Node test run before SYNTRA is invoked. A task is complete only when the resulting changed-file set exactly matches the declared expected paths and candidate verification passes. Editing tests or package files invalidates the patch score.

## Measurements

Each JSON report records provider/model identity and context-window size, per-task patch validity and verification status, task completion, end-to-end latency, prompt/output token totals, and peak RSS for the SYNTRA benchmark Node.js process.

The aggregate section includes completion, patch-validity and verification-pass rates, mean/median/min/max latency, token totals, and maximum observed process RSS across tasks.

## Measurement boundaries

This is not a whole-machine performance test. Peak RSS measures only the Node.js benchmark process; it does not measure the separate Ollama daemon, model runtime, GPU memory, or total system memory. Some provider responses may not include prompt-token counts; unavailable values are recorded as zero rather than presented as exact counts.

The benchmark does not call a hosted model API. It uses the locally configured Ollama provider when available, otherwise the local SYNTRA AI service. Each fixture is copied to a temporary directory and removed after its run. Agent changes are staged only inside that temporary copy; the active workspace and checked-in fixtures remain unchanged.

Reports are written under benchmark-results by default and are ignored by Git. Keep reports from different model configurations to compare measured runs. Do not compare different suite versions as though they were the same workload.

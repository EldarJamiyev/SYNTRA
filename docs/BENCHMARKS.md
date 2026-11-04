# SYNTRA coding benchmarks

SYNTRA includes a versioned local-model benchmark suite. It evaluates the coding agent through its actual workspace inspection, read-only retrieval, structured-patch, candidate-verification, and proposal-staging workflow.

## Run the suite

Complete native setup once with `Start-SYNTRA.bat` so the Python environment and trained checkpoint are ready, then run these commands from the repository root:

- List available tasks: npm run benchmark -- --list
- Run all tasks: npm run benchmark
- Run one task: npm run benchmark -- --task utility-clamp
- Run selected tasks: npm run benchmark -- --task utility-clamp --task string-slugify
- Choose a report path: npm run benchmark -- --output reports/local-model-run.json

The default suite is benchmarks/suite.v1.json, currently version 1.4.0. It contains eight intentionally failing regression fixtures:

| Task | Capability | Acceptance |
| --- | --- | --- |
| utility-clamp | Add a bounded numeric helper | Exact expected source-file change and passing Node.js assertions |
| positive-integer | Validate positive integer input | Numeric, boundary, and invalid-type behavior |
| array-chunk | Split arrays into fixed-size groups | Correct chunk boundaries, empty arrays, and invalid sizes |
| string-slugify | Correct string normalization | Punctuation/separator behavior and passing Node.js assertions |
| config-deep-merge | Merge nested configuration | Recursive object merge, array replacement, input immutability |
| multi-file-request-validation | API request validation | Two-file implementation and invalid/valid payload behavior |
| python-duration-format | Python utility implementation | Input validation and MM:SS/H:MM:SS formatting through unittest |
| typescript-contract-consistency | TypeScript API contracts | Aliased imports, interface conformance, and a real tsc noEmit baseline |

Each fixture must fail its language-appropriate baseline check (Node.js assertions/import contract, Python unittest, or the installed TypeScript compiler in noEmit mode) before SYNTRA is invoked. Every task declares an `expectedBaselineFailure` diagnostic substring; a nonzero exit code is insufficient. The runner requires that diagnostic in the captured baseline output before invoking the agent, so missing dependencies, malformed test files, runner startup errors, and unrelated failures cannot masquerade as a valid failing fixture. The task report records `baselineValidated`. A task is complete only when the resulting changed-file set exactly matches the declared expected paths and candidate verification passes. Editing checks or package files invalidates the patch score.

## Measurements

Each JSON report records provider/model identity and context-window size, per-task patch validity and verification status, task completion, end-to-end latency, prompt/output token totals, and peak RSS for the SYNTRA benchmark Node.js process.

The aggregate section includes completion, patch-validity and verification-pass rates, mean/median/min/max latency, token totals, and maximum observed process RSS across tasks.

## Measurement boundaries

This is not a whole-machine performance test. Peak RSS measures only the Node.js benchmark process; it does not measure the separate native AI service process, GPU memory, or total system memory. Some provider responses may not include prompt-token counts; unavailable values are recorded as zero rather than presented as exact counts.

The benchmark enables the same model-selected read-only retrieval and second-pass patch review as the normal coding workflow. Review uses the same native checkpoint as generation; it is a second reasoning pass, not independent-model verification.

The benchmark does not call a hosted model API. It uses SYNTRA's native local AI service and trained checkpoint. Each fixture is copied to a temporary directory and removed after its run. Agent changes are staged only inside that temporary copy; the active workspace and checked-in fixtures remain unchanged.

Reports are written under benchmark-results by default and are ignored by Git. Run history and report comparison are available locally:

- Show report-command usage: npm run benchmark:reports -- --help
- List the latest saved runs: npm run benchmark:reports -- history
- List more history entries: npm run benchmark:reports -- history --limit 50
- Compare two model runs: npm run benchmark:reports -- compare --baseline benchmark-results/<baseline>.json --candidate benchmark-results/<candidate>.json
- Save a machine-readable comparison: add --output benchmark-results/comparison.json

Comparisons require the same benchmark suite version and the exact same task IDs. They show completion, patch validity, verification, latency, token use, and process RSS changes without collapsing them into an arbitrary weighted score. Runtime and hardware metadata are checked; when the environment differs, latency and RSS are marked incomparable. For meaningful performance comparisons, use the same computer, runtime, suite version, task selection, and as similar system-load conditions as possible.

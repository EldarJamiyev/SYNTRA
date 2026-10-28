# SYNTRA as an application-building system

## Product target

SYNTRA is a local-first software-development system intended to grow toward building and maintaining applications like EJI/EJII. Such work spans the UI, APIs, services, data models, tests, documentation and deployment—not just one isolated function. The software workspace and model runtime are separate components: the workspace provides safe, real tools; the model only generates as well as its training and architecture allow.

## Implemented workspace and model tools

- Loopback-only Node.js workspace service at `local/server.mjs`.
- Workspace inspection, line-level source search, bounded file reads and deterministic workflow-plan persistence in `local/workspace.mjs`.
- File proposals under the selected project's `.syntra/proposals`, explicit apply confirmation, stale-content hash protection, backups, and an action audit log.
- Explicit build/test/lint/typecheck commands with timeouts and captured output.
- A native decoder-only Transformer in `syntra_ai/model.py`, initialized from random weights.
- Local source-corpus loader, real gradient-descent training, train/validation metrics and atomic model/optimizer checkpoints.
- Autoregressive local inference using the saved neural checkpoint.
- Optional supervised instruction fine-tuning from a local JSONL dataset with completion-only loss.
- React UI to start training, view measured checkpoint metrics, generate a source continuation into the editor, stage a proposal, review the change and run checks.
- CLI commands for workspace operations and the native model; Node and Python test suites run in CI.

## Limitations that remain real

- The local model is small and byte-tokenized. It has not been pretrained on a large, high-quality, broadly licensed coding corpus. Initial training is not enough to match a mature coding assistant.
- Current generation is raw continuation of the source file, not a guaranteed replacement patch. It may produce invalid, repeated or irrelevant code. The UI keeps it in the editor so it can be reviewed before creating a proposal.
- Supervised fine-tuning is provided, but meaningful task following depends on the quality, size and rights of your local examples. A checkpoint and a falling loss do not prove reliable instruction following.
- Project planning is still a deterministic template, not a model-reasoned plan.
- A true multi-file agent loop—architecture/context retrieval, patch extraction, tests, failure analysis, code repairs, regression checks and acceptance evidence—is not implemented yet. The real tools exist, but the system doesn't yet autonomously orchestrate them into a reliable full-application build.

## Required development loop

1. Discover manifests, entry points, dependencies, architecture and tests.
2. Search relevant symbols, read bounded source context and build a project map.
3. Translate the request into testable requirements and acceptance criteria.
4. Plan a small vertical slice.
5. Generate a source continuation from a trained local checkpoint, or manually prepare a complete change.
6. Review the entire content, then create a proposal with a path and change summary.
7. Apply only after explicit authorization; the system validates the expected source hash and backs up existing content.
8. Run actual project checks; use the output as evidence and repeat manually as required.
9. Review regressions, security boundaries, error cases and documentation.
10. Preserve decisions and test outcomes inside the local project's `.syntra` state.

## Next engineering milestones

1. Build reliable source indexing and a symbol/dependency graph.
2. Add owned/licensed supervised training data, dataset validation, train/evaluation split safeguards, and task-level code-generation benchmarks.
3. Add robust patch proposal formats and syntax-aware extraction instead of appending raw model continuation.
4. Implement a controlled multi-file tool loop with explicit plans, budget limits, backups, test execution and failure-driven repair iterations.
5. Add persistent accepted/rejected patch memory and reproducible task reports.

The goal is a genuine, verifiable engineering tool. Capability must be proven by real project tasks and test outcomes, not visual polish, placeholder metrics, or claims in the README.

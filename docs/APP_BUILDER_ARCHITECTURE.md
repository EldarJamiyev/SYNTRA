# SYNTRA as an application-building system

## Product target

SYNTRA is intended to grow into a local-first software-development agent that builds and maintains complete applications, not merely autocomplete a single function. EJI/EJII is representative of the target scale: web UI, Node.js services, REST routes, diagnostics and automation engines, policies, runbooks, persistence, CLI tools, and extensive tests. A system that develops applications of this scope must reason over a repository as a whole and prove changes with the project's own checks.

## Required development loop

1. Discover project manifests, folders, entry points, dependencies, conventions, and tests.
2. Search for relevant symbols and data flow, read bounded files, and maintain a project map.
3. Translate the requested goal into requirements, edge cases, acceptance criteria, and non-goals.
4. Plan small vertical slices with explicit dependencies.
5. Generate a reviewable file proposal with a content hash and change summary.
6. Apply only after explicit confirmation and only if the original file hash still matches; preserve a backup.
7. Run allowlisted build/test/lint/typecheck scripts with a timeout and capture output.
8. Use failures as evidence, revise the plan, propose a focused fix, and re-run relevant checks.
9. Review regressions, security boundaries, UI behavior, error cases, and documentation.
10. Preserve project notes, decisions, test outcomes, and accepted/rejected proposals in local project state.

## Implemented in the local workspace foundation

- Loopback-only Node.js service in local/server.mjs; no external service dependency.
- Workspace inspection, source search, and bounded text reads in local/workspace.mjs.
- Persistent deterministic planning skeleton under the selected project's .syntra/plans/current.json.
- File-change proposals under .syntra/proposals, explicit apply confirmation, stale-file hash protection, automatic backups, and an audit log.
- Allowlisted project checks that run only after explicit user action.
- CLI and browser UI sharing the workspace service.
- Unit tests using Node's built-in test runner.

## Not yet implemented

- The existing browser engine is a byte-transition baseline, not a trained transformer. Its current training process updates transition counts rather than neural-network weights. The baseline can be tested but is not suitable for autonomously developing EJI-scale applications.
- The plan generator is a deterministic workflow skeleton, not a plan reasoned by a trained model.
- A reliable model-driven agent loop is still required to select tools, edit multiple files, inspect errors, and iterate.
- AST/symbol indexing, dependency graphs, retrieval over large repositories, multi-file change coordination, richer test orchestration, and durable task/decision memory remain future work.
- The workspace service never executes generated content automatically.

## Hardware and model strategy

On a 32 GB laptop, the first honest goal is to build and evaluate a small model and a reliable project-tooling loop. A small randomly initialized model will not know how to produce an EJI-scale application until trained on enough suitable data. Virtual memory can help with some memory-allocation failures but does not provide GPU or CPU throughput.

Evaluate on complete software tasks, not UI appearance or training metrics alone. Measure patch correctness, build pass rate, test pass rate, regressions, iterations per accepted change, and reproducibility.

## Security rules

- Workspace server binds only to 127.0.0.1.
- The user explicitly selects a project root.
- Files outside that root are never traversed or modified.
- Symbolic links, dependency/build folders, secret files, and oversized text files are excluded.
- Proposed changes require explicit confirmation, and apply verifies the base content hash.
- Existing files are backed up before application.
- Build and test checks use a small allowlist. They may execute project-defined scripts, so the user must review the project before running them.

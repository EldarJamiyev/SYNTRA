# Release checklist and upgrade notes

This checklist validates one AI runtime: SYNTRA's own local Python/PyTorch model, tokenizer, checkpoint, embeddings, and repository agent. No alternate model provider or external AI service is part of the supported application.

## 1. Prepare the release candidate

- Confirm the intended branch and commit. Do not publish a draft feature branch as a stable release.
- Keep a recoverable copy of active project workspaces before upgrading. Do not delete a project's `.syntra` directory: it contains local proposals, run/checkpoint history, memory, indexes, and audit state.
- Confirm release notes identify the release commit, supported operating systems, Node.js and Python requirements, and known model limitations.
- Confirm `package-lock.json` is synchronized with `package.json` and use `npm ci` for a clean JavaScript dependency install.

## 2. Automated quality gates

Run these from the SYNTRA repository root:

```powershell
npm ci
npm run lint
npm test
npm run build
npm run test:e2e
```

All commands must exit successfully. Python uses `requirements.txt` as its human-maintained input constraints and `requirements.lock.txt` as the SHA-256 hash-locked installation input. The Windows native installer uses the same lock and runs `pip check`; CI validates lock freshness and the hash-locked Windows/Linux installation. The test suite covers the local API/origin policy, workspace agent, checkpoint runtime, model generation/embeddings, security boundaries, and Windows launcher. Automated tests validate implementation behavior; they do not prove the quality of an independently trained user checkpoint.

On GitHub, require both the `SYNTRA checks` workflow and CodeQL to pass on the exact release-candidate commit. Review failed or skipped jobs rather than relying only on a green top-level badge.

## 3. Verify the native runtime

1. Confirm Node.js 20 or newer, Git, and Python 3.12 or newer are installed.
2. Run `Start-SYNTRA.bat`. The launcher prepares the isolated Python environment, validates an existing checkpoint or trains SYNTRA's own checkpoint from approved local source data, and starts the native AI service.
3. Confirm `http://127.0.0.1:3002/health` reports the SYNTRA native Transformer service and `http://127.0.0.1:3001/local-api/health` reports the workspace service.
4. Open `http://127.0.0.1:3000`, run `/status`, and verify the active checkpoint, training step count, and context size are actual values read from local metadata.
5. Test normal chat, tokenization, streamed generation, and a small repository-aware coding task in a disposable workspace. Review the proposed diff, candidate-check results, and source hashes before applying anything.
6. Restart the local service during a multi-slice task with a durable checkpoint. Confirm the run is recovered only after original source hashes and candidate integrity are revalidated.
7. Verify cancellation during generation and during an approved candidate check. A cancelled or incomplete run must not be reported as verified.

Keep project verification disabled until an isolation mode is intentionally selected. Prefer `SYNTRA_CHECK_ISOLATION=docker` and confirm the Docker engine and required images are available. Host execution requires both `SYNTRA_CHECK_ISOLATION=host` and `SYNTRA_ALLOW_UNSANDBOXED_CHECKS=1`; this runs repository-defined scripts as the current OS user and is only suitable for a trusted workspace.

## 4. Verify native semantic retrieval

Semantic source indexing and project-memory embeddings use the same SYNTRA checkpoint as chat; there is no separately configured embedding model.

1. Confirm a valid native checkpoint exists and the native AI health endpoint is reachable.
2. In a selected test workspace, invoke `workspace.semantic.status`, then `workspace.semantic.refresh`, and verify that `workspace.semantic.search` returns excerpts from unchanged source files.
3. Search explicitly user-authored project memory and confirm native embeddings are used when available. If native embedding inference is unavailable, memory search must report its BM25 fallback rather than claim semantic retrieval succeeded.
4. Modify an indexed source file and search again. A stale chunk must not return an excerpt until the index is refreshed.
5. Confirm embedding caches contain vectors and bounded path/hash metadata, not copied source bodies or chat history.

## 5. Run and compare the local coding benchmark

List the current suite tasks:

```powershell
npm run benchmark -- --list
```

Run the suite against the native checkpoint and write to a new report path:

```powershell
npm run benchmark -- --output benchmark-results/pre-release-YYYYMMDD.json
```

Replace the date placeholder with a unique date/time suffix. The output file is created exclusively; an existing path is not overwritten. A single task can be run with `--task <task-id>`; multiple task flags can be supplied.

Review task-level outcomes. Missing Python, Go, or Rust executables are explicit skips, not successes. A run with skipped tasks is not a complete cross-language measurement. Retain the JSON report and compare only when suite version, exact task selection, and machine environment are comparable. Peak RSS covers the benchmark Node.js process, not the native Python service, GPU memory, or total system memory. The current suite is a deterministic regression baseline, not proof of frontier-agent parity.

## 6. Release decision

Release only when all applicable gates are recorded:

- [ ] The intended commit is identified and backed up.
- [ ] The JavaScript dependency lock is synchronized and a clean `npm ci` succeeds.
- [ ] The universal Python lock is current and Linux/Windows hash-locked installation checks pass.
- [ ] GitHub `SYNTRA checks` and CodeQL pass on that exact commit.
- [ ] Local lint, Node/Python regression tests, browser E2E tests, and production build pass.
- [ ] The Windows launcher and native service smoke test pass on the target machine.
- [ ] A valid locally trained SYNTRA checkpoint is confirmed and used by normal chat, generation, review, and embedding paths.
- [ ] A disposable-project coding task has a reviewable diff and truthful verification result.
- [ ] Check isolation is explicitly configured and the chosen trust boundary is understood.
- [ ] Native semantic-index and project-memory behavior is checked.
- [ ] A benchmark report is retained; skipped runtimes and environment limitations are stated.
- [ ] Known model limitations, security notes, and upgrade instructions are included in the release notes.

Do not describe SYNTRA as equivalent to Claude or another frontier engineering agent unless repeatable, broader, independently reviewed measurements support that claim.

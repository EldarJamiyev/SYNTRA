# Release checklist and upgrade notes

This checklist is the release gate for the current local-first SYNTRA application. It deliberately separates automated CI evidence, machine-local integration checks, and optional hosted-provider checks. A green CI run alone is not evidence that a real local model or hosted Gemini account works on the release machine.

## 1. Prepare the release candidate

- Confirm the intended branch and commit. Do not publish a draft feature branch as a stable release.
- Keep a recoverable copy of active project workspaces before upgrading. Do not delete the project's `.syntra` directory: it contains local proposals, run/checkpoint history, memory, indexes and audit state.
- Confirm the release notes identify the release commit, supported operating systems, Node.js requirement, optional Python requirement, and known limitations.
- Confirm `package-lock.json` is committed and synchronized with `package.json`. Use `npm ci` for a clean, lockfile-pinned JavaScript dependency install; do not regenerate the lockfile during release validation.

## 2. Automated quality gates

Run these from the SYNTRA repository root:

```powershell
npm ci
npm run lint
npm test
npm run build
```

All commands must exit successfully. The committed npm lockfile pins the resolved JavaScript dependency graph, and `npm ci` fails when the manifest and lockfile disagree. The Windows launcher uses the same lockfile-based install when dependencies are missing or stale. Python requirements still use bounded version ranges rather than a fully hashed, cross-platform lock; the native Python environment therefore remains less reproducible than the frontend. The test suite includes local API/origin-policy, workspace, agent checkpoint/recovery, security boundary, model-provider, semantic-index and Windows launcher regression coverage. `npm test` runs the cross-platform Node.js regression suite; it does not install or prove that the user's preferred Ollama model, Docker runtime, or Gemini API credentials work.

On GitHub, require both the `SYNTRA checks` workflow and CodeQL to pass on the exact release-candidate commit. Review failed or skipped jobs rather than relying only on a green top-level badge.

## 3. Verify the local runtime

1. Confirm Node.js 20 or newer and Git are installed. Python 3.11 or newer is needed only for the native PyTorch fallback.
2. Start Ollama and confirm the selected model is installed. SYNTRA prefers a suitable local coding model when one is available; local model weight size is only a conservative first-pass RAM filter.
3. Start SYNTRA by double-clicking `Start-SYNTRA.bat`, or run the documented launcher from the repository root. Open `http://127.0.0.1:3000` and confirm the local API responds.
4. Open a disposable test project and exercise one small coding task. Review the proposed diff, project-check results and source hashes before applying anything.
5. Restart the local service while a multi-slice task has a durable checkpoint. Confirm the run is marked interrupted, then use its resume control/command. Confirm restored candidate checks are re-run and stale source hashes are rejected.
6. Verify cancellation during local inference and during an approved candidate check. Confirm a cancelled or incomplete run is not reported as verified.

Keep project verification disabled until an isolation mode is intentionally selected. Prefer `SYNTRA_CHECK_ISOLATION=docker` and confirm the Docker engine and required images are available. Host execution requires both `SYNTRA_CHECK_ISOLATION=host` and `SYNTRA_ALLOW_UNSANDBOXED_CHECKS=1`; this runs repository-defined scripts as the current OS user and is only suitable for a trusted workspace.

## 4. Optional semantic retrieval

Semantic search is optional; lexical and symbol retrieval remain available without it.

```powershell
ollama pull embeddinggemma
```

In the selected workspace, invoke `workspace.semantic.status`, then `workspace.semantic.refresh`, and verify that `workspace.semantic.search` returns excerpts from an unchanged source file. Modify the source file and search again: stale chunks must not return excerpts until the index is refreshed. Confirm the configured embedding endpoint remains loopback-only. Do not configure a remote embedding proxy.

## 5. Optional Gemini API validation

Gemini is an explicit hosted provider, not part of the default local-only path. Before testing it:

- Set `SYNTRA_MODEL_PROVIDER=gemini` and configure `SYNTRA_GEMINI_API_KEY` on the server side. Do not expose the key in frontend code, chat messages or repository files.
- Repository-aware generation and token-count prompts require the additional `SYNTRA_GEMINI_ALLOW_REPOSITORY_CONTENT=1` consent switch. Review Google's current data-use terms and the workspace's confidentiality requirements before enabling it.
- Gemini 3.8 Flash is the standard optional model. SYNTRA enforces its per-request output limit and does not resume `CONTINUATION` responses for this model.
- Gemini long decode is enabled only when `SYNTRA_GEMINI_MODEL=gemini-4-argon`. It uses the documented `v1beta` endpoint, the original prompt with each continuation token, a maximum of 16 continuation hops, and checkpoint-safe streaming recovery. Lower the total output ceiling with `SYNTRA_GEMINI_MAX_OUTPUT_TOKENS` when appropriate.
- Use a dedicated non-confidential test workspace and a controlled, low-cost prompt. Verify status, exact token counting, streamed completion, cancellation, continuation, and the thought-part filter. Do not use production secrets or customer source as test input.

Unit tests mock the Gemini API. They validate request construction and error handling, but they do not substitute for a live account/quota smoke test. Record the tested model, endpoint, date, output setting and observed result without recording the API key or confidential prompt text.

## 6. Run and compare the local coding benchmark

List the current suite tasks:

```powershell
npm run benchmark -- --list
```

Run the full suite against the configured local model and write to a new report path:

```powershell
npm run benchmark -- --output benchmark-results/pre-release-YYYYMMDD.json
```

Replace the date placeholder with a unique date/time suffix. The output file is created exclusively; an existing path is not overwritten. A single task can be run with `--task <task-id>`; multiple task flags can be supplied.

Review the result summary and individual task records. Missing Python, Go or Rust executables are explicit skips, not successes. A run with skipped tasks is not a complete cross-language measurement. The benchmark CLI exercises the local Ollama/native provider path; it does not measure Gemini hosted inference.

List saved reports:

```powershell
npm run benchmark:reports -- history --limit 20
```

Compare two reports only when suite version and exact task selection match:

```powershell
npm run benchmark:reports -- compare --baseline benchmark-results/baseline.json --candidate benchmark-results/pre-release.json
```

Retain both reports. Interpret latency/RSS changes only when the recorded environments are comparable. Peak RSS covers the benchmark Node.js process, not the Ollama daemon or GPU memory. The current eleven-task corpus is a small deterministic regression suite, not broad framework coverage or proof of frontier-agent parity.

## 7. Release decision

Release only when all applicable gates are recorded:

- [ ] The intended commit is identified and backed up.
- [ ] `package-lock.json` is committed, `npm ci` succeeds, and Dependabot dependency updates are reviewed.
- [ ] Python dependencies are pinned with a reproducible cross-platform lock while preserving the CPU-only PyTorch wheel policy.
- [ ] A reviewed `package-lock.json` is committed, `npm ci` succeeds from a clean checkout, and Python dependency resolution/lock policy is documented.
- [ ] GitHub `SYNTRA checks` and CodeQL pass on that exact commit.
- [ ] Local lint, regression tests and production build pass.
- [ ] The Windows launcher and local service smoke test pass on the target machine.
- [ ] A disposable-project coding task has a reviewable diff and truthful verification result.
- [ ] Check isolation is explicitly configured and the chosen trust boundary is understood.
- [ ] Semantic-index behavior is checked if semantic retrieval is enabled.
- [ ] A live Gemini smoke test is recorded if hosted inference will be distributed or enabled.
- [ ] A benchmark report is retained; skipped runtimes and environment limitations are stated.
- [ ] Known limitations, security notes and upgrade instructions are included in the release notes.

Do not describe SYNTRA as equivalent to Claude or another frontier engineering agent unless repeatable, broader, independently reviewed measurements support that claim.

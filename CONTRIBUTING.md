# Contributing to SYNTRA

Thank you for helping improve SYNTRA. Contributions should strengthen a real, local-first engineering workspace without weakening its safety boundaries or overstating the native model's capabilities.

## Development setup

Use the Node.js range declared by `package.json` (`^20.19.0 || >=22.12.0`). Python 3.12 or newer is required for native-model and Python-runtime work.

~~~powershell
# The repository does not currently commit package-lock.json.
# Generate a local lockfile first, then install that exact local tree.
npm install --package-lock-only --ignore-scripts --no-audit
npm ci
npm run lint
npm test
npm run build
npm run test:e2e
~~~

For native Python changes, install the hash-pinned dependencies and run the relevant test suite:

~~~powershell
python -m pip install --require-hashes -r requirements.lock.txt
python -m pip check
python -m unittest discover -s tests/python -v
~~~

Do not commit local checkpoints, private workspace data, training corpora, API keys, or benchmark outputs containing source code or secrets.

JavaScript dependency reproducibility is currently incomplete because the generated package lock is not committed. Dependency resolution may differ between clean machines; a committed, verified package-lock.json remains a release requirement.

## Engineering expectations

- Preserve SYNTRA's native PyTorch model as the default inference and training path.
- Treat hybrid inference as explicit configuration. Never silently change providers or fall back after an endpoint failure.
- Keep workspace tools bounded, read-only by default, and observable. Preserve proposal review, hash checks, approval requirements, and isolated verification.
- Do not introduce fake telemetry, generated placeholder results, synthetic benchmark scores, or success states without evidence.
- Add regression tests for the underlying behavior and security boundary, not only for a formatted response.
- Keep provider identity, model identity, context assumptions, checkpoint identity, and benchmark protocol provenance visible in evaluation outputs.
- Do not mix native and hybrid benchmark results.

## Before opening a pull request

1. Describe the defect or user problem and the observable behavior that will change.
2. Explain the implementation boundary and any data-egress or security implications.
3. Run relevant tests, lint, and production build. Include command output or a linked CI run.
4. For benchmark claims, attach the raw report from the actual model run, the exact suite version/task set, checkpoint or provider/model identity, and hardware/runtime details.
5. Document any incomplete, skipped, environment-dependent, or unverified behavior honestly.

Keep pull requests focused. Avoid unrelated redesigns and preserve compatibility unless the change explicitly requires otherwise.

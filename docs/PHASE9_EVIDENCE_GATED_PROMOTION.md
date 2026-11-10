# Phase 9 — Evidence-Gated Checkpoint Promotion

## Objective

Prevent a structurally valid but weaker SFT checkpoint from replacing the active SYNTRA model. A successful training run, lower training loss, valid checkpoint hash, or complete benchmark report is not sufficient evidence of improvement.

The promotion command compares real raw-native-model benchmark reports produced from immutable local checkpoint snapshots. Its score is fully verified task completion: both the held-out target tests and regression-prevention tests must pass. Target-test success by itself is not enough. It does not generate benchmark results, fabricate scores, or call a hosted model.

## SFT promotion workflow

1. Run the configured local training workflow and retain its pretraining and SFT benchmark reports.
2. Identify the actual pretraining baseline report and SFT candidate report from the same training run.
3. Run a dry-run with both report paths:

   ```powershell
   npm run brain:promote -- --report "data/benchmarks/<run-id>/evaluations/sft-step-<step>.json" --baseline-report "data/benchmarks/<run-id>/evaluations/pretraining-step-<step>.json" --dry-run
   ```

4. Review the printed candidate/baseline checkpoint hashes, task counts, score delta, improved task IDs, and regression list. The dry-run must report a strictly positive task-completion delta and no regressions.
5. Copy the exact candidate checkpoint SHA-256 from the verified dry-run output and explicitly acknowledge it:

   ```powershell
   npm run brain:promote -- --report "data/benchmarks/<run-id>/evaluations/sft-step-<step>.json" --baseline-report "data/benchmarks/<run-id>/evaluations/pretraining-step-<step>.json" --confirm-sha256 "<verified-checkpoint-sha256>"
   ```

Replace the documented path and digest values with the actual paths and SHA-256 returned by the local run. Do not invent or reuse a digest from another checkpoint.

## Acceptance rules

The promotion verifier independently authenticates both reports and their staged checkpoint snapshots. Each report must:

- use the current checked-in raw-model benchmark suite version;
- contain every task in the suite exactly once;
- show successful hidden-fixture baseline validation for every task;
- contain internally consistent per-task outcomes and aggregate counts;
- identify a real checkpoint snapshot with matching full SHA-256 and trained-step metadata;
- use the network-disabled Docker benchmark isolation mode.

The paired comparison then requires:

- SFT candidate stage is exactly `sft`;
- baseline stage is `pretraining` or `baseline`;
- suite versions and task-ID sets match exactly;
- no task that was fully verified in the baseline may fail either its target tests or regression-prevention tests in the SFT candidate;
- at least one additional task is fully verified after SFT; and
- aggregate fully-verified task rate strictly increases.

A tie is not an improvement. A single per-task regression blocks promotion even if other tasks improve enough to raise the aggregate score.

## First activation

A repository with no active checkpoint can explicitly activate an initial pretraining baseline:

```powershell
npm run brain:promote -- --report "data/benchmarks/<run-id>/evaluations/pretraining-step-<step>.json" --allow-initial-baseline --dry-run
```

After reviewing the result, the actual promotion requires the exact digest returned by dry-run:

```powershell
npm run brain:promote -- --report "data/benchmarks/<run-id>/evaluations/pretraining-step-<step>.json" --allow-initial-baseline --confirm-sha256 "<verified-checkpoint-sha256>"
``` The initial-baseline path is rejected if an active checkpoint, metadata file, or inference companion already exists, and it requires at least one completed task. It cannot be used for an SFT candidate.

## Safety and retained behavior

- Dry-run performs verification and checkpoint loadability checks without modifying the active checkpoint.
- Real promotion still archives the existing checkpoint, metadata, and inference companion before activation.
- The inference companion is rebuilt and validated against the activated checkpoint.
- Failed activation restores the prior archive where one existed.
- Existing rollback behavior remains available through `npm run brain:promote -- --rollback <archive-id>`.
- No Ollama, hosted inference API, model registry, or network service is used.

## Interpretation limits

This gate establishes improvement only on the exact checked-in 13-task raw-model suite. It does not establish broad coding ability, arbitrary repository competence, or production readiness. Continue to inspect held-out task outcomes, loss curves, dataset provenance, and real user-workload behavior before relying on a checkpoint.

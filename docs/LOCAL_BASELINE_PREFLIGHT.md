# Local Baseline Preflight

Use the preflight before committing to a multi-hour SYNTRA baseline run. It is a local readiness check, not a training command and not a benchmark.

## One-command check

From the repository root:

    npm run brain:preflight

To validate a different profile's requested training backend:

    npm run brain:preflight -- --config configs/training/local_gpu_scale.json

For machine-readable output:

    npm run brain:preflight -- --json

The command exits with code 0 only when the required checks pass. A NO-GO result exits with code 2. It does not start pretraining or SFT.

## What it verifies

### Python and native PyTorch

- Python must be version 3.12 or newer.
- PyTorch must import successfully from the local environment.
- CPU readiness is mandatory. The preflight allocates local tensors, performs a matrix operation, and verifies the resulting values.
- CUDA and Apple MPS availability are detected. When available, the preflight performs the same real tensor operation and synchronizes the device.
- The selected training backend must pass the operation. For a profile set to device auto, the selection follows SYNTRA's backend order: CUDA, then MPS, then CPU. Unavailable non-selected accelerators are reported but do not prevent a CPU fallback.

The preflight does not claim that an accelerator is ready merely because PyTorch reports it as available; it executes and verifies an operation on each available backend.

### Local Docker images

The local Docker daemon must respond, and all of these exact tags must already exist:

| Required local tag | Runtime |
| --- | --- |
| python:3.12-slim | Python |
| node:20-slim | JavaScript / TypeScript |
| golang:1.22-alpine | Go |
| rust:1.78-slim | Rust |

For each image, docker image inspect must confirm:

- The requested tag is attached to the inspected image.
- The local image ID is a valid sha256: digest.
- At least one repository digest is present and matches the @sha256: format.
- The image is a Linux image.

A tag without a repository digest is not accepted as a fully verified offline benchmark image. An image that exists but has no recorded repository digest produces NO-GO.

## Network and Docker safety

This command is intentionally read-only. It first checks the active Docker context endpoint and accepts only a local unix:// socket or Windows npipe:// named pipe. Remote tcp://, ssh://, and other remote endpoints are rejected so the readiness gate cannot silently inspect a remote daemon. It then invokes the local Docker client for:

- docker context inspect --format {{.Endpoints.docker.Host}}
- docker info
- docker image inspect <required-tag>

It does not invoke docker pull, docker build, docker run, or a registry API. It does not download, update, retag, or replace images. Missing or unverifiable images must be provisioned separately through the approved image-management process, then rechecked locally.

The GO decision requires Python/PyTorch readiness, the selected training device, a responding local Docker engine, and all four images with the required digest evidence.

## After GO

Run the selected workflow explicitly:

    npm run brain:milestone4

Or run the smoke profile:

    npm run brain:smoke

The epoch driver generates Phase 7 repository and paired multi-file curricula in all mode for Python, JavaScript, TypeScript, Go, and Rust. It merges both streams into the configured SFT dataset, then uses the newly written checkpoint's exact BPE tokenizer to filter paired prompts that cannot fit intact within seq_len - 1 tokens.

The completion is never clipped to make it fit. The thought localization trace, SYNTRA_AGENT_PATCH_V1 payload, and EOS are retained across overlapping context windows; prompt and duplicate-overlap targets remain masked. If the remaining dataset cannot pass the checkpoint-tokenizer audit or lacks an independent validation split, SFT is stopped before optimizer steps begin.

## Interpreting the result

GO confirms only that local software, selected accelerator execution, and the required offline container artifacts are ready. It does not establish:

- that a multi-hour training run will fit available RAM/VRAM;
- that the selected profile is optimal for the host;
- that the generated SFT curriculum contains enough held-out diversity;
- or that training will improve coding-task accuracy.

Review the profile's block size, layers, width, batch size, gradient accumulation, checkpoint destination, corpus provenance, and the actual audit/benchmark outputs before starting a production training run.

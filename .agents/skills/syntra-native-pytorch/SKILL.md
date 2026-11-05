---
name: syntra-native-pytorch
description: Use when changing brain/, syntra_ai/, model-provider selection, startup, training, checkpoints, tokenization, inference, or training-corpus handling in SYNTRA.
---

# SYNTRA Native PyTorch Runtime

## Non-negotiable product direction

SYNTRA's owned local PyTorch brain is the target model architecture. Do not introduce, require, or silently fall back to Ollama, Hugging Face model downloads, OpenAI or other hosted inference APIs, third-party model registries, remote pretrained weights, or external inference services. Do not restore an external provider as a startup prerequisite.

The intended model path is trained and run locally from approved local data with SYNTRA-owned code and checkpoints. Verify actual current code before changing provider logic. The checked-in README may describe legacy or optional integration paths; do not treat those as authorization to make an external runtime part of normal startup. Do not delete an existing path without checking its callers, tests, and current user requirements.

## Model and data integrity

- Do not replace native inference with canned text, prompt templates masquerading as intelligence, mock model responses, or hidden external calls.
- Do not fabricate checkpoints, training progress, loss values, evaluation scores, token streams, embeddings, or model availability.
- Distinguish tokenizer output, sampled model output, deterministic test fixtures, and actual trained-checkpoint inference.
- Preserve checkpoint compatibility, tokenizer state, dataset fingerprints, resume behavior, streaming semantics, and the existing unified runtime unless a tested change is required.
- Read the actual model, tokenizer, checkpoint, runtime, setup, and launcher code before modifying their contracts.
- Keep training data local and respect the repository's approved-corpus rules, including SYNTRA_ALLOWED_CORPUS_ROOTS and canonical path/symlink validation.
- Exclude secrets and sensitive data from corpus synthesis. Never expand data collection beyond the explicitly approved roots.
- Treat a from-scratch small model honestly: a successful train or load proves that the operation ran, not that the model is capable of a particular level of coding or reasoning.

## Startup and provider behavior

The normal one-click startup must not depend on an Ollama process or a downloaded third-party model. Verify the launcher's actual environment handling and the Python setup path. Remove stale provider assumptions only when the evidence identifies them as a defect, and preserve the native provider as the actual default.

A missing checkpoint, failed training step, invalid dataset, or failed Python dependency installation must produce an explicit, actionable failure. Do not report the model as ready until the real runtime has loaded a valid native checkpoint and can perform the requested operation.

## Validation

Use the actual Python tests for model, tokenizer, checkpoint, safe-path, setup, and API behavior, together with relevant Node tests and launcher tests. For model quality claims, use reproducible evaluation on real, declared datasets and report the model/checkpoint and evaluation conditions. Never present synthetic test output as real model performance.

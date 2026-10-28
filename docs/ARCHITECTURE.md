# SYNTRA Architecture

SYNTRA has three local processes:

1. The React/Vite UI provides a single chat conversation and one composer.
2. The Node loopback service exposes chat, selects the active local workspace, orchestrates the repository agent, and manages hash-checked proposal application.
3. The Python loopback service runs the native PyTorch Transformer, checkpoint-matched BPE tokenizer, training, instruction fine-tuning, inference, and fresh sampled next-token evaluation.

## Coding request lifecycle

1. Classify obvious implementation/fix/refactor requests.
2. Inventory source files and build a lightweight import graph from local import statements.
3. Rank files by task terms, paths and dependency neighbours. Allocate a prompt-token budget using the active checkpoint's tokenizer; remove low-ranked files when necessary.
4. Ask the local model for a constrained JSON patch with exact search anchors and replacements.
5. Reject malformed JSON, ambiguous anchors, duplicate paths, path traversal, sensitive paths and oversized edits.
6. Apply the candidate only to a temporary copy. Run discovered build/test/lint/type-check commands with a timeout and capture output.
7. On failed verification, feed the failure to the next attempt (maximum three attempts).
8. If all discovered checks pass, stage and apply hash-checked proposals with backups and audit records. If checks fail or cannot be run, leave the proposals unapplied unless the user types `/apply`.

Original project files are not modified while the agent is planning or testing. A stale file hash blocks a proposal from applying.

## Model and tokenizer

SYNTRA uses a decoder-only Transformer. Its byte-BPE merges are learned from the explicitly selected local corpus and are stored with the model/optimizer checkpoint. The tokenizer is not a hosted or pretrained vocabulary. The default context is 4,096 BPE tokens and the configured maximum is 16,384; greater context increases compute and memory requirements.

## Reliability boundary

The agent loop supplies deterministic tooling and verification around probabilistic model output. It does not make the model Claude-class. The model starts with random weights and only has the knowledge, instruction-following and planning ability produced by actual training. A successful test run proves only that the configured tests passed in the temporary candidate copy.

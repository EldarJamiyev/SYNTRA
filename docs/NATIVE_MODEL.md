# Native Model Runtime

- Decoder-only PyTorch Transformer with causal attention, residual MLP blocks, cross-entropy training, AdamW, gradient clipping, atomic checkpoint writes and checkpoint reload.
- Corpus-trained byte-pair tokenization with stable byte IDs and BOS/EOS markers. The exact merge table is saved in the checkpoint so training, inference, token counting and fine-tuning use the same vocabulary.
- Context range: 16–16,384 BPE tokens; default 4,096. Actual capacity and speed depend on hardware and tokenization. The coding agent derives prompt/output budgets from the selected checkpoint's exact context window; it does not promote a 256/512-token checkpoint to a larger assumed window.
- Native generation preserves the complete prompt or returns a structured `context_window_exceeded` response (HTTP 422). It never silently drops the beginning of a prompt. Callers must reduce or retrieve less context and retry.
- Lightweight inference companions are bound to the source checkpoint's device/inode/size/mtime/ctime identity. Legacy or stale companions fall back to the full checkpoint, including after atomic promotion where file size and mtime can remain unchanged.
- Source pretraining starts with random weights; it is not a pretrained coding model. Instruction fine-tuning uses a local JSONL file with string fields `prompt` and `completion`, and masks prompt-token loss.
- Fresh evaluation measures next-token cross-entropy on a selected local corpus. Reusing training data means it is not an independent evaluation, and language-model loss does not measure patch correctness.
- The repository-aware agent can recover once from a native `context_window_exceeded` response: it re-counts with the checkpoint tokenizer, removes optional retrieved context and low-priority metadata, then retries generation once. The user task, required output contract and system instructions are preserved; prompts that still cannot fit fail explicitly.
- AST-derived Python curriculum symbols carry structural quality signals for documentation, type annotations, control flow, calls and operations. Empty stubs, trivial getters/setters, low-quality declarations and structurally equivalent implementations across approved local repositories are filtered deterministically before dataset selection. Deduplication normalizes local identifiers while preserving operators, call names and argument order.
- The HTTP API binds only to `127.0.0.1`; checkpoint loading uses PyTorch's `weights_only=True` path.

Useful commands:

```powershell
npm run ai:cli -- train --root "C:\\path\\to\\source" --seq-len 1024 --steps 1000
npm run ai:cli -- tokenize --text "def add(a, b): return a + b"
npm run ai:cli -- generate --prompt "Explain this source code:"
npm run ai:cli -- evaluate --root "C:\\path\\to\\heldout\\corpus"
npm run ai:cli -- finetune --dataset "C:\\path\\to\\examples.jsonl"
```

A trained checkpoint is required before chat or coding generation works. From-scratch training is not evidence of Claude-equivalent capabilities; progress must be measured with repeatable coding tasks and real pass/fail test outcomes.

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
- Phase 6 context recovery is bounded to one retry after a native HTTP 422 `context_window_exceeded` response. The active checkpoint tokenizer recounts the prompt, optional retrieved context is compacted in priority order, and the original task, system instructions and output contract remain intact. An irreducible overflow fails explicitly.
- Phase 7 curriculum curation covers Python, JavaScript, TypeScript, Go and Rust. Python declarations use `ast`; the other four languages use a conservative bounded declaration/block parser implemented only with the Python standard library. This is not a compiler or complete language parser: incomplete, oversized or unrecognized blocks are not emitted as complete implementation examples.
- Polyglot symbols are scored for documentation, type information, control flow, calls, operations, statement depth and container members. Trivial stubs/accessors, low-quality declarations and identifier-normalized structural duplicates are filtered using deterministic source ordering and a bounded fingerprint set. API calls, member/property names, operators and argument order are retained in fingerprints.
- Repair synthesis is deliberately narrow and source-verifiable: Python comparisons against the `None` singleton, JavaScript/TypeScript `typeof` comparisons using strict equality, Go error guards returning the declared result/error tuple, and Rust `Option<&T>` slice access without a prior bounds guard. Each exact anchor must be unique; unsupported patterns are omitted.
- The curriculum scanner explicitly excludes `tests/benchmark_engine/hidden` under every approved source root. Hidden benchmark targets are evaluation-only labels and must never enter generated training data.
- The dedicated multi-file curriculum requires a real local dependency relationship and a referenced exported declaration, plus one separately unique repair anchor in each of two files.
- The Phase 8 epoch runner excludes `data/benchmarks`, `data/curricula`, and `data/checkpoints` from both curriculum scans, plus the current run's intermediate/final dataset paths. This prevents prior benchmark reports, checkpoints, or generated instruction data from leaking back into subsequent training corpora. Its completion contains a bounded `<thought>` localization trace followed by a two-edit `SYNTRA_AGENT_PATCH_V1` payload. If the evidence is absent, the generator reports zero records and does not create or replace the output file.
- SFT tokenization still constructs `BOS + prompt + completion + EOS`; all prompt targets and overlapping context targets are masked as `-100`, while every completion token—including the `<thought>` trace and patch JSON—and EOS remains supervised. Long completions continue through overlapping bounded context windows without dropping response tokens.
- Phase 8's epoch driver invokes `npm run --silent ai:curriculum -- --mode all --languages python javascript typescript go rust` for each run. It validates and merges repository/symbol/repair records with paired multi-file reasoning records into the profile's single SFT JSONL dataset. The run-specific intermediate datasets are kept under that run's artifact directory.
- After pretraining (or when reusing an existing checkpoint), the pipeline uses that checkpoint's actual BPE tokenizer to drop only multi-file records whose complete prompt exceeds `seq_len - 1`. It does not silently keep the tail of an oversized paired prompt. Completion targets are not truncated: the full `<thought>` trace, patch JSON, and EOS pass through the existing overlapping context-window loader. Audit failure, insufficient prompt diversity, or unavailable validation split stops SFT before optimizer updates.
- Run `npm run brain:preflight` before a long baseline. It verifies Python 3.12+, real local PyTorch CPU/CUDA/MPS tensor execution, and the four required Docker tags with SHA-256 image IDs and repository digests. Docker inspection is read-only; it never pulls, builds, or runs images.
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


### Phase 9 — evidence-gated checkpoint promotion

A valid benchmark report proves that a checkpoint was evaluated against the current hidden-baseline-validated suite. It does not, by itself, prove that fine-tuning improved the model. The promotion CLI now enforces a paired comparison before an SFT checkpoint can replace the active checkpoint.

For an SFT candidate, the operator must provide both the candidate report and its pretraining baseline report using `--report` and `--baseline-report`. Both reports are independently verified against the current 13-task suite, immutable checkpoint snapshots, full SHA-256 identities, trained-step metadata, and Docker isolation evidence. The comparison uses the benchmark's fully verified task outcome (held-out target tests and regression-prevention tests both pass), not target-test completion alone. It then requires:

- Candidate and baseline reports use the same benchmark suite version and exact task-ID set.
- The baseline checkpoint is a `pretraining` or `baseline` snapshot; the candidate checkpoint is an `sft` snapshot.
- Every task fully verified by the baseline must remain fully verified after SFT. Any target-test or regression-test failure on a previously fully verified task blocks promotion.
- The candidate must fully verify at least one additional task and have a strictly higher aggregate fully-verified task rate.
- The exact candidate checkpoint SHA-256 must still be acknowledged with `--confirm-sha256` for a real promotion.

A first-ever pretraining checkpoint has a separate explicit path: `--allow-initial-baseline`. It is accepted only when the active checkpoint and its companion artifacts do not exist and at least one task passes both held-out target tests and regression-prevention tests. This option cannot bypass SFT comparison and cannot replace an existing active checkpoint.

The `--dry-run` path performs all report, snapshot, paired-score, checkpoint-loadability, and destination checks without changing active files. Its output includes baseline/candidate task counts, score delta, improved task IDs, and an empty regression list. Existing atomic activation, previous-checkpoint archival, companion regeneration, and rollback behavior remain in place.

This gate is intentionally strict but narrow: it measures the checked-in 13-task raw-model suite. It does not establish general coding competence, and passing a small controlled suite is not proof of production readiness. An SFT candidate that ties the baseline is not promoted; a candidate that loses even one previously passing task is not promoted.


### Phase 10 — lossless instruction conditioning

Instruction fine-tuning must preserve the complete conditioning prompt. The JSONL loader now checks the prompt's token count using the selected checkpoint's exact tokenizer and rejects any prompt that cannot fit in the available context after reserving the BOS token. It does not silently retain only the tail of an overlong prompt, because that can remove system constraints, task requirements, or output-format instructions. Operators must shorten the source prompt deliberately or use a checkpoint with a larger context window. Completion text remains fully supervised across the existing overlapping causal windows, including the terminal EOS target.

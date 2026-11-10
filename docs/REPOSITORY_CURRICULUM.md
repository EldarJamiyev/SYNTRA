# Repository Curriculum and Reasoning Synthesis

SYNTRA builds instruction data from approved local source repositories. All generation is deterministic and source-grounded; no hosted model, external API, parser package, or network service is used.

## Modes

The ai:curriculum CLI supports three modes:

- repository — module comprehension, symbol implementation/comprehension, and mechanically verified single-file repairs.
- multifile — paired implementation/consumer localization traces and two-file patch payloads.
- all — both datasets, written to separate JSONL paths.

Examples:

    npm run ai:curriculum -- --mode all --languages python javascript typescript go rust --max-examples 240
    npm run ai:curriculum -- --mode repository --languages python javascript typescript go rust --max-examples 240
    npm run ai:curriculum -- --mode multifile --languages javascript typescript go rust --max-examples 120

The default repository dataset is data/curricula/workspace-instructions.jsonl; the paired reasoning dataset is data/curricula/workspace-reasoning.jsonl. Use --root to select one approved local source root, --output for the repository dataset, and --multifile-output for paired examples. External roots must be included in SYNTRA_ALLOWED_CORPUS_ROOTS. The --exclude option accepts additional absolute file or directory paths and applies each exclusion to both generators. The existing corpus reader continues to apply canonical-path, symlink, size, file-count and sensitive-path controls.

The curriculum scanner adds tests/benchmark_engine/hidden to the exclusion list for every approved root. Hidden benchmark targets are evaluation-only material and are not valid training labels.

## Structural curation

Python declarations are parsed with the standard-library ast module. JavaScript, TypeScript, Go and Rust use a bounded declaration/block parser built on Python standard-library text processing. It is intentionally conservative and is not a complete parser, type checker, or compiler. It limits declaration count and block length, tracks braces outside comments and literals, and refuses to label incomplete or oversized declarations as complete source implementations.

Quality signals include documentation, type information, control flow, calls, meaningful operators, substantive statements and container members. Empty stubs, trivial getter/setter accessors, low-scoring declarations and structural duplicates are filtered. Identifier normalization preserves language keywords, API calls, member/property names, operators and ordered parameters. Fingerprint memory is capped; reaching the cap is reported, and novel declarations are retained rather than being falsely marked as duplicates.

The supported repair rules are intentionally limited:

| Language | Verified pattern | Transformation |
| --- | --- | --- |
| Python | Equality/inequality comparison against the None singleton | == / != becomes is / is not |
| JavaScript / TypeScript | A typeof expression compared to a primitive type string | Loose equality becomes strict equality |
| Go | A guarded error path in a function returning (result, error) | return nil becomes return nil, err |
| Rust | Unchecked slice indexing inside a function returning Option<&T>, with no existing bounds guard | Some(&slice[index]) becomes slice.get(index) |

A candidate is emitted only when the surrounding pattern matches the supported rule and the exact search anchor occurs once in that source file. These rules are not generalized to arbitrary comparisons, error returns, or indexing. Ambiguous, already-guarded, malformed, unsupported, or non-unique anchors are skipped.

## Paired multi-file reasoning examples

The multi-file generator does not pair files because their names happen to look related. It requires a local import/module/package relationship and a source-backed reference to a declaration exported or imported from the implementation file. Python relationships are extracted from AST import / from nodes, JavaScript/TypeScript relationships from local relative imports, Go relationships from same-package test/implementation files, and Rust relationships from local mod / use statements. Source files must remain within the same approved root and selected language.

Both files must independently contain an exact, unique supported repair anchor. The generator includes the complete bounded source for both files in the prompt and produces one edit for each repository-relative path. For safety, paired patch examples are emitted only from the active output workspace (the first selected root); additional approved roots are read-only source material and are never encoded as patch targets for a different workspace. If either side lacks verifiable evidence, no record is emitted. A zero-result run does not create or overwrite the requested dataset path.

The completion begins with a short <thought> localization trace that identifies the dependency, referenced declaration, and separate repair anchors. It is a deterministic, evidence-based training trace—not a model-generated transcript or a claim that a model actually reasoned through the task. It is followed by the JSON SYNTRA_AGENT_PATCH_V1 object with exactly two edits. The existing patch validator remains responsible for applying and validating candidate patches at runtime.

## Supervised fine-tuning and masks

The JSONL record schema remains prompt and completion. The native instruction loader tokenizes:

    BOS + prompt + completion + EOS

All prompt target positions are masked with -100; response tokens remain supervised. For paired examples this means the <thought> trace, patch JSON, and terminal EOS are included in the completion loss. Long completions use the existing overlapping context-window logic, which masks repeated overlap targets to avoid counting the same response token twice. No change to the model architecture or training loss is required.

The automated epoch runner calls the Phase 7 CLI with mode all and the five benchmark languages, keeps both generated outputs in the unique training-run artifact directory, validates the two-file patch structure, and merges repository and multi-file records into the profile's single curriculum.dataset. After checkpoint creation, it measures each multi-file prompt using the selected checkpoint's exact BPE tokenizer. Prompts longer than seq_len - 1 are removed as whole records rather than clipped from the left; this prevents losing the task statement or either source file. Completion text is retained intact and can span multiple overlapping context windows. The manifest records candidate, kept, and dropped counts. The final combined JSONL is audited with the checkpoint tokenizer before any SFT optimizer step.

Run the instruction audit before fine-tuning:

    npm run ai:cli -- audit-instructions --dataset "data/curricula/workspace-instructions.jsonl" --seq-len 1024
    npm run ai:cli -- finetune --dataset "data/curricula/workspace-instructions.jsonl" --steps 1000

The audit checks token limits, duplicate prompts, source-group leakage and train/validation split availability. A small or single-pair corpus may not provide an independent validation split; treat that warning as a data-quality limitation rather than proof of generalization.

## Interpretation

Curriculum counts demonstrate only that records were derived and passed the data checks. They do not demonstrate model correctness, improved patch success, or generalization. Evaluate an unchanged checkpoint against a fixed task set before and after fine-tuning, retain actual pass/fail outcomes, and distinguish training-set loss from held-out coding performance.

# Conversational understanding

## Current implementation

The chat router separates ordinary conversation, read-only workspace questions, and coding tasks. A bounded normalization pass improves routing when users use shorthand, informal phrasing, common misspellings, or supported Azerbaijani, Turkish, and Russian command vocabulary.

The normalizer handles:
- Common English shorthand, contractions, Unicode compatibility forms, and a constrained typo vocabulary.
- A conservative command vocabulary for common Azerbaijani, Turkish, and Russian technical requests.
- Common stopwords that must never be fuzzy-corrected. In particular, ordinary words such as "that" remain unchanged.
- Bounded edit-distance correction with candidate-length indexing and a maximum token-processing budget.
- Original source paths and code identifiers extracted from the untouched user message.
- Object-first commands such as "chat UI fix" without broadly routing ordinary discussion as code work.

Normalization is used only for intent routing. The user's actual prompt, code, file paths, and displayed history remain unchanged. This is deterministic routing assistance, not general semantic understanding or a neural reasoning claim.

## Context-aware follow-up routing

The classifier resolves a bounded set of short follow-ups against the most recent relevant user turn. Consecutive task follow-ups inherit the root request and all prior bounded instructions, rather than dropping earlier requirements after each turn.

- Coding continuations can be routed to the repository agent only when the preceding turn independently resolves to a coding task.
- Workspace follow-ups retain the prior file and symbol references.
- The preceding user turn is evaluated against only the history that existed before it. An unrelated question forms a context boundary and cannot revive an older task.
- The lookup is bounded to the last 24 history entries; at most eight follow-up messages are retained for task-goal construction.
- Generic acknowledgements do not authorize a coding action. Normal proposal approval, cancellation, and project verification safeguards remain in force.

## Known limits

This is not unrestricted coreference resolution, arbitrary translation, or a general-purpose language-understanding model. The multilingual vocabulary is intentionally limited and does not imply complete Azerbaijani, Turkish, or Russian grammar support. Unrecognized or ambiguous wording remains unchanged for the native model to interpret.

The locally initialized PyTorch model is not represented as a pretrained frontier model. Higher-level conversational ability requires substantial curated local instruction data and independent task-level evaluation.

## Next engineering stages

1. Expand grammar and multilingual coverage from observed real prompts, with regression tests for both correct routing and false-positive rates.
2. Introduce conversation-state records with explicit conversation/workspace identity and bounded provenance.
3. Separate action, target entities, constraints, output format, and uncertainty in an inspectable intent representation.
4. Improve the local PyTorch model through provenance-tracked instruction training and independent held-out evaluation.
5. Publish benchmark reports covering typo variants, informal grammar, multi-turn references, technical and non-technical dialogue, multilingual prompts, refusal boundaries, and coding-task routing.

## Non-negotiable constraints

- Native inference remains owned by SYNTRA's PyTorch runtime.
- Do not add Ollama, hosted model providers, Hugging Face inference, remote model registries, or hidden network dependencies.
- Never present a deterministic normalizer as neural reasoning or claim model quality from training loss alone.
- Never silently rewrite the user's original prompt or discard constraints to force a route.
- Workspace and coding actions remain inside existing authorization, approval, cancellation, and verification boundaries.

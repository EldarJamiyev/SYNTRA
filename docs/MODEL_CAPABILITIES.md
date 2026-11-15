# Model capabilities and evaluation

## Local inference

SYNTRA uses one inference implementation: its own local Python/PyTorch Transformer and the checkpoint trained and stored by the project. The desktop startup path prepares the hash-locked Python environment, validates an existing checkpoint or trains a local checkpoint, and starts the loopback AI service. No pretrained model weights, hosted inference API, third-party model provider, or model registry is downloaded or selected.

The native model starts from random initialization. It is a research model for experimenting with tokenization, local training, inference, checkpoint integrity, and repository-agent workflows. It is not a pretrained coding model and must not be described as equivalent to Claude or another frontier coding agent.

The checkpoint architecture and positional-embedding configuration define the context window. Runtime context accounting uses the checkpoint's saved tokenizer and refuses requests that exceed the available context; it must not silently enlarge the model's context or discard required task instructions.

## Training and data boundaries

Source pretraining and instruction fine-tuning use local files and the project's native training pipeline. Review the corpus and instruction dataset before training. Do not include credentials, private keys, customer information, or data that is not authorized for use.

Source pretraining uses a deterministic, size-aware 80/20 holdout of complete source documents. Byte-identical documents are grouped and represented once; no exact copy can appear in both partitions, and duplicate copies do not consume the unique-content token budget. The tokenizer's BPE merges are learned only from the training documents, then applied unchanged to validation documents. Files are never cut at a token boundary to fill the budget. Because files are indivisible, the realized token ratio can differ from 80/20. Training fails closed when fewer than two independent documents or enough tokens for both configured context windows are available; it does not fall back to a token-offset split. The chosen strategy and split counts are saved in checkpoint metadata and exposed through native model status. This exact-content policy does not detect semantic or near-duplicate code.
Fresh corpus evaluation uses the same complete-document partitioner and frozen checkpoint tokenizer, and reports the validation strategy and document counts. A fresh loss measurement on source material already seen during training remains non-independent; this split must not be presented as a coding benchmark.

The application’s native source-pretraining path uses a globally stepped warmup/cosine learning-rate schedule: 100 warmup optimizer steps, a 10,000-step decay horizon, and a learning-rate floor of `min(1e-5, configured base learning rate)`. The schedule is based on the checkpoint’s saved global step, not the length of the current training request, so splitting training across resumable runs does not restart warmup or decay. Checkpoints persist the scheduler version and settings. Checkpoints predating scheduler metadata continue with their original 100-step linear warmup followed by a constant learning rate; a resumed checkpoint never silently changes scheduler family. The trainer rejects non-finite pre-clipping gradient norms and records the measured norm and clipping threshold. These settings are stable defaults, not claims of empirically optimal hyperparameters.

Training loss measures next-token prediction on the supplied data. Historical checkpoints created with the previous token-offset source-validation strategy cannot be resumed by the new document-isolated trainer; start a new source-pretraining checkpoint so the tokenizer and model use the same independent holdout boundary.
It is not a coding-accuracy score and does not prove that generated patches are correct, that unfamiliar tasks can be solved, or that fine-tuning improved generalization. Checkpoint promotion remains gated by paired, hash-bound benchmark evidence and explicit operator acknowledgement.

## Retrieval and embeddings

Semantic source retrieval and project-memory embeddings use the same SYNTRA checkpoint through the local AI service. There is no separate embedding model or external embedding endpoint.

Semantic search validates source hashes before returning excerpts and invalidates cached vectors when source identity or checkpoint identity changes. If native embedding inference is unavailable or returns invalid vectors, project-memory search may use its documented BM25 keyword fallback and must report that fallback accurately. It must not claim that semantic inference succeeded.

Chat history is not embedded as project memory. Only explicitly user-authored workspace notes are eligible for memory embeddings.

## Repository-agent behavior

The repository-aware agent investigates the active workspace through bounded, read-only tools, uses task-aware local skill references as untrusted guidance, and stages structured candidate changes for review. Candidate checks, checkpoint hashes, source hashes, and verification outcomes are recorded as evidence.

Generated patches are not silently applied. Applying proposals requires explicit user action. A model-generated answer or self-review is not proof that a change is correct; the relevant project checks must pass.

## Self-review

Patch review uses the same native checkpoint as generation. It is a second reasoning pass, not an independent-model guarantee. When review is enabled and no accepted verdict is produced, the candidate is blocked rather than silently treated as reviewed.

## Benchmark evidence

The benchmark command runs the actual repository agent against disposable, versioned fixture copies using a trained SYNTRA checkpoint. It requires the exact expected changed-file set and passing candidate verification for task completion. Reports record the native engine, checkpoint identity, task outcomes, latency, token counts, and the limitations of process-level memory measurements.

The benchmark suite is a controlled development regression set, not an industry-wide coding benchmark. Passing SYNTRA's own implementation tests validates the software and test harness; it does not establish coding ability of a live checkpoint. Only a real benchmark run against the checkpoint on the target hardware creates model-performance evidence. Skipped language runtimes are not counted as successful tasks, and missing measurements must remain unavailable rather than being estimated or backfilled.

## Task-aware skill grounding

The agent discovers bounded `SKILL.md` references from SYNTRA's bundled skills and the active workspace. It selects applicable project guardrails and ranks task-specific playbooks within the native checkpoint's measured context budget. Skill text is untrusted reference material: it cannot expand the tool allowlist, change permission boundaries, or authorize workspace mutations.

Selected skill identities, excerpt hashes, and retrieval/compaction outcomes are recorded with agent-run evidence. Skill retrieval provides procedural context; it does not add pretrained weights or prove that the model followed the guidance.

## Inference boundary

All model requests are routed to SYNTRA's local PyTorch service. The application, launcher, repository agent, and benchmark runner do not integrate third-party inference endpoints, external embedding services, or model registries. A missing or invalid native checkpoint is reported as an error and must be resolved through SYNTRA's local training and checkpoint-validation workflow.

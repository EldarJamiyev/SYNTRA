# Phase 28 — Native task-aware skill retrieval

## Objective

Make SYNTRA's bundled .agents/skills library and optional active-workspace skills operational in the existing native coding-agent workflow. Relevant engineering playbooks are supplied as bounded reference material to the native PyTorch model, without introducing a hosted inference provider or a second orchestration system.

## Delivered implementation

- Discover immediate child skill directories beneath SYNTRA's bundled .agents/skills directory and the active workspace's .agents/skills directory. Bundled skills take precedence when the same slug exists in both places.
- Read SKILL.md metadata and bounded excerpts; reject symlinked directories/files, path escapes, unreadable entries, and files above the size ceiling.
- Always include the SYNTRA project guardrails and verification playbook when present.
- Rank other skills deterministically against the user's task using frontmatter names and descriptions; include only the highest-scoring relevant playbook.
- Limit the selection to three skills and a 7,200-character total prompt section. The native checkpoint tokenizer remains authoritative for the final context budget.
- Treat selected skill text as untrusted reference material. It cannot authorize tools, shell execution, permission changes, proposal application, or weakening of the existing safety model.
- Preserve the skills selected, their excerpt hashes, and retrieval status in agent-run evidence.
- Progressively compact the selected skill section using the checkpoint's actual tokenizer budget; omit it when necessary before sacrificing the user's task, patch schema, or core request.

## Verification

The Node regression suite covers default guardrails, relevance ranking, deterministic selection, bounded prompt size, missing libraries, oversized files, and symbolic-link exclusion. An agent-level integration test verifies that the retrieved guidance reaches the actual native patch-generation prompt.

## Explicit limitations

Skill retrieval improves task-specific procedural grounding; it does not add model weights, pretraining, semantic reasoning, or guaranteed coding quality. A skill being selected is not proof that the model followed it or that generated changes are correct. The existing candidate verification and proposal review remain authoritative.

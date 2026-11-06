---
name: ui-ux-pro-max
description: Apply a broad, evidence-minded UI/UX review to SYNTRA's web chat without imposing generic templates.
license: MIT
source: adapted
source_repo: sickn33/agentic-awesome-skills
source_ref: 828257c3037dc7c3c3ec987decc5a603025a0ce5
---

# UI/UX Pro Max

Adapted from the UI/UX Pro Max guidance in `sickn33/agentic-awesome-skills`. This repository copy is a focused playbook, not the upstream searchable dataset or scripts.

## Review priority

1. Accessibility: semantic structure, names, contrast, focus, keyboard and reduced motion.
2. Interaction: responsive controls, clear loading/streaming, disabled and failure states, no duplicate submissions.
3. Performance: avoid blocking renders, unnecessary rerenders, layout shifts and unbounded content.
4. Layout: readable conversation width, responsive behavior, safe overflow, stable composer placement.
5. Typography and color: consistent hierarchy and legible contrast using existing tokens.
6. Motion: only purposeful feedback; no motion that hides state or interrupts task completion.
7. Visual style: extend SYNTRA's current chat-first design rather than selecting a new template.

## Workflow

- Identify the existing product pattern, users, target viewport and exact requested change.
- Inspect the current design tokens and components before introducing a new pattern.
- Make the smallest coherent design improvement. Do not add a dashboard or generic SaaS cards.
- Keep icons consistent with the installed icon library; do not use emoji as functional icons.
- Test keyboard operation, visible focus, reduced-motion behavior, long conversation content, narrow viewports and actual streaming/error states.
- Inspect the rendered UI when browser tools are available; source code alone does not prove visual quality.
- Use real content and real state. Never add placeholder AI responses, fabricated telemetry, fake testimonials, simulated operations or demo-only interactions.
- Do not install fonts, animation libraries or component frameworks without confirming the need and compatibility.

Use recommendations as heuristics, not as evidence of conversion, accessibility certification or performance improvements.

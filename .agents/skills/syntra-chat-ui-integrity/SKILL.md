---
name: syntra-chat-ui-integrity
description: Use when changing SYNTRA's React interface, conversation flow, chat components, frontend state, styling, accessibility, or responsive behavior.
---

# SYNTRA Chat UI Integrity

SYNTRA is a focused, chat-first local engineering workspace. Preserve that product experience and its existing visual language.

## Design constraints

- Inspect the current React components, style tokens, layout, and working UI before changing them.
- Keep conversation as the primary interaction. Do not convert the product into a generic admin dashboard, analytics cockpit, multi-panel SaaS template, or button-heavy control surface.
- Do not add a generic component library, design system, or broad UI dependency when a focused change using the existing implementation is sufficient.
- Reuse the repository's existing typography, color tokens, spacing, surfaces, icons, transitions, and responsive patterns. Discover their actual values from source; do not invent a new visual theme.
- Preserve current working flows, conversation history, streaming behavior, workspace selection, proposal/review interactions, and error states.
- Prefer clear hierarchy and restrained presentation over decorative cards, gradients, badges, oversized headings, or excessive borders.
- Maintain accessible labels, keyboard behavior, visible focus, semantic controls, readable contrast, and responsive behavior.
- Do not hide loading, failure, permission, or unavailable states behind optimistic UI or placeholder content.

## Verification

Use the existing frontend tests and Playwright end-to-end suite for behavior changes. Review the actual rendered application at relevant viewport sizes when browser tooling is available. Do not claim visual verification without an actual browser run or captured evidence.

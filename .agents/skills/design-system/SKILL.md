---
name: design-system
description: Preserve and evolve SYNTRA's established chat-first visual system consistently across components and responsive states.
license: MIT
source: SYNTRA local
---

# SYNTRA design system

Use before adding or revising visual components, tokens, typography, layout or motion.

1. Inspect `src/style.css`, the existing React components and rendered application before proposing a visual change.
2. Preserve SYNTRA's chat-first, minimal interface. Do not turn it into a dashboard or add statistics panels, settings surfaces, cards or navigation that the task does not require.
3. Extend existing CSS variables, spacing, colors, radii, typography and component patterns instead of introducing competing token systems.
4. Prefer semantic HTML and existing Lucide icons. Avoid emoji icons, arbitrary Unicode symbols and unnecessary dependencies.
5. Design all real states: initial, loading, streaming, empty, success, error, disabled, focus, keyboard navigation, small viewport and long content.
6. Keep motion purposeful, brief, interruptible and compatible with `prefers-reduced-motion`. Essential meaning must never depend on animation.
7. Test at the viewport sizes relevant to the actual UI and inspect the rendered result after code changes.
8. Do not claim visual consistency from CSS inspection alone; confirm the browser output when browser tooling is available.

The current implementation is the source of truth. Do not invent a new brand palette or overwrite the design language to match a generic template.

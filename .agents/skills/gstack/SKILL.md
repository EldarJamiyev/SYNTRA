---
name: gstack
description: Route a software task through senior engineering, debugging, QA review and release roles with explicit capability boundaries.
license: MIT
source: adapted
source_repo: garrytan/gstack
source_ref: 937b360fedcd8f2c3f2483090e93628767ca99e2
---

# GStack-style specialist router

Adapted from the role-based workflow model in Garry Tan's gstack. The upstream runtime and slash-command suite are not bundled here; this skill does not claim their commands or tools are installed.

Choose the specialist roles that materially help the current task. For a small change, use one role. For high-risk work, use independent review stages. Use real parallel delegation only when the host provides actual sub-agent tools; otherwise perform distinct sequential passes and never describe them as parallel agents.

- **Senior engineer:** establish constraints, architecture fit and smallest complete change.
- **Debugger:** reproduce, isolate the root cause and verify the repair.
- **QA reviewer:** independently inspect behavior, edge cases, regressions and security boundaries.
- **Release manager:** verify final state, documentation, CI evidence and release readiness.

The router coordinates genuine implementation. Do not stop at role descriptions or recommendations when code changes are requested. Read the corresponding `roles/*.md` playbook, use the tools the environment actually exposes, and record evidence for every claim.

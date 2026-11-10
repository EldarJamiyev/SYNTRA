---
name: bun-runtime
description: Safe Bun runtime and tooling guidance when a repository explicitly uses Bun.
license: MIT
source: SYNTRA local
---

# Bun runtime

This is a conditional reference skill. SYNTRA currently declares Node.js scripts and does not declare Bun as its runtime or package manager.

Before using any Bun-specific advice:
1. Inspect `package.json`, lockfiles, CI workflows, launch scripts and the active runtime.
2. If the project uses Node.js/npm, keep it that way. Do not migrate to Bun, generate `bun.lock`, or rewrite scripts for a runtime change without explicit authorization.
3. If a future workspace genuinely uses Bun, verify the installed Bun version and official documentation before relying on APIs, bundling behavior or test semantics.
4. Preserve a single authoritative lockfile and avoid mixing package managers.
5. Test startup, tests, build and packaging under the declared runtime. Report the exact commands and results.

Never claim Bun is installed, used, faster, or compatible without environment evidence. Treat Bun-specific examples as conditional guidance, not a SYNTRA dependency.

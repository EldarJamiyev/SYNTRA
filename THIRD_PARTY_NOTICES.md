# Third-party agent skill notices

SYNTRA vendors the upstream skill files listed below under their respective licenses. Imported files are retained with their upstream content and supporting references. SYNTRA-specific guardrail skills in `.agents/skills/syntra-*/` are original project-local instructions.

| Upstream source | Imported skills | Pinned source commit | License |
|---|---|---|---|
| [obra/superpowers](https://github.com/obra/superpowers) | test-driven-development, systematic-debugging, verification-before-completion, using-git-worktrees, writing-plans, executing-plans, requesting-code-review, receiving-code-review, subagent-driven-development, dispatching-parallel-agents | `22f0b463e7a5e579904dd6211a520321035526f7` | MIT; full notice in `.agents/skills/_licenses/obra-superpowers-MIT.txt` |
| [addyosmani/agent-skills](https://github.com/addyosmani/agent-skills) | api-and-interface-design, code-review-and-quality, code-simplification, incremental-implementation, observability-and-instrumentation, performance-optimization, security-and-hardening, source-driven-development | `301b02b12bff98e4688e6984af6556802322bbde` | MIT; full notice in `.agents/skills/_licenses/addyosmani-agent-skills-MIT.txt` |
| [anthropics/skills](https://github.com/anthropics/skills) | frontend-design | `ca45fc496a3436251f191354091d557da31c15b5` | Apache-2.0; per-skill license retained at `.agents/skills/frontend-design/LICENSE.txt` |
| [mattpocock/skills](https://github.com/mattpocock/skills) | codebase-design | `4ae2317ddb22b046e26455e4f9c9b3aaaf62212d` | MIT; full notice in `.agents/skills/_licenses/mattpocock-skills-MIT.txt` |

The project lockfile `skills-lock.json` records the source, pinned commit, source skill path, and upstream skill-folder tree hash for every imported skill. Review and update imported skills deliberately; do not replace pinned sources with unreviewed moving branches.

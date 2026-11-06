---
name: token-budget-advisor
description: Estimate task and response token budgets using SYNTRA's actual tokenizer and configured context limits.
license: MIT
source: SYNTRA local
---

# Token budget advisor

Use before unusually long code-generation, analysis, repository-ingestion, or documentation tasks.

- Read the active model/checkpoint configuration and the tokenizer implementation before quoting a context limit or estimating usage.
- Use the tokenizer that matches the selected checkpoint. Do not substitute character counts, another model's tokenizer, or a generic tokens-per-word ratio and present it as an exact value.
- Separate measured input tokens, measured generated tokens, reserved output budget, and estimated retrieval overhead. Label estimates as estimates.
- Keep room for tool results, test output and follow-up reasoning. Do not allocate the entire context window to the initial prompt.
- For an expansive deliverable, offer 25%, 50%, 75% or 100% depth only when the user has not already specified the desired scope. Explain what each depth changes; never silently truncate required acceptance criteria.
- If context capacity or counters are not exposed by the runtime, report that limitation rather than inventing a budget.
- Prefer targeted source retrieval and bounded excerpts over repeated full-repository scans.

A token estimate is not a quality score and does not authorize dropping required work.

---
name: git-commit-formatter
description: Write concise Conventional Commit messages that accurately describe the actual diff.
license: MIT
source: adapted
source_repo: rominirani/antigravity-skills
source_note: Independently authored adaptation; upstream repository has no declared repository license.
---

# Git commit formatter

Use when preparing a commit message.

Format: `<type>(<optional-scope>): <imperative description>`

Allowed types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`.

- Determine the type from the real changes, not the intended changes.
- Use a scope only when it clarifies the affected subsystem.
- Keep the subject concise, imperative and specific.
- Use a body for important rationale or non-obvious behavior.
- Add a `BREAKING CHANGE:` footer only when compatibility is actually broken.
- Never claim tests passed or functionality was added in the commit message unless supported by the diff and test evidence.

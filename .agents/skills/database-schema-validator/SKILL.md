---
name: database-schema-validator
description: Review SYNTRA database schemas and migrations for destructive changes, integrity and compatibility.
license: MIT
source: adapted
source_repo: rominirani/antigravity-skills
source_note: Independently authored adaptation; upstream repository has no declared repository license.
---

# Database schema validator

Inspired by the database-schema-validator concept in rominirani/antigravity-skills. This skill does not include or claim to run the upstream tutorial script.

- First determine whether the task actually uses a database and identify the real engine, schema files, migrations and persistence layer.
- Inspect schema consumers, initialization, existing user data and migration history.
- Flag data loss, table/column drops, type narrowing, nullability changes, key changes, index locks and incompatible writes.
- Check primary keys, foreign keys, uniqueness, defaults, indexes and transaction behavior according to the domain. Do not impose a universal `id` column rule when the model does not require one.
- Prefer additive and reversible migrations. Explain any irreversible step and require explicit authorization before executing it.
- Validate with the project's real schema parser, migration runner and integration tests when available. Do not create a validator script and report success unless it was actually run.
- Verify upgrade, fresh initialization and rollback behavior where the project supports them.

Report confirmed violations separately from design recommendations.

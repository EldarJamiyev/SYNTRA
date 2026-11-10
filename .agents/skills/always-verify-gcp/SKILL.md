---
name: always-verify-gcp
description: Verify Google Cloud CLI and SDK guidance against current official documentation when a task explicitly uses GCP.
license: MIT
source: adapted
source_repo: rominirani/antigravity-skills
source_note: Independently authored adaptation; upstream repository has no declared repository license.
---

# Always verify GCP

This conditional skill applies only when the requested task explicitly uses Google Cloud Platform.

1. Confirm the actual project, service, CLI or SDK and installed version. Do not assume SYNTRA uses GCP.
2. Verify command syntax, flags, authentication and API behavior against current official Google Cloud documentation for the installed version.
3. Distinguish a documented command from a command actually executed in the user's environment.
4. Never invent resource names, project IDs, regions, command output or deployment state.
5. Use read-only inspection first. Destructive cloud actions, permission changes, secret changes and deployments require explicit authorization.
6. If the task has no GCP dependency, do not invoke this skill or add Google Cloud dependencies.

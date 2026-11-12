# Security policy

SYNTRA processes local source code and may send selected prompts and repository excerpts to an explicitly configured remote hybrid model endpoint. Treat both the local workspace boundary and any configured provider as security-sensitive.

## Reporting a vulnerability

Please do not publish exploitable vulnerabilities, credentials, private repository content, or complete attack instructions in a public issue.

Use GitHub's private vulnerability reporting for this repository when it is enabled. Otherwise, contact the repository owner through the GitHub profile and request a private disclosure channel before sending technical details.

Include the affected revision, impact, preconditions, a minimal reproducible description, and any safe regression evidence. Redact tokens, private paths, personal data, and customer source code.

## Scope

Security reports are especially relevant to:
- Loopback API origin/host validation and local session handling.
- Workspace path resolution, symlinks, file identity, and proposal application.
- Approval capabilities and execution boundaries.
- Check isolation, candidate integrity, and recovery.
- Dependency and build supply-chain integrity.
- Hybrid provider configuration, HTTPS enforcement, credentials, redirects, and model/data identity.

Do not assume that a remote hybrid provider is private. Operators are responsible for approving the provider's data-handling terms before sending confidential repository context.

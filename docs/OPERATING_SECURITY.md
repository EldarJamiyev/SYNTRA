# Local execution and security model

## Local API boundary

SYNTRA binds its application service to loopback. The local API validates the Host header, accepts only configured loopback browser origins, ignores forwarded-host headers, and requires a process-scoped bearer session for non-public API routes. The frontend keeps that session token in memory rather than local storage.

Privileged tool calls use short-lived, one-time capabilities bound to the resolved workspace, tool name, and exact input. Legacy apply/check endpoints use the same capability mechanism. A caller-supplied boolean is not an approval capability.

## Project verification

A candidate workspace copy protects the original source files from ordinary edits made by test commands. It is not, by itself, an operating-system sandbox.

Project checks are disabled by default. For a containerized check, set `SYNTRA_CHECK_ISOLATION=docker`. SYNTRA starts a Docker container with an init process, no network, private IPC, a read-only container root, dropped Linux capabilities, no-new-privileges, process/memory/swap/CPU limits, an open-file limit, a no-exec temporary filesystem, and only the candidate workspace mounted writable at `/workspace`. The Docker daemon and selected image remain part of the trusted computing base. The writable candidate mount is intentionally limited to the temporary copy; the original project workspace is not mounted into the container. Images must be available locally or Docker must be able to obtain them before the container starts.

For a workspace you personally trust, a user may explicitly opt into host execution with both `SYNTRA_CHECK_ISOLATION=host` and `SYNTRA_ALLOW_UNSANDBOXED_CHECKS=1`. This is not sandboxing. Repository-defined build/test scripts can execute with the current OS user account's permissions. Do not use this mode for untrusted repositories.

The regression-test launcher sets the host opt-in only in its child test process because its temporary fixtures are controlled test inputs. This does not change the defaults of the launched SYNTRA application.

## Data retention

Chat sessions are held in the browser's local storage and are not encrypted at rest. Sessions older than 30 days are pruned when history is loaded. Avoid putting passwords, API keys, private keys, or other credentials in chat. Workspace audit logs are bounded to a 5 MiB active file plus five rotated archives; individual records are capped.

## Residual risks

The local session is a bearer secret available to JavaScript running in the same origin. Preventing same-origin script injection remains important. The loopback API and origin checks reduce browser-origin attack surface but are not a substitute for operating-system account isolation. The Docker runner still trusts the Docker engine and the mounted candidate directory.

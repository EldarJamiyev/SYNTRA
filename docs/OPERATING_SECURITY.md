# Local execution and security model

## Local API boundary

SYNTRA's Node workspace API, Vite development server, and Vite preview server bind to loopback (`127.0.0.1`). The Vite host allowlist accepts only `localhost` and `127.0.0.1`; the maintained npm scripts do not expose the frontend on all network interfaces. The local API validates the Host header, accepts only configured loopback browser origins, ignores forwarded-host headers, and requires a process-scoped bearer session for non-public API routes. The frontend keeps that session token in memory rather than local storage.

Privileged tool calls use short-lived, one-time capabilities bound to the resolved workspace, tool name, and exact input. Legacy apply/check endpoints use the same capability mechanism. A caller-supplied boolean is not an approval capability.

## Project verification

A candidate workspace copy protects the original source files from ordinary edits made by test commands. It is not, by itself, an operating-system sandbox.

Project checks are disabled by default. For a containerized check, set `SYNTRA_CHECK_ISOLATION=docker`. SYNTRA starts a Docker container with no network, a read-only container root, dropped Linux capabilities, no-new-privileges, process/memory/CPU limits, a temporary filesystem, and only the candidate workspace mounted writable at `/workspace`. The Docker daemon and selected image remain part of the trusted computing base. Images must be available locally or Docker must be able to obtain them before the container starts.

For a workspace you personally trust, a user may explicitly opt into host execution with both `SYNTRA_CHECK_ISOLATION=host` and `SYNTRA_ALLOW_UNSANDBOXED_CHECKS=1`. This is not sandboxing. Repository-defined build/test scripts can execute with the current OS user account's permissions. Do not use this mode for untrusted repositories.

The regression-test launcher sets the host opt-in only in its child test process because its temporary fixtures are controlled test inputs. This does not change the defaults of the launched SYNTRA application.

## Native model data paths

The native Python API does not treat an arbitrary absolute path supplied in an HTTP request as trusted. Training and evaluation roots, plus fine-tuning JSONL files, must resolve inside the repository root or an explicitly approved directory in `SYNTRA_ALLOWED_CORPUS_ROOTS`. The environment variable is a platform-separated list of existing directories (`;` on Windows, `:` on Linux/macOS). Resolution occurs before containment checks; filesystem-root inputs, traversal outside an approved directory, missing paths, and non-JSONL fine-tuning files are rejected. Keep approved directories as narrow as practical rather than allowing an entire home or system directory.


## Native corpus and curriculum filesystem boundaries

Native source discovery re-canonicalizes each selected root and each discovered file at the point of use. The canonical candidate must be equal to, or a path-component descendant of, an explicitly approved root. Corpus reads additionally reject paths whose absolute spelling differs from their resolved path, use no-follow file opening where the platform supports it, require a regular file, and compare device/inode/size/mtime/ctime before and after reading. Directory traversal repeats containment checks for each visited directory and child directory; exclusions are compared as canonical path components rather than unchecked string fragments.

Instruction JSONL loading repeats the approved-root containment check immediately before opening the file, then verifies the opened file's identity and metadata before parsing its bounded UTF-8 content. Curriculum output resolution requires the selected workspace itself to be approved. Output parents are checked before directory creation and checked again after creation; the atomic writer repeats the root check at the write boundary and rejects symbolic-link destinations. These repeated checks reduce path substitution risk but do not replace operating-system isolation against an attacker who can concurrently modify the trusted directory tree.

Curriculum generation excludes `tests/benchmark_engine/hidden` beneath every approved root. Multi-file reasoning examples are emitted only for source relationships and unique patch anchors verified within the active output workspace (the first selected root). Additional approved roots remain read-only source material and are not used to create edits that would target a different workspace.

## Data retention

Chat sessions are held in the browser's local storage and are not encrypted at rest. Sessions older than 30 days are pruned when history is loaded. Avoid putting passwords, API keys, private keys, or other credentials in chat. Workspace audit logs are bounded to a 5 MiB active file plus five rotated archives; individual records are capped.

## Residual risks

The local session is a bearer secret available to JavaScript running in the same origin. Preventing same-origin script injection remains important. The loopback API and origin checks reduce browser-origin attack surface but are not a substitute for operating-system account isolation. The Docker runner still trusts the Docker engine and the mounted candidate directory.

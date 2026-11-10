# Phase 19 — End-to-End Chat Cancellation Integrity

## Defect addressed

The local chat API exposed `/local-api/chat/cancel`, but cancellation was not consistently connected to the work performed by each chat path. In particular, ordinary chat tokenization and streamed generation omitted the job's AbortSignal. Several shortcut operations and local network diagnostics also ran without it. The API could acknowledge cancellation while work continued, and a late result could finish a job after cancellation.

## Implementation

- Pass the chat AbortSignal to normal-chat tokenizer and native inference requests, including streamed generation.
- Pass the signal to workspace secret scanning, proposal listing, transactional apply, proposal discard, model status, and workspace switching.
- Make terminal chat completion detect an already-aborted controller and retain the cancelled state rather than accepting a late result.
- Keep durable agent-run cancellation updates scoped to the workspace captured when that chat job started.
- Support cancellation in local network diagnostics: TCP probes destroy their socket, ping terminates its child process, and DNS callers stop waiting promptly when aborted.
- Preserve the existing rule that a transactional apply already in progress cannot be interrupted through the chat cancellation endpoint.

## Regression coverage

- An HTTP integration test starts the actual local SYNTRA chat server and connects it to a local test inference endpoint. The test holds an NDJSON generation response open, requests `/cancel`, verifies the local AI response is closed by the propagated abort signal, and confirms the chat job remains `cancelled`.
- A TCP diagnostic test holds a socket open, aborts the task, and verifies both an `AbortError` result and socket destruction.
- Existing end-to-end chat, network diagnostic, workspace, proposal, agent-run, frontend, and native-model tests remain part of the repository's CI matrix.

## Boundaries

DNS resolution is not forcibly cancelled inside the platform resolver; the waiting chat operation is released promptly and its result is discarded. Cancellation is not retroactive: operations already committed transactionally remain committed. The change does not add process-wide cancellation, change permission policy, or claim model-quality or performance improvements.

## Verification

GitHub Actions for the Phase 19 pull request is the validation source of truth. No test result is considered passed until the corresponding workflow job completes successfully.

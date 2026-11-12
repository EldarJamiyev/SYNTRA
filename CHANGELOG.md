# Changelog

All notable user-facing changes will be documented here.

## Unreleased

### Added
- Explicit opt-in OpenAI-compatible inference for chat and repository coding-agent generation, with provider/model verification, bounded context settings, and no silent fallback.
- Provider provenance in coding benchmark reports and an explicit indication when hybrid context/token measurements are estimates.
- A real browser-based React UI screenshot capture in CI and a visual architecture overview.
- Contribution, security, issue-reporting, and pull-request guidance.

### Changed
- Preserve the configured hybrid context independently of the native model's architectural context ceiling.
- Enforce the Node.js version range required by the current Vite runtime in package metadata, startup validation, and CI.
- Clarify that no model-performance score or pretrained starter checkpoint is published until genuine artifacts and reports exist.

### Not included
- No pretrained SYNTRA checkpoint or model release asset is published by this change.
- No benchmark performance numbers are fabricated or inferred from software test results.

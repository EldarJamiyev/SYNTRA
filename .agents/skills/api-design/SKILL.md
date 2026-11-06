---
source_repo: sickn33/agentic-awesome-skills
source_ref: 828257c3037dc7c3c3ec987decc5a603025a0ce5
source_path: skills/api-design-principles/SKILL.md
name: api-design
description: "Master REST and GraphQL API design principles to build intuitive, scalable, and maintainable APIs that delight developers and stand the test of time."
risk: safe
source: community
date_added: "2026-02-27"
---

# API Design Principles

## SYNTRA applicability
Preserve the existing local API surface and client/server contract. Prefer the smallest compatible contract change, validate request and response boundaries, and do not add GraphQL or a new schema library unless the existing architecture and task justify it.
Master REST and GraphQL API design principles to build intuitive, scalable, and maintainable APIs that delight developers and stand the test of time.

## Use this skill when

- Designing new REST or GraphQL APIs
- Refactoring existing APIs for better usability
- Establishing API design standards for your team
- Reviewing API specifications before implementation
- Migrating between API paradigms (REST to GraphQL, etc.)
- Creating developer-friendly API documentation
- Optimizing APIs for specific use cases (mobile, third-party integrations)

## Do not use this skill when

- You only need implementation guidance for a specific framework
- You are doing infrastructure-only work without API contracts
- You cannot change or version public interfaces

## Instructions

1. Define consumers, use cases, and constraints.
2. Choose API style and model resources or types.
3. Specify errors, versioning, pagination, and auth strategy.
4. Validate with examples and review for consistency.

Refer to `resources/implementation-playbook.md` for detailed patterns, checklists, and templates.

## Resources

- `resources/implementation-playbook.md` for detailed patterns, checklists, and templates.

## Limitations
- Use this skill only when the task clearly matches the scope described above.
- Do not treat the output as a substitute for environment-specific validation, testing, or expert review.
- Stop and ask for clarification if required inputs, permissions, safety boundaries, or success criteria are missing.

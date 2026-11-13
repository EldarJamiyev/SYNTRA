import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import { loadRelevantSkills, MAX_SKILL_CONTEXT_CHARS } from "../local/skill-library.mjs";

async function createWorkspace() {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "syntra-skill-library-"));
  await fs.mkdir(path.join(root, ".agents", "skills"), { recursive: true });
  return root;
}

async function writeSkill(root, slug, description, body) {
  const directory = path.join(root, ".agents", "skills", slug);
  await fs.mkdir(directory, { recursive: true });
  await fs.writeFile(path.join(directory, "SKILL.md"),
    "---\nname: " + slug + "\ndescription: " + description + "\n---\n\n" + body + "\n",
    "utf8");
}

async function removeWorkspace(root) {
  await fs.rm(root, { recursive: true, force: true, maxRetries: 8, retryDelay: 100 });
}

test("retrieval always includes SYNTRA guardrails and verification, then selects a task-relevant skill", async () => {
  const root = await createWorkspace();
  try {
    await writeSkill(root, "syntra-project-guardrails",
      "Use for every task in the SYNTRA repository.",
      "# SYNTRA Project Guardrails\nPreserve architecture, permissions, evidence, and local-first behavior.");
    await writeSkill(root, "verification-before-completion",
      "Use before reporting any task as complete.",
      "# Verification Before Completion\nRun the relevant checks and inspect their actual results.");
    await writeSkill(root, "systematic-debugging",
      "Use when encountering any bug, test failure, or unexpected behavior.",
      "# Systematic Debugging\nFind the root cause before proposing a fix.");
    await writeSkill(root, "frontend-design",
      "Use when building a new user interface or designing visual layouts.",
      "# Frontend Design\nUse deliberate layout and typography choices.");

    const result = await loadRelevantSkills(root, "Fix the failing test caused by unexpected behavior in the API service.");
    const names = result.skills.map(skill => skill.slug);

    assert.equal(result.status, "selected");
    assert.equal(result.discovered, 4);
    assert.ok(names.includes("syntra-project-guardrails"));
    assert.ok(names.includes("verification-before-completion"));
    assert.ok(names.includes("systematic-debugging"));
    assert.ok(!names.includes("frontend-design"));
    assert.match(result.context, /Find the root cause before proposing a fix/u);
    assert.ok(result.context.length <= MAX_SKILL_CONTEXT_CHARS);
    assert.ok(result.skills.every(skill => /^[a-f0-9]{64}$/u.test(skill.excerptSha256)));
  } finally {
    await removeWorkspace(root);
  }
});

test("skill ranking and selected excerpts are deterministic and bounded", async () => {
  const root = await createWorkspace();
  try {
    await writeSkill(root, "syntra-project-guardrails", "Always preserve SYNTRA safety boundaries.",
      "# Guardrails\nNever bypass approvals.");
    await writeSkill(root, "security-and-hardening", "Use when handling untrusted input and security boundaries.",
      "# Security\nValidate input, preserve authorization, and keep audit evidence.");
    await writeSkill(root, "api-and-interface-design", "Use when designing APIs and public interfaces.",
      "# API design\nDefine stable contracts and validate them at boundaries.");

    const first = await loadRelevantSkills(root, "Harden API input validation and authorization.");
    const second = await loadRelevantSkills(root, "Harden API input validation and authorization.");

    assert.deepEqual(first.skills.map(skill => skill.slug), second.skills.map(skill => skill.slug));
    assert.equal(first.context, second.context);
    assert.ok(first.context.length <= MAX_SKILL_CONTEXT_CHARS);
    assert.ok(first.skills.some(skill => skill.slug === "security-and-hardening"));
  } finally {
    await removeWorkspace(root);
  }
});

test("missing skill libraries are reported without failing the coding-agent task", async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "syntra-no-skills-"));
  try {
    const result = await loadRelevantSkills(root, "Implement a feature.");
    assert.equal(result.status, "not_available");
    assert.equal(result.skills.length, 0);
    assert.equal(result.context, "");
    assert.match(result.reason, /No safe/u);
  } finally {
    await removeWorkspace(root);
  }
});

test("oversized skill files and symbolic-link skill folders are excluded", async () => {
  const root = await createWorkspace();
  const outside = await fs.mkdtemp(path.join(os.tmpdir(), "syntra-skill-outside-"));
  try {
    await writeSkill(root, "syntra-project-guardrails",
      "Use for every task in SYNTRA.",
      "# Guardrails\nKeep changes local and verifiable.");
    const oversizedDirectory = path.join(root, ".agents", "skills", "oversized");
    await fs.mkdir(oversizedDirectory, { recursive: true });
    await fs.writeFile(path.join(oversizedDirectory, "SKILL.md"), "x".repeat(1024 * 1024 + 1));

    await writeSkill(outside, "systematic-debugging",
      "Use for debugging test failures.",
      "# Debugging\nFind the root cause.");
    try {
      await fs.symlink(
        path.join(outside, ".agents", "skills", "systematic-debugging"),
        path.join(root, ".agents", "skills", "external-link"),
        "dir"
      );
    } catch {
      // Symlink creation can be restricted on Windows hosts; the oversize-file
      // boundary remains testable without elevated filesystem privileges.
    }

    const result = await loadRelevantSkills(root, "Debug a failing test.");
    assert.ok(!result.skills.some(skill => skill.slug === "oversized"));
    assert.ok(!result.skills.some(skill => skill.slug === "external-link"));
    assert.ok(result.skills.some(skill => skill.slug === "syntra-project-guardrails"));
  } finally {
    await removeWorkspace(root);
    await removeWorkspace(outside);
  }
});

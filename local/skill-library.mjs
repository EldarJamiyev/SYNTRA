import fs from "node:fs/promises";
import { constants } from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";

export const MAX_SKILL_CONTEXT_CHARS = 7200;

const MAX_DISCOVERED_SKILLS = 1024;
const MAX_DISCOVERY_BYTES = 8 * 1024;
const MAX_SELECTED_SOURCE_BYTES = 32 * 1024;
const MAX_SKILL_FILE_BYTES = 1024 * 1024;
const MAX_SELECTED_SKILLS = 3;
const MAX_SKILL_BODY_CHARS = 2600;
const ALWAYS_SELECTED = new Set([
  "syntra-project-guardrails",
  "verification-before-completion"
]);
const STOP_WORDS = new Set([
  "about", "after", "again", "also", "and", "any", "are", "because", "before",
  "being", "between", "both", "but", "can", "could", "does", "each", "from",
  "have", "into", "just", "make", "more", "most", "need", "only", "other",
  "our", "over", "please", "same", "should", "some", "such", "than", "that",
  "the", "their", "them", "then", "there", "these", "they", "this", "those",
  "through", "use", "using", "very", "want", "was", "were", "what", "when",
  "where", "which", "while", "with", "would", "your"
]);

function isWithin(root, target) {
  const relative = path.relative(root, target);
  return relative === "" || (
    relative !== ".."
    && !relative.startsWith(".." + path.sep)
    && !path.isAbsolute(relative)
  );
}

function sameFileIdentity(left, right) {
  return left.isFile() && right.isFile()
    && left.dev === right.dev
    && left.ino === right.ino
    && left.size === right.size
    && left.mtimeMs === right.mtimeMs
    && left.ctimeMs === right.ctimeMs;
}

async function realDirectory(directoryPath, expectedPath) {
  try {
    const info = await fs.lstat(directoryPath);
    if (!info.isDirectory() || info.isSymbolicLink()) return null;
    const resolved = await fs.realpath(directoryPath);
    if (resolved !== expectedPath) return null;
    return resolved;
  } catch {
    return null;
  }
}

async function readVerifiedPrefix(filePath, expectedParent, maxBytes) {
  let handle;
  try {
    const before = await fs.lstat(filePath);
    if (!before.isFile() || before.isSymbolicLink() || before.size > MAX_SKILL_FILE_BYTES) return null;
    if (await fs.realpath(path.dirname(filePath)) !== expectedParent) return null;

    const flags = constants.O_RDONLY | (constants.O_NOFOLLOW || 0);
    handle = await fs.open(filePath, flags);
    const opened = await handle.stat();
    if (!sameFileIdentity(before, opened)) return null;

    const length = Math.min(opened.size, maxBytes);
    const buffer = Buffer.alloc(length);
    let offset = 0;
    while (offset < length) {
      const { bytesRead } = await handle.read(buffer, offset, length - offset, offset);
      if (bytesRead <= 0) return null;
      offset += bytesRead;
    }

    const afterHandle = await handle.stat();
    const afterPath = await fs.lstat(filePath);
    if (!sameFileIdentity(opened, afterHandle) || !sameFileIdentity(opened, afterPath)) return null;
    if (await fs.realpath(path.dirname(filePath)) !== expectedParent) return null;

    return buffer.toString("utf8");
  } catch {
    return null;
  } finally {
    await handle?.close().catch(() => {});
  }
}

function cleanScalar(value) {
  return String(value || "")
    .trim()
    .replace(/^(['"])(.*)\1$/u, "$2")
    .replace(/\s+/gu, " ")
    .slice(0, 600);
}

function parseSkillDocument(source, fallbackName) {
  const lines = String(source || "").split(/\r?\n/u);
  let bodyStart = 0;
  let name = fallbackName;
  let description = "";

  if (lines[0]?.trim() === "---") {
    const closing = lines.findIndex((line, index) => index > 0 && line.trim() === "---");
    if (closing > 0 && closing <= 120) {
      for (let index = 1; index < closing; index += 1) {
        const match = lines[index].match(/^([A-Za-z][A-Za-z0-9_-]*):\s*(.*)$/u);
        if (!match) continue;
        const key = match[1].toLowerCase();
        const value = cleanScalar(match[2]);
        if (key === "name" && value) name = value;
        if (key === "description" && value) description = value;
      }
      bodyStart = closing + 1;
    }
  }

  const body = lines.slice(bodyStart).join("\n").trim();
  if (!description) {
    const paragraph = body.split(/\n\s*\n/u).find(part => part.trim() && !part.trim().startsWith("#"));
    description = cleanScalar(paragraph || "");
  }
  return {
    name: cleanScalar(name || fallbackName),
    description,
    body
  };
}

function termsOf(value) {
  return new Set(
    (String(value || "").toLowerCase().match(/[\p{L}\p{N}_-]{2,}/gu) || [])
      .filter(term => !STOP_WORDS.has(term))
      .slice(0, 160)
  );
}

function scoreSkill(goalTerms, skill) {
  const nameTerms = termsOf(skill.name + " " + skill.slug);
  const descriptionTerms = termsOf(skill.description);
  let score = 0;
  for (const term of goalTerms) {
    if (nameTerms.has(term)) score += 5;
    if (descriptionTerms.has(term)) score += 2;
  }
  return score;
}

async function discoverSkillDirectory(rootRealPath) {
  const agentsPath = path.join(rootRealPath, ".agents");
  const agentsReal = await realDirectory(agentsPath, agentsPath);
  if (!agentsReal) return null;

  const skillsPath = path.join(agentsReal, "skills");
  const skillsReal = await realDirectory(skillsPath, skillsPath);
  if (!skillsReal || !isWithin(rootRealPath, skillsReal)) return null;
  return skillsReal;
}

export async function loadRelevantSkills(workspaceRoot, goal) {
  const empty = (status, reason, discovered = 0) => ({
    status,
    reason,
    discovered,
    skills: [],
    context: ""
  });

  try {
    const rootRealPath = await fs.realpath(workspaceRoot);
    const skillsRoot = await discoverSkillDirectory(rootRealPath);
    if (!skillsRoot) {
      return empty("not_available", "No safe .agents/skills directory was found in the selected workspace.");
    }

    const entries = (await fs.readdir(skillsRoot, { withFileTypes: true }))
      .filter(entry => entry.isDirectory() && /^[a-z0-9][a-z0-9._-]{0,79}$/u.test(entry.name))
      .sort((left, right) => left.name.localeCompare(right.name))
      .slice(0, MAX_DISCOVERED_SKILLS);

    const discovered = [];
    for (const entry of entries) {
      const skillDirectory = path.join(skillsRoot, entry.name);
      const expectedDirectory = path.join(skillsRoot, entry.name);
      const realSkillDirectory = await realDirectory(skillDirectory, expectedDirectory);
      if (!realSkillDirectory || !isWithin(skillsRoot, realSkillDirectory)) continue;

      const skillPath = path.join(realSkillDirectory, "SKILL.md");
      const prefix = await readVerifiedPrefix(skillPath, realSkillDirectory, MAX_DISCOVERY_BYTES);
      if (prefix === null || !prefix.trim()) continue;

      const parsed = parseSkillDocument(prefix, entry.name);
      discovered.push({
        slug: entry.name,
        path: skillPath,
        name: parsed.name,
        description: parsed.description
      });
    }

    if (!discovered.length) {
      return empty("empty", "The local skill directory contains no readable SKILL.md files.", entries.length);
    }

    const goalTerms = termsOf(goal);
    const mandatory = discovered.filter(skill => ALWAYS_SELECTED.has(skill.slug));
    const ranked = discovered
      .filter(skill => !ALWAYS_SELECTED.has(skill.slug))
      .map(skill => ({ ...skill, score: scoreSkill(goalTerms, skill) }))
      .filter(skill => skill.score > 0)
      .sort((left, right) =>
        right.score - left.score || left.slug.localeCompare(right.slug)
      );

    const selectedMetadata = [...mandatory, ...ranked].slice(0, MAX_SELECTED_SKILLS);
    const selected = [];
    let remainingChars = MAX_SKILL_CONTEXT_CHARS;

    for (const item of selectedMetadata) {
      if (remainingChars <= 0) break;
      const source = await readVerifiedPrefix(item.path, path.dirname(item.path), MAX_SELECTED_SOURCE_BYTES);
      if (source === null) continue;

      const parsed = parseSkillDocument(source, item.slug);
      const excerpt = parsed.body.slice(0, MAX_SKILL_BODY_CHARS).trim();
      const heading = [
        "[SKILL: " + item.slug + "]",
        parsed.description ? "Purpose: " + parsed.description : ""
      ].filter(Boolean).join("\n");
      const section = [heading, excerpt].filter(Boolean).join("\n\n");
      const clipped = section.slice(0, remainingChars);
      if (!clipped.trim()) continue;

      const excerptSha256 = createHash("sha256").update(clipped, "utf8").digest("hex");
      selected.push({
        slug: item.slug,
        name: parsed.name,
        description: parsed.description,
        score: ALWAYS_SELECTED.has(item.slug) ? 1000 : item.score,
        excerptSha256,
        characters: clipped.length,
        content: clipped
      });
      remainingChars -= clipped.length;
    }

    const context = selected.map(skill => skill.content).join("\n\n---\n\n");
    return {
      status: selected.length ? "selected" : "no_match",
      reason: selected.length
        ? "Loaded a bounded set of local skill references."
        : "No relevant local skill references were selected.",
      discovered: discovered.length,
      skills: selected.map(({ content, ...metadata }) => metadata),
      context
    };
  } catch (error) {
    return empty(
      "unavailable",
      "Local skill retrieval was skipped because the selected workspace skill library could not be read safely: "
        + String(error instanceof Error ? error.message : error).slice(0, 240)
    );
  }
}

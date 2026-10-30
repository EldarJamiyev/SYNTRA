import fs from "node:fs/promises";
import path from "node:path";
import crypto from "node:crypto";
import {
  ensureWorkspaceStateDirectory,
  getWorkspaceStateDirectory,
  resolveWorkspace
} from "./workspace.mjs";

const MEMORY_FILENAME = "project-memory-v1.json";
const MEMORY_SCHEMA_VERSION = 1;
const MAX_NOTES = 200;
const MAX_MEMORY_BYTES = 1024 * 1024;
const MAX_TITLE_LENGTH = 120;
const MAX_CONTENT_LENGTH = 2400;
const MAX_TAGS = 8;
const NOTE_CATEGORIES = new Set([
  "architecture", "convention", "decision", "workflow", "testing", "other"
]);
const NOTE_ID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const TAG_PATTERN = /^[a-zA-Z0-9][a-zA-Z0-9 _.-]{0,31}$/;
const SECRET_PATTERNS = [
  /-----BEGIN [A-Z ]*PRIVATE KEY-----/i,
  /\b(?:password|passwd|secret|api[_-]?key|access[_-]?token|refresh[_-]?token|client[_-]?secret|private[_-]?key)\b\s*[:=]\s*["']?(?!your[_ -]|<|process\.env)[A-Za-z0-9_./+=-]{12,}/i,
  /\bBearer\s+[A-Za-z0-9._~-]{20,}/i,
  /\b(?:ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{20,}|xox[baprs]-[A-Za-z0-9-]{20,})\b/,
  /\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b/,
  /\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?):\/\/[^/\s:@]+:[^/\s@]+@/i
];

function hasCredentialPattern(value) {
  return SECRET_PATTERNS.some(pattern => pattern.test(value));
}

function validateNoteText(title, content) {
  if (typeof title !== "string" || !title.trim() || title.length > MAX_TITLE_LENGTH) {
    throw new Error("Note title must contain 1-120 characters.");
  }
  if (typeof content !== "string" || !content.trim() || content.length > MAX_CONTENT_LENGTH) {
    throw new Error("Note content must contain 1-2400 characters.");
  }
  if (hasCredentialPattern(title + "\n" + content)) {
    throw new Error("This note looks like it may contain a credential or secret. Store credentials in a secret manager, not project memory.");
  }
}

function normalizeCategory(value) {
  if (typeof value !== "string" || !NOTE_CATEGORIES.has(value)) {
    throw new Error("Choose a supported project-memory category.");
  }
  return value;
}

function normalizeTags(value = []) {
  if (!Array.isArray(value) || value.length > MAX_TAGS) {
    throw new Error("A project-memory note may have up to eight tags.");
  }
  const tags = [];
  const seen = new Set();
  for (const item of value) {
    if (typeof item !== "string" || !TAG_PATTERN.test(item.trim())) {
      throw new Error("Each note tag must contain 1-32 letters, numbers, spaces, dots, underscores, or hyphens.");
    }
    const tag = item.trim();
    const key = tag.toLowerCase();
    if (!seen.has(key)) {
      seen.add(key);
      tags.push(tag);
    }
  }
  return tags;
}

function validTimestamp(value) {
  return typeof value === "string" && Number.isFinite(Date.parse(value));
}

function validateStore(root, value) {
  if (!value || typeof value !== "object" || Array.isArray(value)
    || value.schemaVersion !== MEMORY_SCHEMA_VERSION
    || value.root !== root
    || !validTimestamp(value.updatedAt)
    || !Array.isArray(value.notes)
    || value.notes.length > MAX_NOTES) {
    throw new Error("The local project-memory file is invalid or belongs to a different workspace. It was not overwritten.");
  }
  const ids = new Set();
  const notes = [];
  for (const item of value.notes) {
    if (!item || typeof item !== "object" || Array.isArray(item)
      || !NOTE_ID_PATTERN.test(item.id || "")
      || ids.has(item.id)
      || !validTimestamp(item.createdAt) || !validTimestamp(item.updatedAt)
      || !NOTE_CATEGORIES.has(item.category) || item.owner !== "user"
      || !Array.isArray(item.tags) || item.tags.length > MAX_TAGS) {
      throw new Error("The local project-memory file contains an invalid note. It was not overwritten.");
    }
    validateNoteText(item.title, item.content);
    const tags = normalizeTags(item.tags);
    ids.add(item.id);
    notes.push({
      id: item.id, title: item.title, content: item.content, category: item.category,
      tags, createdAt: item.createdAt, updatedAt: item.updatedAt, owner: "user"
    });
  }
  return {
    schemaVersion: MEMORY_SCHEMA_VERSION, root, updatedAt: value.updatedAt, notes
  };
}

async function memoryPathFor(root, create) {
  const directory = create
    ? await ensureWorkspaceStateDirectory(root)
    : await getWorkspaceStateDirectory(root);
  return path.join(directory, MEMORY_FILENAME);
}

async function readMemory(root) {
  const target = await memoryPathFor(root, false);
  let stat;
  try {
    stat = await fs.lstat(target);
  } catch (error) {
    if (error && error.code === "ENOENT") {
      return { schemaVersion: MEMORY_SCHEMA_VERSION, root, updatedAt: null, notes: [] };
    }
    throw error;
  }
  if (stat.isSymbolicLink() || !stat.isFile()) {
    throw new Error("The project-memory file must be a regular file, not a symbolic link.");
  }
  if (stat.size > MAX_MEMORY_BYTES) {
    throw new Error("The project-memory file exceeds its 1 MB storage limit.");
  }
  try {
    return validateStore(root, JSON.parse(await fs.readFile(target, "utf8")));
  } catch (error) {
    if (error instanceof SyntaxError) {
      throw new Error("The local project-memory file contains invalid JSON. It was not overwritten.");
    }
    throw error;
  }
}

async function writeMemory(root, store) {
  const target = await memoryPathFor(root, true);
  const existing = await fs.lstat(target).catch(error => error?.code === "ENOENT" ? null : Promise.reject(error));
  if (existing && (existing.isSymbolicLink() || !existing.isFile())) {
    throw new Error("The project-memory file must be a regular file, not a symbolic link.");
  }
  const serialized = JSON.stringify(store, null, 2);
  if (Buffer.byteLength(serialized, "utf8") > MAX_MEMORY_BYTES) {
    throw new Error("Project memory exceeded its 1 MB storage limit.");
  }
  const temporary = target + "." + process.pid + "." + crypto.randomBytes(4).toString("hex") + ".tmp";
  await fs.writeFile(temporary, serialized + "\n", { encoding: "utf8", flag: "wx" });
  try {
    await fs.rename(temporary, target);
  } catch (error) {
    await fs.rm(temporary, { force: true });
    throw error;
  }
}

export async function listWorkspaceMemory(inputRoot) {
  const root = await resolveWorkspace(inputRoot);
  const store = await readMemory(root);
  return {
    root, total: store.notes.length, updatedAt: store.updatedAt,
    notes: store.notes.slice().sort((a, b) => b.updatedAt.localeCompare(a.updatedAt))
  };
}

export async function searchWorkspaceMemory(inputRoot, query, limit = 6) {
  const root = await resolveWorkspace(inputRoot);
  if (typeof query !== "string" || query.trim().length < 2 || query.length > 160) {
    throw new Error("Project-memory search query must contain 2-160 characters.");
  }
  if (!Number.isInteger(limit) || limit < 1 || limit > 12) {
    throw new Error("Project-memory search limit must be an integer from 1 to 12.");
  }
  const store = await readMemory(root);
  const normalized = query.trim().toLowerCase();
  const terms = [...new Set(normalized.match(/[a-z0-9_.-]{2,}/g) || [])].slice(0, 8);
  const ranked = [];
  for (const note of store.notes) {
    const title = note.title.toLowerCase();
    const content = note.content.toLowerCase();
    const tags = note.tags.join(" ").toLowerCase();
    let score = title === normalized ? 100 : title.includes(normalized) ? 70 : 0;
    for (const term of terms) {
      if (title.includes(term)) score += 12;
      if (tags.includes(term)) score += 8;
      if (content.includes(term)) score += 3;
    }
    if (score > 0) ranked.push({ note, score });
  }
  ranked.sort((a, b) => b.score - a.score || b.note.updatedAt.localeCompare(a.note.updatedAt));
  return {
    root, query, total: store.notes.length,
    results: ranked.slice(0, limit).map(({ note, score }) => ({
      id: note.id, title: note.title, content: note.content.slice(0, 1800),
      category: note.category, tags: note.tags, updatedAt: note.updatedAt,
      owner: "user", score
    }))
  };
}

export async function addWorkspaceMemory(inputRoot, input) {
  const root = await resolveWorkspace(inputRoot);
  if (!input || typeof input !== "object" || Array.isArray(input)) throw new Error("A project-memory note is required.");
  validateNoteText(input.title, input.content);
  const category = normalizeCategory(input.category);
  const tags = normalizeTags(input.tags);
  const store = await readMemory(root);
  if (store.notes.length >= MAX_NOTES) throw new Error("This workspace has reached the 200-note limit. Edit or delete an existing note first.");
  const now = new Date().toISOString();
  const note = {
    id: crypto.randomUUID(), title: input.title.trim(), content: input.content.trim(),
    category, tags, createdAt: now, updatedAt: now, owner: "user"
  };
  const next = { schemaVersion: MEMORY_SCHEMA_VERSION, root, updatedAt: now, notes: [...store.notes, note] };
  await writeMemory(root, next);
  return { note, total: next.notes.length, storedLocally: true };
}

export async function updateWorkspaceMemory(inputRoot, input) {
  const root = await resolveWorkspace(inputRoot);
  if (!input || typeof input !== "object" || Array.isArray(input)) throw new Error("A project-memory update is required.");
  const store = await readMemory(root);
  const index = store.notes.findIndex(note => note.id === input.id);
  if (index < 0) throw new Error("Project-memory note was not found.");
  const current = store.notes[index];
  if (input.expectedUpdatedAt !== current.updatedAt) {
    throw new Error("This note changed after it was opened. Reload it before saving to avoid overwriting newer edits.");
  }
  const title = input.title === undefined ? current.title : input.title;
  const content = input.content === undefined ? current.content : input.content;
  const category = input.category === undefined ? current.category : normalizeCategory(input.category);
  const tags = input.tags === undefined ? current.tags : normalizeTags(input.tags);
  validateNoteText(title, content);
  const now = new Date().toISOString();
  const updated = { ...current, title: title.trim(), content: content.trim(), category, tags, updatedAt: now };
  const notes = store.notes.slice();
  notes[index] = updated;
  await writeMemory(root, { schemaVersion: MEMORY_SCHEMA_VERSION, root, updatedAt: now, notes });
  return { note: updated, total: notes.length, storedLocally: true };
}

export async function deleteWorkspaceMemory(inputRoot, input) {
  const root = await resolveWorkspace(inputRoot);
  if (!input || typeof input !== "object" || Array.isArray(input)) throw new Error("A project-memory deletion is required.");
  const store = await readMemory(root);
  const current = store.notes.find(note => note.id === input.id);
  if (!current) throw new Error("Project-memory note was not found.");
  if (input.expectedUpdatedAt !== current.updatedAt) {
    throw new Error("This note changed after it was opened. Reload it before deleting it.");
  }
  const notes = store.notes.filter(note => note.id !== input.id);
  const now = new Date().toISOString();
  await writeMemory(root, { schemaVersion: MEMORY_SCHEMA_VERSION, root, updatedAt: now, notes });
  return { id: current.id, deleted: true, total: notes.length };
}

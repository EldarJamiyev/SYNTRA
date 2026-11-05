import fs from "node:fs/promises";
import path from "node:path";
import crypto from "node:crypto";
import {
  ensureWorkspaceStateDirectory,
  getWorkspaceStateDirectory,
  resolveWorkspace
} from "./workspace.mjs";

const MEMORY_FILENAME = "project-memory-v1.json";
const MEMORY_EMBEDDINGS_FILENAME = "project-memory-embeddings-v1.json";
const MAX_EMBEDDING_CACHE_BYTES = 8 * 1024 * 1024;
const MAX_SEMANTIC_INDEX_NOTES = 64;
const EMBEDDING_BATCH_SIZE = 32;
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

function memoryEmbeddingText(note) {
  return [
    note.title,
    "Category: " + note.category,
    "Tags: " + note.tags.join(", "),
    note.content
  ].join("\n");
}

function memoryTextHash(text) {
  return crypto.createHash("sha256").update(text, "utf8").digest("hex");
}

function validEmbeddingVector(value, dimension = null) {
  return Array.isArray(value)
    && value.length >= 8 && value.length <= 4096
    && (dimension === null || value.length === dimension)
    && value.every(item => Number.isFinite(item));
}

async function readMemoryEmbeddingCache(root, model) {
  const directory = path.dirname(await memoryPathFor(root, false));
  const target = path.join(directory, MEMORY_EMBEDDINGS_FILENAME);
  let stat;
  try { stat = await fs.lstat(target); }
  catch (error) {
    if (error?.code === "ENOENT") return { dimension: null, entries: [] };
    throw error;
  }
  if (stat.isSymbolicLink() || !stat.isFile()) {
    throw new Error("The project-memory embedding cache must be a regular file, not a symbolic link.");
  }
  if (stat.size > MAX_EMBEDDING_CACHE_BYTES) {
    throw new Error("The project-memory embedding cache exceeds its 8 MB limit.");
  }
  let value;
  try { value = JSON.parse(await fs.readFile(target, "utf8")); }
  catch (error) {
    throw new Error("The local project-memory embedding cache contains invalid JSON and was not overwritten.");
  }
  if (!value || typeof value !== "object" || Array.isArray(value)
    || value.schemaVersion !== 1 || value.root !== root
    || typeof value.model !== "string" || !Array.isArray(value.entries)
    || value.entries.length > MAX_NOTES) {
    throw new Error("The local project-memory embedding cache is invalid or belongs to another workspace.");
  }
  if (value.model !== model) return { dimension: null, entries: [] };
  const dimension = Number.isInteger(value.dimension) && value.dimension >= 8 && value.dimension <= 4096
    ? value.dimension : null;
  const ids = new Set();
  const entries = [];
  for (const item of value.entries) {
    if (!item || typeof item !== "object" || !NOTE_ID_PATTERN.test(item.noteId || "")
      || ids.has(item.noteId) || !/^[a-f0-9]{64}$/.test(item.textHash || "")
      || !validEmbeddingVector(item.embedding, dimension)) {
      throw new Error("The local project-memory embedding cache contains an invalid vector entry.");
    }
    ids.add(item.noteId);
    entries.push({ noteId: item.noteId, textHash: item.textHash, embedding: item.embedding });
  }
  return { dimension, entries };
}

async function writeMemoryEmbeddingCache(root, model, dimension, entries) {
  const memoryPath = await memoryPathFor(root, true);
  const target = path.join(path.dirname(memoryPath), MEMORY_EMBEDDINGS_FILENAME);
  const current = await fs.lstat(target).catch(error => error?.code === "ENOENT" ? null : Promise.reject(error));
  if (current && (current.isSymbolicLink() || !current.isFile())) {
    throw new Error("The project-memory embedding cache must be a regular file, not a symbolic link.");
  }
  const serialized = JSON.stringify({
    schemaVersion: 1, root, model, dimension,
    updatedAt: new Date().toISOString(), entries
  });
  if (Buffer.byteLength(serialized, "utf8") > MAX_EMBEDDING_CACHE_BYTES) {
    throw new Error("The local project-memory embedding cache exceeded its 8 MB limit.");
  }
  const temporary = target + "." + process.pid + "." + crypto.randomBytes(4).toString("hex") + ".tmp";
  await fs.writeFile(temporary, serialized + "\n", { encoding: "utf8", flag: "wx" });
  try { await fs.rename(temporary, target); }
  catch (error) {
    await fs.rm(temporary, { force: true });
    throw error;
  }
}

const NATIVE_EMBEDDING_MODEL = "syntra-native";

async function requestLocalEmbeddings(fetchImpl, input) {
  if (!Array.isArray(input) || input.length < 1 || input.length > EMBEDDING_BATCH_SIZE) {
    throw new Error("Native embedding batches must contain 1-32 texts.");
  }
  const endpoint = `http://127.0.0.1:${Number(process.env.SYNTRA_AI_PORT || 3002)}/model/embed`;
  const response = await fetchImpl(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ texts: input, max_length: 256, device: "auto" }),
    signal: AbortSignal.timeout(4000)
  });
  if (!response.ok) throw new Error("SYNTRA native embedding service returned HTTP " + response.status + ".");
  const payload = await response.json();
  if (!payload || !Array.isArray(payload.embeddings) || payload.embeddings.length !== input.length) {
    throw new Error("SYNTRA native runtime returned an invalid embedding response.");
  }
  const dimension = payload.embeddings[0]?.length;
  if (!Number.isInteger(dimension) || dimension < 8 || dimension > 4096
    || !payload.embeddings.every(vector => validEmbeddingVector(vector, dimension))) {
    throw new Error("SYNTRA native runtime returned an invalid or inconsistent embedding dimension.");
  }
  return { dimension, embeddings: payload.embeddings };
}

function cosineSimilarity(left, right) {
  if (!Array.isArray(left) || !Array.isArray(right) || left.length !== right.length || !left.length) return null;
  let dot = 0, leftNorm = 0, rightNorm = 0;
  for (let index = 0; index < left.length; index += 1) {
    dot += left[index] * right[index];
    leftNorm += left[index] * left[index];
    rightNorm += right[index] * right[index];
  }
  if (leftNorm <= 0 || rightNorm <= 0) return null;
  return dot / Math.sqrt(leftNorm * rightNorm);
}

async function searchMemorySemantically(root, store, query, ranked, options = {}) {
  const model = String(options.embedModel ?? NATIVE_EMBEDDING_MODEL).trim();
  if (model !== NATIVE_EMBEDDING_MODEL) throw new Error("SYNTRA project memory accepts only native checkpoint embeddings.");
  const fetchImpl = options.fetchImpl ?? globalThis.fetch;
  if (typeof fetchImpl !== "function") throw new Error("SYNTRA native embeddings are unavailable in this runtime.");
  const cache = await readMemoryEmbeddingCache(root, model);
  const noteById = new Map(store.notes.map(note => [note.id, note]));
  const validEntries = cache.entries.filter(entry => {
    const note = noteById.get(entry.noteId);
    return note && entry.textHash === memoryTextHash(memoryEmbeddingText(note));
  });
  const vectorById = new Map(validEntries.map(entry => [entry.noteId, entry.embedding]));
  const recent = store.notes.slice().sort((a, b) => b.updatedAt.localeCompare(a.updatedAt))
    .slice(0, MAX_SEMANTIC_INDEX_NOTES);
  const lexicalCandidates = ranked.slice(0, 16).map(item => item.note);
  const selected = [...new Map([...lexicalCandidates, ...recent].map(note => [note.id, note])).values()]
    .slice(0, MAX_SEMANTIC_INDEX_NOTES);
  const missing = selected.filter(note => !vectorById.has(note.id));
  let dimension = cache.dimension;
  const inputs = [...missing.map(note => memoryEmbeddingText(note)), query];
  const vectors = [];
  for (let offset = 0; offset < inputs.length; offset += EMBEDDING_BATCH_SIZE) {
    const batch = await requestLocalEmbeddings(
      fetchImpl, inputs.slice(offset, offset + EMBEDDING_BATCH_SIZE)
    );
    if (dimension !== null && dimension !== batch.dimension) {
      throw new Error("The local embedding model returned a different vector size than the cached memory index.");
    }
    dimension = batch.dimension;
    vectors.push(...batch.embeddings);
  }
  if (vectors.length !== inputs.length) throw new Error("Local embedding generation returned an incomplete vector set.");
  const newEntries = missing.map((note, index) => ({
    noteId: note.id,
    textHash: memoryTextHash(memoryEmbeddingText(note)),
    embedding: vectors[index]
  }));
  for (const entry of newEntries) vectorById.set(entry.noteId, entry.embedding);
  const currentEntries = store.notes
    .filter(note => vectorById.has(note.id))
    .map(note => ({
      noteId: note.id,
      textHash: memoryTextHash(memoryEmbeddingText(note)),
      embedding: vectorById.get(note.id)
    }));
  await writeMemoryEmbeddingCache(root, model, dimension, currentEntries);
  const queryEmbedding = vectors[vectors.length - 1];
  const lexicalById = new Map(ranked.map(item => [item.note.id, item.score]));
  const maxLexical = Math.max(1, ...ranked.map(item => item.score));
  const semanticRanked = [];
  for (const note of store.notes) {
    const lexicalScore = lexicalById.get(note.id) || 0;
    const lexicalNormalized = lexicalScore / maxLexical;
    const vector = vectorById.get(note.id);
    const cosine = vector ? cosineSimilarity(queryEmbedding, vector) : null;
    const semanticScore = cosine === null ? 0 : Math.max(0, cosine);
    const score = vector
      ? (semanticScore * 0.65) + (lexicalNormalized * 0.35)
      : lexicalNormalized * 0.35;
    if (score > 0) semanticRanked.push({
      note, score, semanticScore, lexicalScore
    });
  }
  semanticRanked.sort((left, right) =>
    right.score - left.score || right.note.updatedAt.localeCompare(left.note.updatedAt)
  );
  return {
    retrievalMode: "hybrid_semantic",
    embeddingModel: model,
    semanticCoverage: currentEntries.length,
    results: semanticRanked
  };
}

export async function searchWorkspaceMemory(inputRoot, query, limit = 6, options = {}) {
  const root = await resolveWorkspace(inputRoot);
  if (typeof query !== "string" || query.trim().length < 2 || query.length > 160) throw new Error("Project-memory search query must contain 2-160 characters.");
  if (!Number.isInteger(limit) || limit < 1 || limit > 12) throw new Error("Project-memory search limit must be an integer from 1 to 12.");
  const store = await readMemory(root);
  const normalized = query.trim().toLowerCase();
  const terms = [...new Set(normalized.match(/[a-z0-9_.-]{2,}/g) || [])].slice(0, 8);
  const documents = store.notes.map(note => ({
    note,
    titleTokens: note.title.toLowerCase().match(/[a-z0-9_.-]{2,}/g) || [],
    tagTokens: note.tags.join(" ").toLowerCase().match(/[a-z0-9_.-]{2,}/g) || [],
    bodyTokens: note.content.toLowerCase().match(/[a-z0-9_.-]{2,}/g) || []
  }));
  const df = new Map(terms.map(term => [term, documents.filter(doc => doc.titleTokens.includes(term) || doc.tagTokens.includes(term) || doc.bodyTokens.includes(term)).length]));
  const averageLength = documents.length ? documents.reduce((sum, doc) => sum + doc.bodyTokens.length, 0) / documents.length : 0;
  const count = (tokens, term) => tokens.filter(token => token === term).length;
  const ranked = [];
  for (const doc of documents) {
    let score = doc.note.title.toLowerCase() === normalized ? 8 : doc.note.title.toLowerCase().includes(normalized) ? 4 : 0;
    const lengthRatio = averageLength > 0 ? doc.bodyTokens.length / averageLength : 1;
    for (const term of terms) {
      const titleTf = count(doc.titleTokens, term);
      const tagTf = count(doc.tagTokens, term);
      const bodyTf = count(doc.bodyTokens, term);
      const frequency = df.get(term) || 0;
      const inverseFrequency = Math.log(1 + (documents.length - frequency + 0.5) / (frequency + 0.5));
      const bodyScore = bodyTf * 2.2 / (bodyTf + 1.2 * (0.25 + 0.75 * lengthRatio));
      score += inverseFrequency * (titleTf * 3 + tagTf * 2 + bodyScore);
    }
    if (score > 0) ranked.push({ note: doc.note, score });
  }
  ranked.sort((left, right) => right.score - left.score || right.note.updatedAt.localeCompare(left.note.updatedAt));
  const keywordResults = ranked.slice(0, limit).map(({ note, score }) => ({
    id: note.id, title: note.title, content: note.content.slice(0, 1800),
    category: note.category, tags: note.tags, updatedAt: note.updatedAt, owner: "user", score
  }));
  const embeddingModel = String(options.embedModel ?? NATIVE_EMBEDDING_MODEL).trim();
  if (embeddingModel !== NATIVE_EMBEDDING_MODEL) {
    throw new Error("SYNTRA project memory uses only embeddings produced by its own trained checkpoint.");
  }
  try {
    const semantic = await searchMemorySemantically(root, store, normalized, ranked, options);
    return {
      root, query, total: store.notes.length,
      retrievalMode: semantic.retrievalMode,
      embeddingModel: semantic.embeddingModel,
      semanticCoverage: semantic.semanticCoverage,
      results: semantic.results.slice(0, limit).map(({ note, score, semanticScore, lexicalScore }) => ({
        id: note.id, title: note.title, content: note.content.slice(0, 1800),
        category: note.category, tags: note.tags, updatedAt: note.updatedAt, owner: "user",
        score: Number(score.toFixed(6)),
        semanticScore: Number(semanticScore.toFixed(6)),
        keywordScore: Number(lexicalScore.toFixed(4))
      }))
    };
  } catch (error) {
    return {
      root, query, total: store.notes.length,
      retrievalMode: "bm25_fallback",
      semanticError: (error instanceof Error ? error.message : String(error)).slice(0, 240),
      results: keywordResults
    };
  }
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

export function synthesizeMemoryNoteFromAgentRun(run) {
  if (!run || typeof run !== "object") return null;
  if (run.verification?.passed !== true && run.status !== "verified_proposals_ready") return null;
  const files = (Array.isArray(run.changedFiles) ? run.changedFiles : [])
    .map(item => String(item?.path || "").trim())
    .filter(Boolean);
  if (!files.length) return null;
  const primaryFile = path.posix.basename(files[0].replace(/\\/g, "/"));
  const goalSnippet = String(run.goal || "Verified change").replace(/\s+/g, " ").trim().slice(0, 72);
  const title = `Verified pattern: ${primaryFile} — ${goalSnippet}`.slice(0, MAX_TITLE_LENGTH).trim();
  const reasons = (Array.isArray(run.changedFiles) ? run.changedFiles : [])
    .map(item => item?.reason ? `${item.path}: ${item.reason}` : item?.path)
    .filter(Boolean)
    .slice(0, 6);
  const checkNames = (Array.isArray(run.verification?.checks) ? run.verification.checks : [])
    .filter(check => check?.passed)
    .map(check => check.task)
    .filter(Boolean);
  const contentLines = [
    `Goal: ${String(run.goal || "").trim().slice(0, 360)}`,
    `Modified files (${files.length}): ${files.slice(0, 8).join(", ")}`,
    ...(reasons.length ? [`Implementation notes: ${reasons.join(" | ").slice(0, 800)}`] : []),
    ...(checkNames.length ? [`Verified checks: ${checkNames.join(", ")}`] : [])
  ];
  const content = contentLines.join("\n").slice(0, MAX_CONTENT_LENGTH).trim();
  if (!title || !content || hasCredentialPattern(title + "\n" + content)) return null;
  const tags = [];
  for (const file of files.slice(0, 5)) {
    const base = path.posix.basename(file.replace(/\\/g, "/")).replace(/[^a-zA-Z0-9_.-]/g, "").slice(0, 28);
    if (base && TAG_PATTERN.test(base) && !tags.includes(base)) tags.push(base);
  }
  if (!tags.includes("verified") && tags.length < MAX_TAGS) tags.push("verified");
  return {
    title,
    content,
    category: "workflow",
    tags: tags.slice(0, MAX_TAGS)
  };
}

export async function recordVerifiedAgentMemory(inputRoot, run) {
  const candidate = synthesizeMemoryNoteFromAgentRun(run);
  if (!candidate) return null;
  const root = await resolveWorkspace(inputRoot);
  const store = await readMemory(root);
  const existing = store.notes.find(note => note.title.toLowerCase() === candidate.title.toLowerCase());
  if (existing) {
    return updateWorkspaceMemory(root, {
      id: existing.id,
      expectedUpdatedAt: existing.updatedAt,
      title: candidate.title,
      content: candidate.content,
      category: candidate.category,
      tags: candidate.tags
    });
  }
  if (store.notes.length >= MAX_NOTES) return null;
  return addWorkspaceMemory(root, candidate);
}


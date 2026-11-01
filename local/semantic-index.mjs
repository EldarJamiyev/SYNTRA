import fs from "node:fs/promises";
import path from "node:path";
import crypto from "node:crypto";
import {
  ensureWorkspaceStateDirectory, getWorkspaceStateDirectory, isWorkspacePathExcluded,
  listWorkspaceIndexCandidates, readWorkspaceFile, resolveWorkspace
} from "./workspace.mjs";

const INDEX_NAME = "semantic-index-v1.json";
const MAX_INDEX_BYTES = 8 * 1024 * 1024;
const MAX_FILES = 500, MAX_CHUNKS = 600, MAX_CHUNK_CHARS = 2400;
const CHUNK_LINES = 36, CHUNK_OVERLAP = 6;
const DEFAULT_EMBEDDING_MODEL = "embeddinggemma";
const OLLAMA_BASE = process.env.SYNTRA_OLLAMA_BASE_URL || "http://127.0.0.1:11434";

function cosineSimilarity(a, b) {
  if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length || !a.length) return -1;
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i += 1) {
    dot += a[i] * b[i]; normA += a[i] * a[i]; normB += b[i] * b[i];
  }
  return normA > 0 && normB > 0 ? dot / Math.sqrt(normA * normB) : -1;
}
function splitIntoChunks(relativePath, content) {
  const lines = content.split(/\r?\n/), chunks = [];
  const step = Math.max(1, CHUNK_LINES - CHUNK_OVERLAP);
  for (let start = 0; start < lines.length; start += step) {
    const end = Math.min(lines.length, start + CHUNK_LINES);
    const text = lines.slice(start, end).join("\n").slice(0, MAX_CHUNK_CHARS);
    if (text.trim().length >= 40) chunks.push({ path: relativePath, startLine: start + 1, endLine: end, text });
    if (end >= lines.length || chunks.length >= MAX_CHUNKS) break;
  }
  return chunks;
}
async function readIndex(root) {
  const target = path.join(await getWorkspaceStateDirectory(root), INDEX_NAME);
  let stat;
  try { stat = await fs.lstat(target); }
  catch (error) { if (error?.code === "ENOENT") return null; throw error; }
  if (stat.isSymbolicLink() || !stat.isFile()) throw new Error("Semantic index must be a regular file, not a symbolic link.");
  if (stat.size > MAX_INDEX_BYTES) return null;
  try {
    const data = JSON.parse(await fs.readFile(target, "utf8"));
    if (data?.schemaVersion !== 1 || data.root !== root || typeof data.embeddingModel !== "string"
      || !Number.isInteger(data.dimensions) || data.dimensions < 8 || data.dimensions > 4096
      || !Array.isArray(data.chunks) || data.chunks.length > MAX_CHUNKS) return null;
    const valid = data.chunks.filter(item => item && typeof item.path === "string"
      && Number.isInteger(item.startLine) && item.startLine > 0
      && Number.isInteger(item.endLine) && item.endLine >= item.startLine
      && typeof item.sha256 === "string" && /^[a-f0-9]{64}$/i.test(item.sha256)
      && Array.isArray(item.vector) && item.vector.length === data.dimensions
      && item.vector.every(value => Number.isFinite(value) && Math.abs(value) <= 100));
    return valid.length === data.chunks.length ? { ...data, chunks: valid } : null;
  } catch { return null; }
}
async function writeIndex(root, index) {
  const target = path.join(await ensureWorkspaceStateDirectory(root), INDEX_NAME);
  const serialized = JSON.stringify(index);
  if (Buffer.byteLength(serialized, "utf8") > MAX_INDEX_BYTES) throw new Error("Semantic index exceeded its 8 MB storage limit.");
  const temporary = target + "." + process.pid + "." + crypto.randomBytes(4).toString("hex") + ".tmp";
  await fs.writeFile(temporary, serialized + "\n", { encoding: "utf8", flag: "wx" });
  try { await fs.rename(temporary, target); }
  catch (error) { await fs.rm(temporary, { force: true }); throw error; }
}
async function embed(texts, model, fetchImpl, signal) {
  if (!texts.length) return [];
  const timeout = AbortSignal.timeout(120000);
  const response = await fetchImpl(OLLAMA_BASE + "/api/embed", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model, input: texts, truncate: true }),
    signal: signal ? AbortSignal.any([signal, timeout]) : timeout
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(String(payload.error || "Ollama embeddings request failed with HTTP " + response.status));
  if (!Array.isArray(payload.embeddings) || payload.embeddings.length !== texts.length) throw new Error("Ollama returned an invalid embedding batch.");
  const dims = payload.embeddings[0]?.length;
  if (!Number.isInteger(dims) || dims < 8 || dims > 4096
    || payload.embeddings.some(v => !Array.isArray(v) || v.length !== dims || v.some(n => !Number.isFinite(n)))) {
    throw new Error("Ollama returned inconsistent or invalid embedding dimensions.");
  }
  return payload.embeddings.map(v => v.map(n => Number(Number(n).toFixed(5))));
}

export async function getSemanticIndexStatus(inputRoot) {
  const root = await resolveWorkspace(inputRoot), index = await readIndex(root);
  return index
    ? { available: true, root, embeddingModel: index.embeddingModel, indexedAt: index.indexedAt, chunkCount: index.chunks.length, dimensions: index.dimensions }
    : { available: false, root, embeddingModel: process.env.SYNTRA_EMBEDDING_MODEL || DEFAULT_EMBEDDING_MODEL, chunkCount: 0, reason: "not_built_or_invalid" };
}
export async function refreshWorkspaceSemanticIndex(inputRoot, {
  fetchImpl = globalThis.fetch, model = process.env.SYNTRA_EMBEDDING_MODEL || DEFAULT_EMBEDDING_MODEL, signal
} = {}) {
  const root = await resolveWorkspace(inputRoot);
  if (typeof model !== "string" || !/^[a-zA-Z0-9._:-]{2,100}$/.test(model)) throw new Error("Invalid SYNTRA embedding model name.");
  const previous = await readIndex(root), oldByPath = new Map();
  if (previous?.embeddingModel === model) for (const item of previous.chunks) {
    const list = oldByPath.get(item.path) || []; list.push(item); oldByPath.set(item.path, list);
  }
  const candidates = await listWorkspaceIndexCandidates(root);
  const pending = [], chunks = [];
  let scannedFiles = 0, reusedChunks = 0, skippedFiles = 0, scannedBytes = 0;
  for (const candidate of candidates.files.slice(0, MAX_FILES)) {
    if (await isWorkspacePathExcluded(root, candidate.path)) { skippedFiles += 1; continue; }
    let source;
    try { source = await readWorkspaceFile(root, candidate.path); }
    catch { skippedFiles += 1; continue; }
    if (scannedBytes + source.bytes > 20 * 1024 * 1024) { skippedFiles += 1; continue; }
    scannedBytes += source.bytes; scannedFiles += 1;
    const old = oldByPath.get(candidate.path) || [];
    if (old.length && old.every(item => item.sha256 === source.sha256)) { chunks.push(...old); reusedChunks += old.length; continue; }
    for (const chunk of splitIntoChunks(candidate.path, source.content)) {
      pending.push({ ...chunk, sha256: source.sha256 });
      if (chunks.length + pending.length >= MAX_CHUNKS) break;
    }
    if (chunks.length + pending.length >= MAX_CHUNKS) break;
  }
  for (let offset = 0; offset < pending.length; offset += 24) {
    const batch = pending.slice(offset, offset + 24);
    const vectors = await embed(batch.map(item => item.text), model, fetchImpl, signal);
    for (let i = 0; i < batch.length; i += 1) {
      const { text: _sourceTextNotPersisted, ...metadata } = batch[i];
      chunks.push({ ...metadata, vector: vectors[i] });
    }
  }
  const dimensions = chunks[0]?.vector?.length || 0;
  if (chunks.some(item => item.vector.length !== dimensions)) throw new Error("Semantic index contains inconsistent vector dimensions.");
  const index = { schemaVersion: 1, root, embeddingModel: model, dimensions, indexedAt: new Date().toISOString(),
    scannedFiles, scannedBytes, truncated: candidates.truncated || candidates.files.length > MAX_FILES || chunks.length >= MAX_CHUNKS, chunks };
  await writeIndex(root, index);
  return { available: true, root, embeddingModel: model, indexedAt: index.indexedAt, chunkCount: chunks.length,
    dimensions, scannedFiles, scannedBytes, reusedChunks, skippedFiles, truncated: index.truncated };
}
export async function searchWorkspaceSemantic(inputRoot, query, limit = 8, {
  fetchImpl = globalThis.fetch, model = process.env.SYNTRA_EMBEDDING_MODEL || DEFAULT_EMBEDDING_MODEL, signal
} = {}) {
  const root = await resolveWorkspace(inputRoot);
  if (typeof query !== "string" || query.trim().length < 2 || query.length > 300) throw new Error("Semantic query must contain 2-300 characters.");
  if (!Number.isInteger(limit) || limit < 1 || limit > 20) throw new Error("Semantic result limit must be an integer from 1 to 20.");
  const index = await readIndex(root);
  if (!index || index.embeddingModel !== model) return { available: false, root, query, results: [], reason: "semantic_index_missing_or_embedding_model_changed" };
  const [queryVector] = await embed([query.trim()], model, fetchImpl, signal);
  if (queryVector.length !== index.dimensions) throw new Error("Query embedding dimensions do not match the saved index. Refresh it with the current embedding model.");
  const ranked = index.chunks.map(item => ({ ...item, score: cosineSimilarity(queryVector, item.vector) }))
    .filter(item => item.score >= 0).sort((a, b) => b.score - a.score || a.path.localeCompare(b.path) || a.startLine - b.startLine);
  const results = [], checked = new Map();
  for (const item of ranked) {
    if (results.length >= limit) break;
    if (checked.has(item.path) && !checked.get(item.path)) continue;
    if (!checked.has(item.path)) {
      try {
        if (await isWorkspacePathExcluded(root, item.path)) { checked.set(item.path, false); continue; }
        const current = await readWorkspaceFile(root, item.path);
        checked.set(item.path, current.sha256 === item.sha256 ? current : false);
      } catch { checked.set(item.path, false); }
    }
    const source = checked.get(item.path);
    if (!source) continue;
    const excerpt = source.content.split(/\r?\n/).slice(item.startLine - 1, item.endLine).join("\n").slice(0, MAX_CHUNK_CHARS);
    results.push({ path: item.path, startLine: item.startLine, endLine: item.endLine, score: Number(item.score.toFixed(5)), sha256: item.sha256, excerpt });
  }
  return { available: true, root, query: query.trim(), embeddingModel: model, results, staleResults: ranked.length - results.length };
}

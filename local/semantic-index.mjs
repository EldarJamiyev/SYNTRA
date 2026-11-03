import fs from "node:fs/promises";
import path from "node:path";
import crypto from "node:crypto";
import {
  ensureWorkspaceStateDirectory,
  getWorkspaceStateDirectory,
  isWorkspacePathExcluded,
  listWorkspaceIndexCandidates,
  readWorkspaceFile,
  resolveWorkspace
} from "./workspace.mjs";

const INDEX_NAME = "semantic-index-v2.json";
const INDEX_METHOD = "syntra-local-bm25-v1";
const MAX_INDEX_BYTES = 8 * 1024 * 1024;
const MAX_FILES = 500;
const MAX_CHUNKS = 600;
const MAX_CHUNK_CHARS = 2400;
const CHUNK_LINES = 36;
const CHUNK_OVERLAP = 6;
const STOP_WORDS = new Set([
  "about", "after", "again", "also", "and", "are", "because", "been", "before",
  "being", "between", "both", "but", "can", "could", "did", "does", "doing",
  "down", "each", "for", "from", "get", "got", "had", "has", "have", "here",
  "how", "into", "its", "just", "like", "made", "make", "many", "more", "most",
  "not", "now", "only", "other", "our", "out", "over", "same", "should", "some",
  "such", "than", "that", "the", "their", "them", "then", "there", "these",
  "they", "this", "those", "through", "too", "under", "use", "used", "using",
  "very", "was", "were", "what", "when", "where", "which", "while", "who",
  "will", "with", "would", "you", "your"
]);

function tokenizeTerms(text) {
  const camelSplit = String(text).replace(/([a-z0-9])([A-Z])/g, "$1 $2");
  const values = camelSplit.toLocaleLowerCase("en").match(/[\p{L}\p{N}]{2,}/gu) || [];
  return values.filter(term => !STOP_WORDS.has(term));
}

function termFrequencies(text) {
  const counts = new Map();
  for (const term of tokenizeTerms(text)) counts.set(term, (counts.get(term) || 0) + 1);
  return [...counts.entries()].sort((a, b) => a[0].localeCompare(b[0]));
}

function splitIntoChunks(relativePath, content) {
  const lines = content.split(/\r?\n/);
  const chunks = [];
  const step = Math.max(1, CHUNK_LINES - CHUNK_OVERLAP);
  for (let start = 0; start < lines.length; start += step) {
    const end = Math.min(lines.length, start + CHUNK_LINES);
    const text = lines.slice(start, end).join("\n").slice(0, MAX_CHUNK_CHARS);
    if (text.trim().length >= 40) {
      chunks.push({
        path: relativePath,
        startLine: start + 1,
        endLine: end,
        text
      });
    }
    if (end >= lines.length || chunks.length >= MAX_CHUNKS) break;
  }
  return chunks;
}

function validTermFrequencies(value) {
  if (!Array.isArray(value) || value.length > 1500) return false;
  let previous = "";
  for (const entry of value) {
    if (!Array.isArray(entry) || entry.length !== 2
      || typeof entry[0] !== "string" || entry[0].length < 2 || entry[0].length > 80
      || !Number.isInteger(entry[1]) || entry[1] < 1 || entry[1] > 10000
      || entry[0] <= previous) return false;
    previous = entry[0];
  }
  return true;
}

async function readIndex(root) {
  const target = path.join(await getWorkspaceStateDirectory(root), INDEX_NAME);
  let stat;
  try { stat = await fs.lstat(target); }
  catch (error) { if (error?.code === "ENOENT") return null; throw error; }
  if (stat.isSymbolicLink() || !stat.isFile()) {
    throw new Error("Workspace retrieval index must be a regular file, not a symbolic link.");
  }
  if (stat.size > MAX_INDEX_BYTES) return null;
  try {
    const data = JSON.parse(await fs.readFile(target, "utf8"));
    if (data?.schemaVersion !== 2 || data.root !== root || data.indexMethod !== INDEX_METHOD
      || !Array.isArray(data.chunks) || data.chunks.length > MAX_CHUNKS) return null;
    const valid = data.chunks.filter(item => item && typeof item.path === "string"
      && Number.isInteger(item.startLine) && item.startLine > 0
      && Number.isInteger(item.endLine) && item.endLine >= item.startLine
      && typeof item.sha256 === "string" && /^[a-f0-9]{64}$/i.test(item.sha256)
      && validTermFrequencies(item.terms)
      && Number.isInteger(item.termCount) && item.termCount >= 0);
    return valid.length === data.chunks.length ? { ...data, chunks: valid } : null;
  } catch {
    return null;
  }
}

async function writeIndex(root, index) {
  const target = path.join(await ensureWorkspaceStateDirectory(root), INDEX_NAME);
  const serialized = JSON.stringify(index);
  if (Buffer.byteLength(serialized, "utf8") > MAX_INDEX_BYTES) {
    throw new Error("Local workspace retrieval index exceeded its 8 MB storage limit.");
  }
  const temporary = target + "." + process.pid + "." + crypto.randomBytes(4).toString("hex") + ".tmp";
  try {
    await fs.writeFile(temporary, serialized + "\n", { encoding: "utf8", flag: "wx" });
    await fs.rename(temporary, target);
  } catch (error) {
    await fs.rm(temporary, { force: true }).catch(() => {});
    throw error;
  }
}

export async function getSemanticIndexStatus(inputRoot) {
  const root = await resolveWorkspace(inputRoot);
  const index = await readIndex(root);
  return index
    ? {
        available: true,
        root,
        indexMethod: index.indexMethod,
        indexedAt: index.indexedAt,
        chunkCount: index.chunks.length,
        dimensions: null
      }
    : {
        available: false,
        root,
        indexMethod: INDEX_METHOD,
        chunkCount: 0,
        reason: "not_built_or_invalid"
      };
}

export async function refreshWorkspaceSemanticIndex(inputRoot, { signal } = {}) {
  const root = await resolveWorkspace(inputRoot);
  const previous = await readIndex(root);
  const oldByPath = new Map();

  if (previous) {
    for (const item of previous.chunks) {
      const list = oldByPath.get(item.path) || [];
      list.push(item);
      oldByPath.set(item.path, list);
    }
  }

  const candidates = await listWorkspaceIndexCandidates(root);
  const chunks = [];
  let scannedFiles = 0;
  let reusedChunks = 0;
  let skippedFiles = 0;
  let scannedBytes = 0;

  for (const candidate of candidates.files.slice(0, MAX_FILES)) {
    if (signal?.aborted) throw new Error("Workspace indexing was cancelled.");
    if (await isWorkspacePathExcluded(root, candidate.path)) {
      skippedFiles += 1;
      continue;
    }
    let source;
    try { source = await readWorkspaceFile(root, candidate.path); }
    catch { skippedFiles += 1; continue; }
    if (scannedBytes + source.bytes > 20 * 1024 * 1024) {
      skippedFiles += 1;
      continue;
    }

    scannedBytes += source.bytes;
    scannedFiles += 1;
    const old = oldByPath.get(candidate.path) || [];
    if (old.length && old.every(item => item.sha256 === source.sha256)) {
      chunks.push(...old);
      reusedChunks += old.length;
      continue;
    }

    for (const chunk of splitIntoChunks(candidate.path, source.content)) {
      const terms = termFrequencies(chunk.text);
      chunks.push({
        path: chunk.path,
        startLine: chunk.startLine,
        endLine: chunk.endLine,
        sha256: source.sha256,
        terms,
        termCount: terms.reduce((sum, entry) => sum + entry[1], 0)
      });
      if (chunks.length >= MAX_CHUNKS) break;
    }
    if (chunks.length >= MAX_CHUNKS) break;
  }

  const index = {
    schemaVersion: 2,
    root,
    indexMethod: INDEX_METHOD,
    indexedAt: new Date().toISOString(),
    scannedFiles,
    scannedBytes,
    truncated: candidates.truncated || candidates.files.length > MAX_FILES || chunks.length >= MAX_CHUNKS,
    chunks
  };
  await writeIndex(root, index);
  return {
    available: true,
    root,
    indexMethod: INDEX_METHOD,
    indexedAt: index.indexedAt,
    chunkCount: chunks.length,
    dimensions: null,
    scannedFiles,
    scannedBytes,
    reusedChunks,
    skippedFiles,
    truncated: index.truncated
  };
}

function scoreChunks(chunks, queryTerms) {
  if (!chunks.length || !queryTerms.length) return [];
  const documentFrequency = new Map();
  for (const chunk of chunks) {
    for (const [term] of chunk.terms) {
      documentFrequency.set(term, (documentFrequency.get(term) || 0) + 1);
    }
  }
  const averageLength = chunks.reduce((sum, item) => sum + item.termCount, 0) / chunks.length || 1;
  const k1 = 1.2;
  const b = 0.75;
  const uniqueQueryTerms = [...new Set(queryTerms)];

  return chunks.map(chunk => {
    const frequencies = new Map(chunk.terms);
    let score = 0;
    for (const term of uniqueQueryTerms) {
      const frequency = frequencies.get(term) || 0;
      if (!frequency) continue;
      const documentCount = documentFrequency.get(term) || 0;
      const inverseDocumentFrequency = Math.log(
        1 + (chunks.length - documentCount + 0.5) / (documentCount + 0.5)
      );
      const denominator = frequency + k1 * (
        1 - b + b * chunk.termCount / averageLength
      );
      score += inverseDocumentFrequency * frequency * (k1 + 1) / denominator;
    }
    const pathTerms = tokenizeTerms(chunk.path);
    const pathMatches = uniqueQueryTerms.filter(term => pathTerms.includes(term)).length;
    score += pathMatches * 0.15;
    return { ...chunk, score };
  }).filter(item => item.score > 0)
    .sort((a, b) => b.score - a.score
      || a.path.localeCompare(b.path)
      || a.startLine - b.startLine);
}

export async function searchWorkspaceSemantic(inputRoot, query, limit = 8, { signal } = {}) {
  const root = await resolveWorkspace(inputRoot);
  if (typeof query !== "string" || query.trim().length < 2 || query.length > 300) {
    throw new Error("Workspace search query must contain 2-300 characters.");
  }
  if (!Number.isInteger(limit) || limit < 1 || limit > 20) {
    throw new Error("Workspace search result limit must be an integer from 1 to 20.");
  }

  const index = await readIndex(root);
  if (!index) {
    return {
      available: false,
      root,
      query,
      results: [],
      reason: "local_retrieval_index_missing_or_invalid"
    };
  }

  const ranked = scoreChunks(index.chunks, tokenizeTerms(query));
  const results = [];
  const checked = new Map();
  for (const item of ranked) {
    if (signal?.aborted) throw new Error("Workspace search was cancelled.");
    if (results.length >= limit) break;
    if (checked.has(item.path) && !checked.get(item.path)) continue;

    if (!checked.has(item.path)) {
      try {
        if (await isWorkspacePathExcluded(root, item.path)) {
          checked.set(item.path, false);
          continue;
        }
        const current = await readWorkspaceFile(root, item.path);
        checked.set(item.path, current.sha256 === item.sha256 ? current : false);
      } catch {
        checked.set(item.path, false);
      }
    }

    const source = checked.get(item.path);
    if (!source) continue;
    const excerpt = source.content.split(/\r?\n/)
      .slice(item.startLine - 1, item.endLine)
      .join("\n")
      .slice(0, MAX_CHUNK_CHARS);
    results.push({
      path: item.path,
      startLine: item.startLine,
      endLine: item.endLine,
      score: Number(item.score.toFixed(5)),
      sha256: item.sha256,
      excerpt
    });
  }

  return {
    available: true,
    root,
    query: query.trim(),
    indexMethod: INDEX_METHOD,
    results,
    staleResults: ranked.length - results.length
  };
}

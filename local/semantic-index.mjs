import fs from "node:fs/promises";
import path from "node:path";
import crypto from "node:crypto";
import {
  ensureWorkspaceStateDirectory, getWorkspaceStateDirectory, isWorkspacePathExcluded,
  listWorkspaceIndexCandidates, readWorkspaceFile, resolveWorkspace
} from "./workspace.mjs";

const INDEX_NAME = "semantic-index-v2.json";
const MAX_INDEX_BYTES = 8 * 1024 * 1024;
const MAX_FILES = 500, MAX_CHUNKS = 600, MAX_CHUNK_CHARS = 2400;
const CHUNK_LINES = 36, CHUNK_OVERLAP = 6;
const MAX_EMBEDDING_TOKENS = 256;
const DEFAULT_EMBEDDING_MODEL = "syntra-native";

function cosineSimilarity(a, b) {
  if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length || !a.length) return -1;
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i += 1) {
    dot += a[i] * b[i]; normA += a[i] * a[i]; normB += b[i] * b[i];
  }
  return normA > 0 && normB > 0 ? dot / Math.sqrt(normA * normB) : -1;
}
const DECLARATION_BOUNDARY = /^(?:export\s+(?:default\s+)?)?(?:async\s+)?(?:function|class|interface|type|enum|def|fn|pub\s+fn|func|struct|impl)\b/;

export function splitIntoChunks(relativePath, content) {
  const lines = String(content || "").split(/\r?\n/), chunks = [];
  let start = 0;
  while (start < lines.length && chunks.length < MAX_CHUNKS) {
    let end = Math.min(lines.length, start + CHUNK_LINES);
    let nextStart = Math.max(start + 1, end - CHUNK_OVERLAP);
    if (end < lines.length) {
      for (let probe = start + 14; probe < end; probe += 1) {
        if (DECLARATION_BOUNDARY.test(lines[probe].trim())) {
          end = probe;
          nextStart = Math.max(start + 1, probe - Math.min(CHUNK_OVERLAP, 2));
          break;
        }
      }
    }

    let text = lines.slice(start, end).join("\n");
    while (text.length > MAX_CHUNK_CHARS && end > start + 1) {
      end -= 1;
      nextStart = Math.max(start + 1, end - CHUNK_OVERLAP);
      text = lines.slice(start, end).join("\n");
    }

    if (end === start + 1 && Array.from(lines[start]).length > MAX_CHUNK_CHARS) {
      const points = Array.from(lines[start]);
      for (let offset = 0; offset < points.length && chunks.length < MAX_CHUNKS; offset += MAX_CHUNK_CHARS) {
        const finish = Math.min(points.length, offset + MAX_CHUNK_CHARS);
        const part = points.slice(offset, finish).join("");
        if (part.trim().length >= 40) {
          chunks.push({
            path: relativePath,
            startLine: start + 1,
            endLine: start + 1,
            startColumn: offset + 1,
            endColumn: finish + 1,
            text: part
          });
        }
      }
      start += 1;
      continue;
    }

    if (text.trim().length >= 40) {
      chunks.push({ path: relativePath, startLine: start + 1, endLine: end, text });
    }
    if (end >= lines.length || chunks.length >= MAX_CHUNKS) break;
    start = Math.max(start + 1, Math.min(nextStart, end));
  }
  return chunks;
}

function splitOversizedChunk(chunk) {
  const lines = String(chunk.text).split(/\r?\n/);
  if (lines.length > 1) {
    const midpoint = Math.floor(lines.length / 2);
    if (midpoint < 1 || midpoint >= lines.length) return null;
    const first = {
      ...chunk,
      endLine: chunk.startLine + midpoint - 1,
      text: lines.slice(0, midpoint).join("\n")
    };
    delete first.endColumn;
    const second = {
      ...chunk,
      startLine: chunk.startLine + midpoint,
      startColumn: 1,
      text: lines.slice(midpoint).join("\n")
    };
    return [first, second].filter(item => item.text.trim().length >= 1);
  }

  const points = Array.from(lines[0]);
  if (points.length < 2) return null;
  const midpoint = Math.floor(points.length / 2);
  const baseColumn = Number.isInteger(chunk.startColumn) ? chunk.startColumn : 1;
  const finalColumn = Number.isInteger(chunk.endColumn) ? chunk.endColumn : baseColumn + points.length;
  return [
    {
      ...chunk,
      startColumn: baseColumn,
      endColumn: baseColumn + midpoint,
      text: points.slice(0, midpoint).join("")
    },
    {
      ...chunk,
      startColumn: baseColumn + midpoint,
      endColumn: finalColumn,
      text: points.slice(midpoint).join("")
    }
  ].filter(item => item.text.trim().length >= 1);
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
    if (data?.schemaVersion !== 2 || data.root !== root || data.embeddingModel !== DEFAULT_EMBEDDING_MODEL
      || !/^[a-f0-9]{64}$/i.test(data.checkpointIdentity || "")
      || !Number.isInteger(data.dimensions) || data.dimensions < 8 || data.dimensions > 4096
      || !Number.isInteger(data.maxInputTokens) || data.maxInputTokens < 1 || data.maxInputTokens > MAX_EMBEDDING_TOKENS
      || !Array.isArray(data.chunks) || data.chunks.length > MAX_CHUNKS) return null;
    const valid = data.chunks.filter(item => item && typeof item.path === "string"
      && Number.isInteger(item.startLine) && item.startLine > 0
      && Number.isInteger(item.endLine) && item.endLine >= item.startLine
      && (item.startColumn === undefined || (Number.isInteger(item.startColumn) && item.startColumn > 0))
      && (item.endColumn === undefined || (Number.isInteger(item.endColumn) && item.endColumn > 1))
      && (item.startColumn === undefined || item.endLine === item.startLine)
      && Number.isInteger(item.tokenCount) && item.tokenCount >= 1 && item.tokenCount <= data.maxInputTokens
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
async function nativeRuntimeRequest(route, body, fetchImpl, signal) {
  const aiPort = Number(process.env.SYNTRA_AI_PORT || 3002);
  if (!Number.isInteger(aiPort) || aiPort < 1024 || aiPort > 65535) {
    throw new Error("SYNTRA_AI_PORT must be an integer between 1024 and 65535.");
  }
  const timeout = AbortSignal.timeout(120000);
  const response = await fetchImpl(`http://127.0.0.1:${aiPort}${route}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    redirect: "error",
    body: JSON.stringify(body),
    signal: signal ? AbortSignal.any([signal, timeout]) : timeout
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(String(payload.error || "Native runtime request failed with HTTP " + response.status));
  if (!/^[a-f0-9]{64}$/i.test(payload.checkpoint_identity || "")) {
    throw new Error("Native runtime did not identify the checkpoint used for this request.");
  }
  return payload;
}

async function countViaNativeRuntime(texts, fetchImpl, signal) {
  const payload = await nativeRuntimeRequest("/model/tokenize", {
    texts,
    max_length: MAX_EMBEDDING_TOKENS
  }, fetchImpl, signal);
  if (!Array.isArray(payload.token_counts) || payload.token_counts.length !== texts.length
    || payload.token_counts.some(value => !Number.isInteger(value) || value < 0)
    || !Number.isInteger(payload.max_input_tokens) || payload.max_input_tokens < 1
    || payload.max_input_tokens > MAX_EMBEDDING_TOKENS) {
    throw new Error("Native runtime returned invalid token-count evidence for semantic chunks.");
  }
  return {
    checkpointIdentity: payload.checkpoint_identity,
    tokenCounts: payload.token_counts,
    maxInputTokens: payload.max_input_tokens
  };
}

async function embedViaNativeRuntime(texts, fetchImpl, signal) {
  const payload = await nativeRuntimeRequest("/model/embed", {
    texts,
    max_length: MAX_EMBEDDING_TOKENS,
    device: "auto"
  }, fetchImpl, signal);
  if (!Array.isArray(payload.embeddings) || payload.embeddings.length !== texts.length
    || !Array.isArray(payload.input_token_counts) || payload.input_token_counts.length !== texts.length
    || payload.input_token_counts.some(value => !Number.isInteger(value) || value < 0)
    || !Number.isInteger(payload.max_input_tokens) || payload.max_input_tokens < 1
    || payload.max_input_tokens > MAX_EMBEDDING_TOKENS
    || payload.input_token_counts.some(value => value > payload.max_input_tokens)) {
    throw new Error("Native runtime returned incomplete or truncated embedding-input evidence.");
  }
  const dims = payload.embeddings[0]?.length;
  if (!Number.isInteger(dims) || dims < 8 || dims > 4096
    || payload.embeddings.some(v => !Array.isArray(v) || v.length !== dims || v.some(n => !Number.isFinite(n)))) {
    throw new Error("Native runtime returned inconsistent or invalid embedding dimensions.");
  }
  return {
    checkpointIdentity: payload.checkpoint_identity,
    inputTokenCounts: payload.input_token_counts,
    maxInputTokens: payload.max_input_tokens,
    embeddings: payload.embeddings.map(v => v.map(n => Number(Number(n).toFixed(5))))
  };
}

async function fitChunksToNativeContext(chunks, expectedCheckpointIdentity, fetchImpl, signal) {
  const queue = [...chunks];
  const fitted = [];
  let maxInputTokens = MAX_EMBEDDING_TOKENS;
  while (queue.length) {
    const batch = queue.splice(0, 64);
    const measured = await countViaNativeRuntime(batch.map(item => item.text), fetchImpl, signal);
    if (measured.checkpointIdentity !== expectedCheckpointIdentity) {
      throw new Error("The native checkpoint changed during semantic chunk tokenization; retry the operation.");
    }
    maxInputTokens = Math.min(maxInputTokens, measured.maxInputTokens);
    for (let index = 0; index < batch.length; index += 1) {
      const item = batch[index];
      const tokenCount = measured.tokenCounts[index];
      if (tokenCount <= measured.maxInputTokens) {
        fitted.push({ ...item, tokenCount });
        continue;
      }
      const parts = splitOversizedChunk(item);
      if (!parts || parts.length < 2) {
        throw new Error(
          "Cannot index " + item.path + ":" + item.startLine +
          " because this source fragment requires " + tokenCount +
          " native tokenizer tokens and cannot be split further without losing source identity."
        );
      }
      if (fitted.length + queue.length + parts.length > MAX_CHUNKS) {
        throw new Error(
          "Lossless token-aware chunking would exceed the semantic index limit of " +
          MAX_CHUNKS + " chunks; no partial index was written."
        );
      }
      queue.unshift(...parts);
    }
  }
  return { chunks: fitted, maxInputTokens };
}

async function embed(texts, model, fetchImpl, signal) {
  if (!texts.length) return [];
  if (model !== DEFAULT_EMBEDDING_MODEL) {
    throw new Error("SYNTRA semantic retrieval accepts only embeddings produced by its own trained checkpoint.");
  }
  return embedViaNativeRuntime(texts, fetchImpl, signal);
}

export async function getSemanticIndexStatus(inputRoot) {
  const root = await resolveWorkspace(inputRoot), index = await readIndex(root);
  return index
    ? { available: true, root, embeddingModel: index.embeddingModel, checkpointIdentity: index.checkpointIdentity, indexedAt: index.indexedAt, chunkCount: index.chunks.length, dimensions: index.dimensions, maxInputTokens: index.maxInputTokens, schemaVersion: index.schemaVersion }
    : { available: false, root, embeddingModel: DEFAULT_EMBEDDING_MODEL, checkpointIdentity: null, chunkCount: 0, reason: "not_built_or_invalid" };
}
export async function refreshWorkspaceSemanticIndex(inputRoot, {
  fetchImpl = globalThis.fetch, model = DEFAULT_EMBEDDING_MODEL, signal
} = {}) {
  const root = await resolveWorkspace(inputRoot);
  if (model !== DEFAULT_EMBEDDING_MODEL) throw new Error("SYNTRA semantic retrieval uses only its native checkpoint embeddings.");
  const identityProbe = await embed(["SYNTRA native semantic index checkpoint identity probe"], model, fetchImpl, signal);
  const checkpointIdentity = identityProbe.checkpointIdentity;
  const previous = await readIndex(root), oldByPath = new Map();
  if (previous?.embeddingModel === model && previous.checkpointIdentity === checkpointIdentity) {
    for (const item of previous.chunks) {
      const list = oldByPath.get(item.path) || []; list.push(item); oldByPath.set(item.path, list);
    }
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
  const fitted = await fitChunksToNativeContext(
    pending,
    checkpointIdentity,
    fetchImpl,
    signal,
    identityProbe.maxInputTokens
  );
  if (chunks.length + fitted.chunks.length > MAX_CHUNKS) {
    throw new Error(
      "Lossless semantic-index refresh would exceed the " + MAX_CHUNKS +
      " chunk storage limit; no partial index was written."
    );
  }
  for (let offset = 0; offset < fitted.chunks.length; offset += 24) {
    const batch = fitted.chunks.slice(offset, offset + 24);
    const generated = await embed(batch.map(item => item.text), model, fetchImpl, signal);
    if (generated.checkpointIdentity !== checkpointIdentity) {
      throw new Error("The native checkpoint changed during semantic-index refresh; retry the operation.");
    }
    if (generated.maxInputTokens !== fitted.maxInputTokens
      || generated.inputTokenCounts.some((count, index) => count !== batch[index].tokenCount)) {
      throw new Error("Native tokenizer counts changed between semantic preflight and embedding; no index was written.");
    }
    for (let i = 0; i < batch.length; i += 1) {
      const { text: _sourceTextNotPersisted, ...metadata } = batch[i];
      chunks.push({ ...metadata, vector: generated.embeddings[i] });
    }
  }
  const dimensions = chunks[0]?.vector?.length || identityProbe.embeddings[0]?.length || 0;
  if (chunks.some(item => item.vector.length !== dimensions)) throw new Error("Semantic index contains inconsistent vector dimensions.");
  const index = {
    schemaVersion: 2,
    root,
    embeddingModel: model,
    checkpointIdentity,
    dimensions,
    maxInputTokens: fitted.maxInputTokens,
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
    embeddingModel: model,
    checkpointIdentity,
    indexedAt: index.indexedAt,
    chunkCount: chunks.length,
    dimensions,
    maxInputTokens: index.maxInputTokens,
    scannedFiles,
    scannedBytes,
    reusedChunks,
    splitForContextChunks: Math.max(0, fitted.chunks.length - pending.length),
    skippedFiles,
    truncated: index.truncated
  };
}
export async function searchWorkspaceSemantic(inputRoot, query, limit = 8, {
  fetchImpl = globalThis.fetch, model = DEFAULT_EMBEDDING_MODEL, signal
} = {}) {
  const root = await resolveWorkspace(inputRoot);
  if (model !== DEFAULT_EMBEDDING_MODEL) throw new Error("SYNTRA semantic retrieval uses only its native checkpoint embeddings.");
  if (typeof query !== "string" || query.trim().length < 2 || query.length > 300) throw new Error("Semantic query must contain 2-300 characters.");
  if (!Number.isInteger(limit) || limit < 1 || limit > 20) throw new Error("Semantic result limit must be an integer from 1 to 20.");
  const index = await readIndex(root);
  if (!index || index.embeddingModel !== model) return { available: false, root, query, results: [], reason: "semantic_index_missing_or_invalid" };
  const queryResult = await embed([query.trim()], model, fetchImpl, signal);
  if (queryResult.checkpointIdentity !== index.checkpointIdentity) {
    return {
      available: false, root, query, results: [],
      reason: "native_checkpoint_changed_refresh_required",
      checkpointIdentity: queryResult.checkpointIdentity
    };
  }
  if (queryResult.maxInputTokens !== index.maxInputTokens) {
    throw new Error("Native embedding token limits differ from the saved semantic index. Refresh the semantic index.");
  }
  const [queryVector] = queryResult.embeddings;
  if (queryVector.length !== index.dimensions) throw new Error("Query embedding dimensions do not match the saved native checkpoint. Refresh the semantic index.");
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
    const excerptLines = source.content.split(/\r?\n/).slice(item.startLine - 1, item.endLine);
    if (excerptLines.length === 1 && (item.startColumn !== undefined || item.endColumn !== undefined)) {
      const points = Array.from(excerptLines[0]);
      const from = Math.max(0, (item.startColumn || 1) - 1);
      const to = item.endColumn === undefined ? points.length : Math.max(from, item.endColumn - 1);
      excerptLines[0] = points.slice(from, to).join("");
    } else {
      if (item.startColumn !== undefined && excerptLines.length) {
        excerptLines[0] = Array.from(excerptLines[0]).slice(item.startColumn - 1).join("");
      }
      if (item.endColumn !== undefined && excerptLines.length) {
        const last = excerptLines.length - 1;
        excerptLines[last] = Array.from(excerptLines[last]).slice(0, item.endColumn - 1).join("");
      }
    }
    const excerpt = excerptLines.join("\n");
    results.push({
      path: item.path,
      startLine: item.startLine,
      endLine: item.endLine,
      startColumn: item.startColumn,
      endColumn: item.endColumn,
      tokenCount: item.tokenCount,
      score: Number(item.score.toFixed(5)),
      sha256: item.sha256,
      excerpt
    });
  }
  return { available: true, root, query: query.trim(), embeddingModel: model,
    checkpointIdentity: index.checkpointIdentity, results, staleResults: ranked.length - results.length };
}

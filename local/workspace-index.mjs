import fs from "node:fs/promises";
import path from "node:path";
import crypto from "node:crypto";
import {
  ensureWorkspaceStateDirectory,
  getWorkspaceStateDirectory,
  listWorkspaceIndexCandidates,
  readWorkspaceFile,
  resolveWorkspace
} from "./workspace.mjs";

const INDEX_FILENAME = "workspace-index-v1.json";
const INDEX_SCHEMA_VERSION = 1;
const MAX_INDEX_CANDIDATES = 2000;
const MAX_INDEX_SCAN_BYTES = 32 * 1024 * 1024;
const MAX_INDEX_BYTES = 4 * 1024 * 1024;
const MAX_TOTAL_SYMBOLS = 6000;
const MAX_SYMBOLS_PER_FILE = 80;
const MAX_SEARCH_VALIDATIONS = 100;
const HASH_PATTERN = /^[a-f0-9]{64}$/i;
const SYMBOL_KINDS = new Set(["function", "class", "interface", "type", "enum", "struct", "trait", "module"]);
const IDENTIFIER_PATTERN = /^[A-Za-z_$][A-Za-z0-9_$]{0,119}$/;
const SOURCE_EXTENSIONS = new Set([
  ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".mts", ".cts", ".py", ".go", ".rs"
]);

function hash(value) {
  return crypto.createHash("sha256").update(value).digest("hex");
}

function safeIndexedPath(value) {
  if (typeof value !== "string" || !value || value.length > 1024 || value.includes("\\")) return false;
  if (value.startsWith("/") || /^[a-zA-Z]:/.test(value)) return false;
  const parts = value.split("/");
  return !parts.some(part => !part || part === "." || part === "..")
    && parts[0] !== ".syntra"
    && !parts.some(part => {
      const lower = part.toLowerCase();
      return lower === ".git" || lower === "secrets" || lower === "credentials"
        || lower === ".env" || lower.startsWith(".env.") && lower !== ".env.example"
        || lower.endsWith(".secret") || [".pem", ".key", ".p12", ".pfx", ".keystore"].includes(path.posix.extname(lower));
    });
}

function parseDeclarations(relativePath, content) {
  const extension = path.posix.extname(relativePath).toLowerCase();
  if (!SOURCE_EXTENSIONS.has(extension)) return [];
  const symbols = [];
  const seen = new Set();
  const lines = content.split(/\r?\n/);

  function add(kind, name, line) {
    if (!SYMBOL_KINDS.has(kind) || !IDENTIFIER_PATTERN.test(name) || symbols.length >= MAX_SYMBOLS_PER_FILE) return;
    const key = kind + ":" + name + ":" + line;
    if (seen.has(key)) return;
    seen.add(key);
    symbols.push({ name, kind, line, signature: kind + " " + name });
  }

  for (let index = 0; index < lines.length && symbols.length < MAX_SYMBOLS_PER_FILE; index += 1) {
    const line = lines[index];
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("//") || trimmed.startsWith("#") || trimmed.startsWith("*")) continue;
    let match;

    if ([".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".mts", ".cts"].includes(extension)) {
      match = /^\s*(?:(?:export|declare|default|abstract)\s+)*(?:async\s+)?function\s*\*?\s+([A-Za-z_$][\w$]*)\b/.exec(line);
      if (match) add("function", match[1], index + 1);
      match = /^\s*(?:(?:export|default|abstract|declare)\s+)*class\s+([A-Za-z_$][\w$]*)\b/.exec(line);
      if (match) add("class", match[1], index + 1);
      match = /^\s*(?:(?:export|declare)\s+)*interface\s+([A-Za-z_$][\w$]*)\b/.exec(line);
      if (match) add("interface", match[1], index + 1);
      match = /^\s*(?:(?:export|declare)\s+)*type\s+([A-Za-z_$][\w$]*)\s*=/.exec(line);
      if (match) add("type", match[1], index + 1);
      match = /^\s*(?:(?:export|declare)\s+)*enum\s+([A-Za-z_$][\w$]*)\b/.exec(line);
      if (match) add("enum", match[1], index + 1);
      match = /^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*=>/.exec(line);
      if (match) add("function", match[1], index + 1);
    } else if (extension === ".py") {
      match = /^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\(/.exec(line);
      if (match) add("function", match[1], index + 1);
      match = /^\s*class\s+([A-Za-z_]\w*)\b/.exec(line);
      if (match) add("class", match[1], index + 1);
    } else if (extension === ".go") {
      match = /^\s*func\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)\s*\(/.exec(line);
      if (match) add("function", match[1], index + 1);
      match = /^\s*type\s+([A-Za-z_]\w*)\s+(?:struct|interface)\b/.exec(line);
      if (match) add("type", match[1], index + 1);
      match = /^\s*type\s+([A-Za-z_]\w*)\s+/.exec(line);
      if (match) add("type", match[1], index + 1);
    } else if (extension === ".rs") {
      match = /^\s*(?:(?:pub(?:\([^)]*\))?|async|unsafe|extern(?:\s+"[^"]+")?)\s+)*(fn|struct|enum|trait|type|mod)\s+([A-Za-z_]\w*)\b/.exec(line);
      if (match) {
        const kind = match[1] === "fn" ? "function"
          : match[1] === "mod" ? "module" : match[1];
        add(kind, match[2], index + 1);
      }
    }
  }
  return symbols;
}

function validateLoadedIndex(root, value) {
  if (!value || typeof value !== "object" || Array.isArray(value)
    || value.schemaVersion !== INDEX_SCHEMA_VERSION
    || value.root !== root
    || typeof value.indexedAt !== "string" || !Number.isFinite(Date.parse(value.indexedAt))
    || !Array.isArray(value.files) || value.files.length > MAX_INDEX_CANDIDATES) return null;

  let totalSymbols = 0;
  const seenPaths = new Set();
  const files = [];
  for (const file of value.files) {
    if (!file || typeof file !== "object" || Array.isArray(file)
      || !safeIndexedPath(file.path) || seenPaths.has(file.path)
      || !SOURCE_EXTENSIONS.has(path.posix.extname(file.path).toLowerCase())
      || !Number.isInteger(file.size) || file.size < 0 || file.size > 512 * 1024
      || typeof file.sha256 !== "string" || !HASH_PATTERN.test(file.sha256)
      || !Array.isArray(file.symbols) || file.symbols.length > MAX_SYMBOLS_PER_FILE) return null;
    seenPaths.add(file.path);
    const symbols = [];
    for (const symbol of file.symbols) {
      if (!symbol || typeof symbol !== "object" || Array.isArray(symbol)
        || !IDENTIFIER_PATTERN.test(symbol.name || "") || !SYMBOL_KINDS.has(symbol.kind)
        || !Number.isInteger(symbol.line) || symbol.line < 1 || symbol.line > 10_000_000
        || typeof symbol.signature !== "string" || symbol.signature.length > 160
        || symbol.signature !== symbol.kind + " " + symbol.name) return null;
      symbols.push({
        name: symbol.name, kind: symbol.kind, line: symbol.line, signature: symbol.signature
      });
    }
    totalSymbols += symbols.length;
    if (totalSymbols > MAX_TOTAL_SYMBOLS) return null;
    files.push({
      path: file.path, size: file.size, sha256: file.sha256, symbols
    });
  }

  return {
    schemaVersion: INDEX_SCHEMA_VERSION,
    root,
    indexedAt: value.indexedAt,
    truncated: value.truncated === true,
    files,
    symbolCount: totalSymbols
  };
}

async function indexPathFor(root, create) {
  const directory = create
    ? await ensureWorkspaceStateDirectory(root)
    : await getWorkspaceStateDirectory(root);
  return path.join(directory, INDEX_FILENAME);
}

async function readIndex(root) {
  const target = await indexPathFor(root, false);
  let stat;
  try {
    stat = await fs.lstat(target);
  } catch (error) {
    if (error && error.code === "ENOENT") return null;
    throw error;
  }
  if (stat.isSymbolicLink() || !stat.isFile()) {
    throw new Error("The workspace index must be a regular file, not a symbolic link.");
  }
  if (stat.size > MAX_INDEX_BYTES) return null;
  try {
    const value = JSON.parse(await fs.readFile(target, "utf8"));
    return validateLoadedIndex(root, value);
  } catch {
    return null;
  }
}

async function writeIndex(root, value) {
  const target = await indexPathFor(root, true);
  const serialized = JSON.stringify(value);
  if (Buffer.byteLength(serialized, "utf8") > MAX_INDEX_BYTES) {
    throw new Error("The workspace index exceeded its 4 MB storage limit.");
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

function countSymbols(files) {
  return files.reduce((total, file) => total + file.symbols.length, 0);
}

export async function getWorkspaceIndexStatus(inputRoot) {
  const root = await resolveWorkspace(inputRoot);
  const index = await readIndex(root);
  if (!index) {
    return {
      available: false, root, indexedAt: null, fileCount: 0, symbolCount: 0,
      indexedBytes: 0, truncated: false, reason: "not_built_or_invalid"
    };
  }
  return {
    available: true, root, indexedAt: index.indexedAt, fileCount: index.files.length,
    symbolCount: index.symbolCount, indexedBytes: index.files.reduce((sum, file) => sum + file.size, 0),
    truncated: index.truncated
  };
}

export async function refreshWorkspaceIndex(inputRoot) {
  const root = await resolveWorkspace(inputRoot);
  await ensureWorkspaceStateDirectory(root);
  const previous = await readIndex(root);
  const candidates = await listWorkspaceIndexCandidates(root);
  const candidatesByPath = new Map(candidates.files.map(file => [file.path, file]));
  const previousByPath = new Map((previous?.files || []).map(file => [file.path, file]));
  const nextFiles = [];
  const currentPaths = new Set(candidates.files.map(file => file.path));
  const removedFiles = [...previousByPath.keys()].filter(filePath => !currentPaths.has(filePath)).length;
  let parsedFiles = 0;
  let reusedFiles = 0;
  let skippedFiles = 0;
  let indexedBytes = 0;
  let truncated = candidates.truncated || candidates.files.length > MAX_INDEX_CANDIDATES;
  let totalSymbols = 0;

  for (let index = 0; index < candidates.files.length; index += 1) {
    const candidate = candidates.files[index];
    if (index >= MAX_INDEX_CANDIDATES) {
      skippedFiles += candidates.files.length - index;
      truncated = true;
      break;
    }
    if (indexedBytes + candidate.size > MAX_INDEX_SCAN_BYTES) {
      skippedFiles += candidates.files.length - index;
      truncated = true;
      break;
    }
    let source;
    try {
      source = await readWorkspaceFile(root, candidate.path);
    } catch {
      skippedFiles += 1;
      truncated = true;
      continue;
    }
    if (indexedBytes + source.bytes > MAX_INDEX_SCAN_BYTES) {
      skippedFiles += candidates.files.length - index;
      truncated = true;
      break;
    }
    indexedBytes += source.bytes;

    const cached = previousByPath.get(candidate.path);
    if (cached && cached.sha256 === source.sha256 && cached.size === source.bytes) {
      const symbols = cached.symbols.slice(0, Math.max(0, MAX_TOTAL_SYMBOLS - totalSymbols));
      nextFiles.push({
        path: candidate.path, size: source.bytes, sha256: source.sha256, symbols
      });
      totalSymbols += symbols.length;
      reusedFiles += 1;
      if (symbols.length < cached.symbols.length) truncated = true;
      continue;
    }

    const parsed = parseDeclarations(candidate.path, source.content);
    const symbols = parsed.slice(0, Math.max(0, MAX_TOTAL_SYMBOLS - totalSymbols));
    nextFiles.push({
      path: candidate.path, size: source.bytes, sha256: source.sha256, symbols
    });
    totalSymbols += symbols.length;
    parsedFiles += 1;
    if (symbols.length < parsed.length) truncated = true;
  }

  const index = {
    schemaVersion: INDEX_SCHEMA_VERSION,
    root,
    indexedAt: new Date().toISOString(),
    truncated,
    files: nextFiles,
    symbolCount: countSymbols(nextFiles)
  };
  await writeIndex(root, index);
  return {
    available: true, root, indexedAt: index.indexedAt,
    fileCount: nextFiles.length, symbolCount: index.symbolCount,
    parsedFiles, reusedFiles, removedFiles, skippedFiles, indexedBytes,
    truncated, rebuilt: !previous
  };
}

function symbolScore(symbol, filePath, words, query) {
  const name = symbol.name.toLowerCase();
  const signature = symbol.signature.toLowerCase();
  const file = filePath.toLowerCase();
  if (name === query) return 100;
  if (name.startsWith(query)) return 80;
  if (name.includes(query)) return 65;
  let score = 0;
  for (const word of words) {
    if (name.includes(word)) score += 12;
    if (signature.includes(word)) score += 5;
    if (file.includes(word)) score += 2;
  }
  return score;
}

export async function searchWorkspaceSymbols(inputRoot, query, limit = 10) {
  const root = await resolveWorkspace(inputRoot);
  if (typeof query !== "string" || query.trim().length < 2 || query.length > 160) {
    throw new Error("Symbol search query must contain 2-160 characters.");
  }
  if (!Number.isInteger(limit) || limit < 1 || limit > 20) {
    throw new Error("Symbol search limit must be an integer from 1 to 20.");
  }
  const index = await readIndex(root);
  if (!index) return { available: false, root, query, results: [], staleFiles: 0, truncated: false };

  const normalized = query.trim().toLowerCase();
  const words = [...new Set(normalized.match(/[a-z0-9_$]{2,}/g) || [])].slice(0, 8);
  const ranked = [];
  for (const file of index.files) {
    for (const symbol of file.symbols) {
      const score = symbolScore(symbol, file.path, words, normalized);
      if (score > 0) ranked.push({ ...symbol, path: file.path, score });
    }
  }
  ranked.sort((a, b) => b.score - a.score || a.path.localeCompare(b.path) || a.line - b.line);
  const selected = [];
  const validatedPaths = new Map();
  let staleFiles = 0;
  let validationCount = 0;
  for (const item of ranked) {
    if (selected.length >= limit) break;
    if (!validatedPaths.has(item.path)) {
      if (validationCount >= MAX_SEARCH_VALIDATIONS) break;
      validationCount += 1;
      try {
        const current = await readWorkspaceFile(root, item.path);
        validatedPaths.set(item.path, current.sha256 === index.files.find(file => file.path === item.path)?.sha256);
      } catch {
        validatedPaths.set(item.path, false);
      }
      if (!validatedPaths.get(item.path)) staleFiles += 1;
    }
    if (validatedPaths.get(item.path)) selected.push(item);
  }

  return {
    available: true, root, query, results: selected, staleFiles,
    truncated: index.truncated || (ranked.length > selected.length && validationCount >= MAX_SEARCH_VALIDATIONS),
    indexedAt: index.indexedAt
  };
}

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
const INDEX_SCHEMA_VERSION = 2;
const MAX_INDEX_CANDIDATES = 2000;
const MAX_INDEX_SCAN_BYTES = 32 * 1024 * 1024;
const MAX_INDEX_BYTES = 4 * 1024 * 1024;
const MAX_TOTAL_SYMBOLS = 6000;
const MAX_SYMBOLS_PER_FILE = 80;
const MAX_RELATIONS_PER_FILE = 40;
const MAX_SPECIFIERS_PER_FILE = 80;
const MAX_SEARCH_VALIDATIONS = 100;
const HASH_PATTERN = /^[a-f0-9]{64}$/i;
const SYMBOL_KINDS = new Set(["function", "class", "interface", "type", "enum", "struct", "trait", "module"]);
const IDENTIFIER_PATTERN = /^[A-Za-z_$][A-Za-z0-9_$]{0,119}$/;
const SOURCE_EXTENSIONS = new Set([
  ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".mts", ".cts", ".py", ".pyi", ".go", ".rs"
]);
const DOC_EXTENSIONS = new Set([".md", ".mdx", ".rst"]);
const IGNORED_PATH_SEGMENTS = new Set([
  ".git", ".syntra", ".venv", "venv", "env", "node_modules", "__pycache__",
  ".pytest_cache", ".mypy_cache", ".ruff_cache", "dist", "build", "coverage",
  ".next", ".nuxt", ".idea", ".vscode", "target", "vendor", "obj"
]);
const CODE_EXTENSIONS = new Set([...SOURCE_EXTENSIONS]);
const JS_EXTENSIONS = new Set([".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".mts", ".cts"]);
const SOURCE_CANDIDATES = [
  ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".py", ".pyi", ".go", ".rs"
];

function safeIndexedPath(value) {
  if (typeof value !== "string" || !value || value.length > 1024 || value.includes("\\")) return false;
  if (value.startsWith("/") || /^[a-zA-Z]:/.test(value)) return false;
  const parts = value.split("/");
  if (parts.some(part => !part || part === "." || part === "..")) return false;
  if (parts.some(part => IGNORED_PATH_SEGMENTS.has(part.toLowerCase()))) return false;
  return !parts.some(part => {
    const lower = part.toLowerCase();
    return lower === ".env" || lower.startsWith(".env.") && lower !== ".env.example"
      || lower === "secrets" || lower === "credentials" || lower.endsWith(".secret")
      || [".pem", ".key", ".p12", ".pfx", ".keystore"].includes(path.posix.extname(lower));
  });
}

function uniqueStrings(values, maximum, maxLength = 1024) {
  if (!Array.isArray(values) || values.length > maximum) return null;
  const result = [];
  const seen = new Set();
  for (const value of values) {
    if (typeof value !== "string" || !value || value.length > maxLength || seen.has(value)) return null;
    seen.add(value);
    result.push(value);
  }
  return result;
}

function isLocalSpecifier(value) {
  if (typeof value !== "string" || !value || value.length > 300) return false;
  const lower = value.toLowerCase();
  return !value.includes("\\") && !value.includes("\0")
    && !lower.startsWith("http:") && !lower.startsWith("https:")
    && !lower.startsWith("data:") && !lower.startsWith("mailto:")
    && !value.startsWith("//") && !value.startsWith("/");
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

    if (JS_EXTENSIONS.has(extension)) {
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
    } else if (extension === ".py" || extension === ".pyi") {
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

function classifyRole(relativePath, suggestedRole) {
  if (suggestedRole === "documentation" || DOC_EXTENSIONS.has(path.posix.extname(relativePath).toLowerCase())) {
    return "documentation";
  }
  const segments = relativePath.toLowerCase().split("/");
  const base = path.posix.basename(relativePath).toLowerCase();
  const testDirectory = segments.some(segment => ["test", "tests", "__tests__", "spec", "specs"].includes(segment));
  const testName = /(?:^test[_-].+|.*[._-](?:test|spec)|.*_test)$/.test(base.replace(/\.[^.]+$/, ""));
  return testDirectory || testName ? "test" : "source";
}

function extractImportSpecifiers(relativePath, content) {
  const extension = path.posix.extname(relativePath).toLowerCase();
  const result = [];
  const add = value => {
    if (!isLocalSpecifier(value) || result.length >= MAX_SPECIFIERS_PER_FILE) return;
    if (!result.includes(value)) result.push(value);
  };

  if (JS_EXTENSIONS.has(extension)) {
    const pattern = /\bfrom\s*["']([^"']+)["']|\bimport\s*(?:\(\s*)?["']([^"']+)["']|\brequire\s*\(\s*["']([^"']+)["']/g;
    for (const match of content.matchAll(pattern)) {
      const specifier = match[1] || match[2] || match[3];
      if (specifier?.startsWith(".")) add(specifier);
    }
    const css = /@import\s+(?:url\()?["']([^"']+)["']/g;
    for (const match of content.matchAll(css)) if (match[1]?.startsWith(".")) add(match[1]);
  } else if (extension === ".py" || extension === ".pyi") {
    const pattern = /^\s*(?:from\s+([.\w]+)\s+import\b|import\s+([\w.]+))/gm;
    for (const match of content.matchAll(pattern)) add(match[1] || match[2]);
  } else if (extension === ".go") {
    const blocks = content.match(/\bimport\s*\([\s\S]*?\)/g) || [];
    for (const block of blocks) {
      for (const match of block.matchAll(/["']([^"']+)["']/g)) add(match[1]);
    }
    const single = /^\s*import\s+(?:[A-Za-z_]\w*\s+)?["']([^"']+)["']/gm;
    for (const match of content.matchAll(single)) add(match[1]);
  } else if (extension === ".rs") {
    const modules = /^\s*(?:pub(?:\([^)]*\))?\s+)?mod\s+([A-Za-z_]\w*)\s*;/gm;
    for (const match of content.matchAll(modules)) add("mod:" + match[1]);
    const uses = /^\s*(?:pub(?:\([^)]*\))?\s+)?use\s+((?:crate|self|super)::[A-Za-z0-9_:{}*, ]+)\s*;/gm;
    for (const match of content.matchAll(uses)) {
      const specifier = match[1].replace(/[{}* ]/g, "");
      if (specifier) add(specifier);
    }
  }
  return result;
}

function extractDocumentationTargets(content) {
  const targets = [];
  const seen = new Set();
  const pattern = /!?\[[^\]]*\]\(\s*<?([^)>\s]+)>?(?:\s+["'][^"']*["'])?\s*\)/g;
  for (const match of content.matchAll(pattern)) {
    let target = match[1] || "";
    if (!target || target.startsWith("#") || !isLocalSpecifier(target)) continue;
    target = target.split("#", 1)[0].split("?", 1)[0];
    if (!target || !isLocalSpecifier(target)) continue;
    try { target = decodeURIComponent(target); } catch {}
    if (!seen.has(target) && targets.length < MAX_RELATIONS_PER_FILE) {
      seen.add(target);
      targets.push(target);
    }
  }
  return targets;
}

function resolveJsImport(source, specifier, pathSet) {
  const base = path.posix.normalize(path.posix.join(path.posix.dirname(source), specifier));
  if (base.startsWith("../") || base === ".." || base.startsWith("/")) return [];
  const candidates = [
    base, ...SOURCE_CANDIDATES.map(extension => base + extension),
    ...[".css", ".scss", ".json"].map(extension => base + extension),
    ...["index.ts", "index.tsx", "index.js", "index.jsx", "index.mjs", "index.cjs"].map(name =>
      path.posix.join(base, name))
  ];
  return candidates.filter(item => pathSet.has(item)).slice(0, 4);
}

function resolvePythonImport(source, specifier, pathSet) {
  if (!specifier || specifier.includes("\\") || specifier.startsWith("/")) return [];
  let base;
  if (!specifier.startsWith(".")) {
    base = specifier.replace(/\./g, "/");
  } else {
    const dots = (specifier.match(/^\.+/) || ["."])[0].length;
    const tail = specifier.slice(dots).replace(/\./g, "/");
    let directory = path.posix.dirname(source);
    for (let count = 1; count < dots; count += 1) directory = path.posix.dirname(directory);
    base = path.posix.join(directory, tail);
  }
  if (base.startsWith("../") || base === ".." || base.startsWith("/")) return [];
  const candidates = [base + ".py", base + ".pyi", base + "/__init__.py", base + "/__init__.pyi"];
  return candidates.filter(item => pathSet.has(item)).slice(0, 2);
}

function resolveRustImport(source, specifier, pathSet) {
  const baseDir = path.posix.dirname(source);
  let segments = [];
  if (specifier.startsWith("mod:")) {
    segments = [specifier.slice(4)];
  } else if (specifier.startsWith("crate::")) {
    segments = ["src", ...specifier.slice(7).split("::")];
    if (segments.length > 2) segments = segments.slice(0, -1);
  } else if (specifier.startsWith("self::")) {
    segments = [baseDir, ...specifier.slice(6).split("::")].filter(Boolean);
    if (segments.length > 1) segments = segments.slice(0, -1);
  } else if (specifier.startsWith("super::")) {
    segments = [path.posix.dirname(baseDir), ...specifier.slice(7).split("::")].filter(Boolean);
    if (segments.length > 1) segments = segments.slice(0, -1);
  } else {
    return [];
  }
  const joined = path.posix.normalize(segments.join("/"));
  if (joined.startsWith("../") || joined === ".." || joined.startsWith("/")) return [];
  const candidates = [joined + ".rs", path.posix.join(joined, "mod.rs")];
  return candidates.filter(item => pathSet.has(item)).slice(0, 2);
}

function resolveGoImport(specifier, modulePath, pathSet, roles) {
  if (!modulePath || (specifier !== modulePath && !specifier.startsWith(modulePath + "/"))) return [];
  const suffix = specifier === modulePath ? "" : specifier.slice(modulePath.length + 1);
  const directory = suffix ? suffix : ".";
  const prefix = directory === "." ? "" : directory.replace(/\/+$/, "") + "/";
  const matches = [...pathSet].filter(item => {
    const extension = path.posix.extname(item).toLowerCase();
    return extension === ".go" && item.startsWith(prefix) && !item.slice(prefix.length).includes("/")
      && roles.get(item) === "source";
  });
  return matches.sort().slice(0, 12);
}

function resolveSourceImport(source, specifier, fileSet, roles, goModulePath) {
  const extension = path.posix.extname(source).toLowerCase();
  if (JS_EXTENSIONS.has(extension)) return resolveJsImport(source, specifier, fileSet);
  if (extension === ".py" || extension === ".pyi") return resolvePythonImport(source, specifier, fileSet);
  if (extension === ".go") return resolveGoImport(specifier, goModulePath, fileSet, roles);
  if (extension === ".rs") return resolveRustImport(source, specifier, fileSet);
  return [];
}

function resolveDocumentationTarget(source, target, pathSet) {
  const base = path.posix.normalize(path.posix.join(path.posix.dirname(source), target));
  if (base.startsWith("../") || base === ".." || base.startsWith("/")) return null;
  const extension = path.posix.extname(base).toLowerCase();
  const candidates = extension
    ? [base]
    : [base, ...[...DOC_EXTENSIONS].map(item => base + item),
      ...SOURCE_CANDIDATES.map(item => base + item), path.posix.join(base, "index.md")];
  return candidates.find(item => pathSet.has(item)) || null;
}

function moduleStem(relativePath) {
  let name = path.posix.basename(relativePath).toLowerCase();
  name = name.replace(/\.[^.]+$/, "");
  name = name.replace(/[._-](?:test|spec)$/, "");
  name = name.replace(/^test[_-]/, "");
  return name;
}

function inferTestOwners(testPath, sourceFiles) {
  const stem = moduleStem(testPath);
  if (!stem) return [];
  const candidates = sourceFiles.filter(file => path.posix.basename(file.path).replace(/\.[^.]+$/, "").toLowerCase() === stem);
  candidates.sort((a, b) => {
    const testExtension = path.posix.extname(testPath).toLowerCase().replace(".test", "");
    const aSame = path.posix.extname(a.path).toLowerCase() === testExtension ? 1 : 0;
    const bSame = path.posix.extname(b.path).toLowerCase() === testExtension ? 1 : 0;
    if (aSame !== bSame) return bSame - aSame;
    const aPreferred = /^(src|lib|local|app|packages)\//i.test(a.path) ? 1 : 0;
    const bPreferred = /^(src|lib|local|app|packages)\//i.test(b.path) ? 1 : 0;
    if (aPreferred !== bPreferred) return bPreferred - aPreferred;
    return a.path.localeCompare(b.path);
  });
  return candidates.slice(0, 3).map(item => item.path);
}

function validIndexedFile(root, item) {
  if (!item || typeof item !== "object" || Array.isArray(item)
    || !safeIndexedPath(item.path)
    || !(SOURCE_EXTENSIONS.has(path.posix.extname(item.path).toLowerCase())
      || DOC_EXTENSIONS.has(path.posix.extname(item.path).toLowerCase()))
    || !Number.isInteger(item.size) || item.size < 0 || item.size > 512 * 1024
    || typeof item.sha256 !== "string" || !HASH_PATTERN.test(item.sha256)
    || !["source", "test", "documentation"].includes(item.role)
    || !Array.isArray(item.symbols) || item.symbols.length > MAX_SYMBOLS_PER_FILE) return null;

  const symbols = [];
  for (const symbol of item.symbols) {
    if (!symbol || typeof symbol !== "object" || Array.isArray(symbol)
      || !IDENTIFIER_PATTERN.test(symbol.name || "") || !SYMBOL_KINDS.has(symbol.kind)
      || !Number.isInteger(symbol.line) || symbol.line < 1 || symbol.line > 10_000_000
      || typeof symbol.signature !== "string" || symbol.signature.length > 160
      || symbol.signature !== symbol.kind + " " + symbol.name) return null;
    symbols.push({ name: symbol.name, kind: symbol.kind, line: symbol.line, signature: symbol.signature });
  }

  const importSpecifiers = uniqueStrings(item.importSpecifiers, MAX_SPECIFIERS_PER_FILE, 300);
  const imports = uniqueStrings(item.imports, MAX_RELATIONS_PER_FILE);
  const importedBy = uniqueStrings(item.importedBy, MAX_RELATIONS_PER_FILE);
  const docLinkTargets = uniqueStrings(item.docLinkTargets, MAX_RELATIONS_PER_FILE, 300);
  const documentationReferences = uniqueStrings(item.documentationReferences, MAX_RELATIONS_PER_FILE);
  const referencedByDocs = uniqueStrings(item.referencedByDocs, MAX_RELATIONS_PER_FILE);
  const testOwners = uniqueStrings(item.testOwners, 8);
  const ownedTests = uniqueStrings(item.ownedTests, MAX_RELATIONS_PER_FILE);
  if (!importSpecifiers || !imports || !importedBy || !docLinkTargets
    || !documentationReferences || !referencedByDocs || !testOwners || !ownedTests
    || importSpecifiers.some(value => !isLocalSpecifier(value))
    || docLinkTargets.some(value => !isLocalSpecifier(value))) return null;
  return {
    path: item.path, size: item.size, sha256: item.sha256, role: item.role, symbols,
    importSpecifiers, imports, importedBy, docLinkTargets, documentationReferences,
    referencedByDocs, testOwners, ownedTests
  };
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
  for (const raw of value.files) {
    const file = validIndexedFile(root, raw);
    if (!file || seenPaths.has(file.path)) return null;
    seenPaths.add(file.path);
    totalSymbols += file.symbols.length;
    if (totalSymbols > MAX_TOTAL_SYMBOLS) return null;
    files.push(file);
  }
  const byPath = new Map(files.map(file => [file.path, file]));
  for (const file of files) {
    const relations = [
      ...file.imports.map(target => [target, ["source", "test"]]),
      ...file.importedBy.map(target => [target, ["source", "test"]]),
      ...file.documentationReferences.map(target => [target, null]),
      ...file.referencedByDocs.map(target => [target, ["documentation"]]),
      ...file.testOwners.map(target => [target, ["source"]]),
      ...file.ownedTests.map(target => [target, ["test"]])
    ];
    for (const [target, expectedRoles] of relations) {
      const linked = byPath.get(target);
      if (!linked || (expectedRoles && !expectedRoles.includes(linked.role))) return null;
    }
    if (file.role !== "documentation" && file.docLinkTargets.length) return null;
    if (file.role === "documentation" && (file.symbols.length || file.importSpecifiers.length || file.imports.length)) return null;
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

function countRelations(files, field) {
  return files.reduce((total, file) => total + file[field].length, 0);
}

async function readGoModulePath(root) {
  try {
    const goMod = await readWorkspaceFile(root, "go.mod");
    const match = /^\s*module\s+([^\s]+)/m.exec(goMod.content);
    return match?.[1] || null;
  } catch {
    return null;
  }
}

export async function getWorkspaceIndexStatus(inputRoot) {
  const root = await resolveWorkspace(inputRoot);
  const index = await readIndex(root);
  if (!index) {
    return {
      available: false, root, indexedAt: null, fileCount: 0, symbolCount: 0,
      documentationFileCount: 0, testFileCount: 0, importEdgeCount: 0,
      documentationReferenceCount: 0, testOwnershipCount: 0,
      indexedBytes: 0, truncated: false, reason: "not_built_or_invalid"
    };
  }
  return {
    available: true, root, indexedAt: index.indexedAt, fileCount: index.files.length,
    symbolCount: index.symbolCount,
    documentationFileCount: index.files.filter(file => file.role === "documentation").length,
    testFileCount: index.files.filter(file => file.role === "test").length,
    importEdgeCount: countRelations(index.files, "imports"),
    documentationReferenceCount: countRelations(index.files, "documentationReferences"),
    testOwnershipCount: countRelations(index.files, "testOwners"),
    indexedBytes: index.files.reduce((sum, file) => sum + file.size, 0),
    truncated: index.truncated
  };
}

export async function refreshWorkspaceIndex(inputRoot) {
  const root = await resolveWorkspace(inputRoot);
  await ensureWorkspaceStateDirectory(root);
  const previous = await readIndex(root);
  const candidates = await listWorkspaceIndexCandidates(root);
  const previousByPath = new Map((previous?.files || []).map(file => [file.path, file]));
  const nextFiles = [];
  const candidateFiles = candidates.files;
  const currentPaths = new Set(candidateFiles.map(file => file.path));
  const removedFiles = [...previousByPath.keys()].filter(filePath => !currentPaths.has(filePath)).length;
  let parsedFiles = 0;
  let reusedFiles = 0;
  let skippedFiles = 0;
  let indexedBytes = 0;
  let truncated = candidates.truncated || candidateFiles.length > MAX_INDEX_CANDIDATES;
  let totalSymbols = 0;

  for (let index = 0; index < candidateFiles.length; index += 1) {
    const candidate = candidateFiles[index];
    if (index >= MAX_INDEX_CANDIDATES) {
      skippedFiles += candidateFiles.length - index;
      truncated = true;
      break;
    }
    if (indexedBytes + candidate.size > MAX_INDEX_SCAN_BYTES) {
      skippedFiles += candidateFiles.length - index;
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
      skippedFiles += candidateFiles.length - index;
      truncated = true;
      break;
    }
    indexedBytes += source.bytes;

    const role = classifyRole(candidate.path, candidate.role);
    const cached = previousByPath.get(candidate.path);
    let symbols = [];
    let importSpecifiers = [];
    let docLinkTargets = [];
    if (cached && cached.sha256 === source.sha256 && cached.size === source.bytes && cached.role === role) {
      symbols = cached.symbols;
      importSpecifiers = cached.importSpecifiers;
      docLinkTargets = cached.docLinkTargets;
      reusedFiles += 1;
    } else {
      symbols = role === "documentation" ? [] : parseDeclarations(candidate.path, source.content);
      importSpecifiers = role === "documentation" ? [] : extractImportSpecifiers(candidate.path, source.content);
      docLinkTargets = role === "documentation" ? extractDocumentationTargets(source.content) : [];
      parsedFiles += 1;
    }
    if (symbols.length > MAX_TOTAL_SYMBOLS - totalSymbols) {
      symbols = symbols.slice(0, Math.max(0, MAX_TOTAL_SYMBOLS - totalSymbols));
      truncated = true;
    }
    totalSymbols += symbols.length;
    nextFiles.push({
      path: candidate.path, size: source.bytes, sha256: source.sha256, role, symbols,
      importSpecifiers, imports: [], importedBy: [],
      docLinkTargets, documentationReferences: [], referencedByDocs: [],
      testOwners: [], ownedTests: []
    });
  }

  const fileByPath = new Map(nextFiles.map(file => [file.path, file]));
  const pathSet = new Set(fileByPath.keys());
  const roles = new Map(nextFiles.map(file => [file.path, file.role]));
  const sourcePaths = nextFiles.filter(file => file.role === "source");
  const goModulePath = await readGoModulePath(root);

  for (const file of nextFiles) {
    if (file.role === "documentation") {
      file.documentationReferences = [...new Set(file.docLinkTargets
        .map(target => resolveDocumentationTarget(file.path, target, pathSet))
        .filter(Boolean))].slice(0, MAX_RELATIONS_PER_FILE);
      continue;
    }
    const resolved = [];
    for (const specifier of file.importSpecifiers) {
      const matches = resolveSourceImport(file.path, specifier, pathSet, roles, goModulePath);
      for (const target of matches) {
        const targetRole = roles.get(target);
        const permittedRole = file.role === "test"
          ? (targetRole === "source" || targetRole === "test")
          : targetRole === "source";
        if (target !== file.path && permittedRole && !resolved.includes(target)) resolved.push(target);
      }
      if (resolved.length >= MAX_RELATIONS_PER_FILE) break;
    }
    file.imports = resolved.slice(0, MAX_RELATIONS_PER_FILE);
  }

  for (const file of nextFiles) {
    for (const target of file.imports) {
      const linked = fileByPath.get(target);
      if (linked && !linked.importedBy.includes(file.path) && linked.importedBy.length < MAX_RELATIONS_PER_FILE) {
        linked.importedBy.push(file.path);
      }
    }
    for (const target of file.documentationReferences) {
      const linked = fileByPath.get(target);
      if (linked && !linked.referencedByDocs.includes(file.path) && linked.referencedByDocs.length < MAX_RELATIONS_PER_FILE) {
        linked.referencedByDocs.push(file.path);
      }
    }
  }

  const testFiles = nextFiles.filter(file => file.role === "test");
  for (const testFile of testFiles) {
    let owners = inferTestOwners(testFile.path, sourcePaths);
    if (!owners.length) {
      owners = testFile.imports.filter(target => roles.get(target) === "source").slice(0, 3);
    }
    testFile.testOwners = owners.slice(0, 3);
    for (const ownerPath of testFile.testOwners) {
      const owner = fileByPath.get(ownerPath);
      if (owner && !owner.ownedTests.includes(testFile.path) && owner.ownedTests.length < MAX_RELATIONS_PER_FILE) {
        owner.ownedTests.push(testFile.path);
      }
    }
  }

  for (const file of nextFiles) {
    file.importedBy.sort();
    file.documentationReferences.sort();
    file.referencedByDocs.sort();
    file.testOwners.sort();
    file.ownedTests.sort();
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
    documentationFileCount: nextFiles.filter(file => file.role === "documentation").length,
    testFileCount: nextFiles.filter(file => file.role === "test").length,
    importEdgeCount: countRelations(nextFiles, "imports"),
    documentationReferenceCount: countRelations(nextFiles, "documentationReferences"),
    testOwnershipCount: countRelations(nextFiles, "testOwners"),
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
  const indexByPath = new Map(index.files.map(file => [file.path, file]));
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
  let validationLimitReached = false;

  async function isCurrent(relativePath) {
    if (validatedPaths.has(relativePath)) return validatedPaths.get(relativePath);
    if (validationCount >= MAX_SEARCH_VALIDATIONS) {
      validationLimitReached = true;
      return false;
    }
    validationCount += 1;
    const indexed = indexByPath.get(relativePath);
    if (!indexed) {
      validatedPaths.set(relativePath, false);
      return false;
    }
    try {
      const current = await readWorkspaceFile(root, relativePath);
      const matches = current.sha256 === indexed.sha256 && current.bytes === indexed.size;
      validatedPaths.set(relativePath, matches);
      if (!matches) staleFiles += 1;
      return matches;
    } catch {
      validatedPaths.set(relativePath, false);
      staleFiles += 1;
      return false;
    }
  }

  async function currentRelations(relativePaths, maximum = 8) {
    const result = [];
    for (const relativePath of relativePaths.slice(0, maximum)) {
      if (await isCurrent(relativePath)) result.push(relativePath);
    }
    return result;
  }

  for (const item of ranked) {
    if (selected.length >= limit) break;
    if (!(await isCurrent(item.path))) continue;
    const file = indexByPath.get(item.path);
    const relations = {
      imports: await currentRelations(file.imports, 6),
      importedBy: await currentRelations(file.importedBy, 6),
      documentationReferences: await currentRelations(file.documentationReferences, 6),
      referencedByDocs: await currentRelations(file.referencedByDocs, 6),
      testOwners: await currentRelations(file.testOwners, 4),
      ownedTests: await currentRelations(file.ownedTests, 6)
    };
    selected.push({ ...item, ...relations, role: file.role });
  }

  return {
    available: true, root, query, results: selected, staleFiles,
    truncated: index.truncated || validationLimitReached || ranked.length > selected.length && validationLimitReached,
    indexedAt: index.indexedAt
  };
}

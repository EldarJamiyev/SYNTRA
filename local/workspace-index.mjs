import fs from "node:fs/promises";
import path from "node:path";
import crypto from "node:crypto";
import ts from "typescript";
import {
  ensureWorkspaceStateDirectory,
  getWorkspaceStateDirectory,
  listWorkspaceIndexCandidates,
  isWorkspacePathExcluded,
  readWorkspaceFile,
  resolveWorkspace,
  searchWorkspace
} from "./workspace.mjs";

const INDEX_FILENAME = "workspace-index-v3.json";
const INDEX_SCHEMA_VERSION = 3;
const SEMANTIC_CACHE_FILENAME = "workspace-semantic-v2.json";
const MAX_SEMANTIC_CACHE_BYTES = 20 * 1024 * 1024;
const MAX_SEMANTIC_FILES = 64;
const MAX_SEMANTIC_CHUNKS = 256;
const MAX_SEMANTIC_CHUNKS_PER_FILE = 16;
const MAX_SEMANTIC_CHUNK_CHARS = 2400;
const MAX_SEMANTIC_CHUNK_LINES = 48;
const SEMANTIC_CHUNK_OVERLAP_LINES = 4;
const MAX_EMBEDDING_BATCH = 32;
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

function createTypeScriptSourceFile(relativePath, content) {
  const extension = path.posix.extname(relativePath).toLowerCase();
  const scriptKind = extension === ".tsx" ? ts.ScriptKind.TSX
    : extension === ".jsx" ? ts.ScriptKind.JSX
      : [".js", ".mjs", ".cjs"].includes(extension) ? ts.ScriptKind.JS : ts.ScriptKind.TS;
  return ts.createSourceFile(relativePath, content, ts.ScriptTarget.Latest, true, scriptKind);
}

function parseDeclarations(relativePath, content) {
  const extension = path.posix.extname(relativePath).toLowerCase();
  if (!SOURCE_EXTENSIONS.has(extension)) return [];
  const symbols = [];
  const seen = new Set();
  function add(kind, name, line) {
    if (!SYMBOL_KINDS.has(kind) || !IDENTIFIER_PATTERN.test(name) || symbols.length >= MAX_SYMBOLS_PER_FILE) return;
    const key = kind + ":" + name + ":" + line;
    if (seen.has(key)) return;
    seen.add(key);
    symbols.push({ name, kind, line, signature: kind + " " + name });
  }

  if (JS_EXTENSIONS.has(extension)) {
    const sourceFile = createTypeScriptSourceFile(relativePath, content);
    function addNamedNode(node, kind) {
      if (node.name && ts.isIdentifier(node.name)) {
        const line = sourceFile.getLineAndCharacterOfPosition(node.getStart(sourceFile)).line + 1;
        add(kind, node.name.text, line);
      }
    }
    function visit(node) {
      if (ts.isFunctionDeclaration(node)) addNamedNode(node, "function");
      else if (ts.isClassDeclaration(node)) addNamedNode(node, "class");
      else if (ts.isInterfaceDeclaration(node)) addNamedNode(node, "interface");
      else if (ts.isTypeAliasDeclaration(node)) addNamedNode(node, "type");
      else if (ts.isEnumDeclaration(node)) addNamedNode(node, "enum");
      else if (ts.isModuleDeclaration(node)) addNamedNode(node, "module");
      else if (ts.isMethodDeclaration(node) || ts.isMethodSignature(node)
        || ts.isGetAccessorDeclaration(node) || ts.isSetAccessorDeclaration(node)) {
        addNamedNode(node, "function");
      } else if (ts.isVariableDeclaration(node) && node.initializer) {
        const initializer = node.initializer;
        if (ts.isArrowFunction(initializer) || ts.isFunctionExpression(initializer)) addNamedNode(node, "function");
        else if (ts.isClassExpression(initializer)) addNamedNode(node, "class");
      } else if (ts.isPropertyDeclaration(node) && node.initializer
        && (ts.isArrowFunction(node.initializer) || ts.isFunctionExpression(node.initializer))) {
        addNamedNode(node, "function");
      }
      ts.forEachChild(node, visit);
    }
    visit(sourceFile);
    return symbols;
  }

  const lines = content.split(/\r?\n/);
  for (let index = 0; index < lines.length && symbols.length < MAX_SYMBOLS_PER_FILE; index += 1) {
    const line = lines[index];
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("//") || trimmed.startsWith("#") || trimmed.startsWith("*")) continue;
    let match;
    if (extension === ".py" || extension === ".pyi") {
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
        const kind = match[1] === "fn" ? "function" : match[1] === "mod" ? "module" : match[1];
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
    const sourceFile = createTypeScriptSourceFile(relativePath, content);
    function visit(node) {
      if ((ts.isImportDeclaration(node) || ts.isExportDeclaration(node))
        && node.moduleSpecifier && ts.isStringLiteralLike(node.moduleSpecifier)) {
        add(node.moduleSpecifier.text);
      } else if (ts.isCallExpression(node)) {
        const expression = node.expression;
        const isRequire = ts.isIdentifier(expression) && expression.text === "require";
        const isDynamicImport = expression.kind === ts.SyntaxKind.ImportKeyword;
        const argument = node.arguments[0];
        if ((isRequire || isDynamicImport) && argument && ts.isStringLiteralLike(argument)) add(argument.text);
      }
      ts.forEachChild(node, visit);
    }
    visit(sourceFile);
  } else if (extension === ".py" || extension === ".pyi") {
    const pattern = /^\s*(?:from\s+([.\w]+)\s+import\b|import\s+([\w.]+))/gm;
    for (const match of content.matchAll(pattern)) add(match[1] || match[2]);
  } else if (extension === ".go") {
    const blocks = content.match(/\bimport\s*\([\s\S]*?\)/g) || [];
    for (const block of blocks) for (const match of block.matchAll(/["']([^"']+)["']/g)) add(match[1]);
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

function extractCallSpecifiers(relativePath, content) {
  if (!JS_EXTENSIONS.has(path.posix.extname(relativePath).toLowerCase())) return [];
  const sourceFile = createTypeScriptSourceFile(relativePath, content);
  const result = [];
  const seen = new Set();
  function visit(node) {
    if (ts.isCallExpression(node)) {
      const expression = node.expression;
      const name = ts.isIdentifier(expression) ? expression.text
        : ts.isPropertyAccessExpression(expression) && ts.isIdentifier(expression.name) ? expression.name.text : "";
      if (IDENTIFIER_PATTERN.test(name) && !seen.has(name) && result.length < MAX_SPECIFIERS_PER_FILE) {
        seen.add(name);
        result.push(name);
      }
    }
    ts.forEachChild(node, visit);
  }
  visit(sourceFile);
  return result;
}

function extractTypeSpecifiers(relativePath, content) {
  if (!JS_EXTENSIONS.has(path.posix.extname(relativePath).toLowerCase())) return [];
  const sourceFile = createTypeScriptSourceFile(relativePath, content);
  const result = [];
  const seen = new Set();
  const add = name => {
    if (IDENTIFIER_PATTERN.test(name || "") && !seen.has(name)
      && result.length < MAX_SPECIFIERS_PER_FILE) {
      seen.add(name);
      result.push(name);
    }
  };
  function rightmostName(node) {
    if (ts.isIdentifier(node)) return node.text;
    if (ts.isQualifiedName(node)) return node.right.text;
    if (ts.isPropertyAccessExpression(node)) return node.name.text;
    return "";
  }
  function visit(node) {
    if (ts.isTypeReferenceNode(node)) add(rightmostName(node.typeName));
    else if (ts.isExpressionWithTypeArguments(node)) add(rightmostName(node.expression));
    else if (ts.isTypeQueryNode(node)) add(rightmostName(node.exprName));
    ts.forEachChild(node, visit);
  }
  visit(sourceFile);
  return result;
}

function extractImportedTypeBindings(relativePath, content) {
  if (!JS_EXTENSIONS.has(path.posix.extname(relativePath).toLowerCase())) return [];
  const sourceFile = createTypeScriptSourceFile(relativePath, content);
  const bindings = [];
  const seen = new Set();
  for (const statement of sourceFile.statements) {
    if (!ts.isImportDeclaration(statement) || !statement.moduleSpecifier
      || !ts.isStringLiteralLike(statement.moduleSpecifier)) continue;
    const named = statement.importClause?.namedBindings;
    if (!named || !ts.isNamedImports(named)) continue;
    for (const element of named.elements) {
      const localName = element.name.text;
      const importedName = element.propertyName?.text || localName;
      const key = [statement.moduleSpecifier.text, localName, importedName].join(":");
      if (seen.has(key) || bindings.length >= MAX_SPECIFIERS_PER_FILE) continue;
      seen.add(key);
      bindings.push({
        moduleSpecifier: statement.moduleSpecifier.text,
        localName,
        importedName
      });
    }
  }
  return bindings;
}

export function analyzeSourceFile(relativePath, content) {
  return {
    symbols: parseDeclarations(relativePath, content),
    importSpecifiers: extractImportSpecifiers(relativePath, content),
    callSpecifiers: extractCallSpecifiers(relativePath, content),
    typeSpecifiers: extractTypeSpecifiers(relativePath, content),
    importedTypeBindings: extractImportedTypeBindings(relativePath, content)
  };
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
    segments = [baseDir, specifier.slice(4)].filter(Boolean);
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

function validIndexedFile(item) {
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
  const callSpecifiers = uniqueStrings(item.callSpecifiers, MAX_SPECIFIERS_PER_FILE, 120);
  const imports = uniqueStrings(item.imports, MAX_RELATIONS_PER_FILE);
  const importedBy = uniqueStrings(item.importedBy, MAX_RELATIONS_PER_FILE);
  const calls = uniqueStrings(item.calls, MAX_RELATIONS_PER_FILE);
  const calledBy = uniqueStrings(item.calledBy, MAX_RELATIONS_PER_FILE);
  const docLinkTargets = uniqueStrings(item.docLinkTargets, MAX_RELATIONS_PER_FILE, 300);
  const documentationReferences = uniqueStrings(item.documentationReferences, MAX_RELATIONS_PER_FILE);
  const referencedByDocs = uniqueStrings(item.referencedByDocs, MAX_RELATIONS_PER_FILE);
  const testOwners = uniqueStrings(item.testOwners, 8);
  const ownedTests = uniqueStrings(item.ownedTests, MAX_RELATIONS_PER_FILE);
  if (!importSpecifiers || !callSpecifiers || !imports || !importedBy || !calls || !calledBy || !docLinkTargets
    || !documentationReferences || !referencedByDocs || !testOwners || !ownedTests
    || importSpecifiers.some(value => !isLocalSpecifier(value))
    || docLinkTargets.some(value => !isLocalSpecifier(value))) return null;
  return {
    path: item.path, size: item.size, sha256: item.sha256, role: item.role, symbols,
    importSpecifiers, callSpecifiers, imports, importedBy, calls, calledBy, docLinkTargets, documentationReferences,
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
    const file = validIndexedFile(raw);
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
      ...file.calls.map(target => [target, ["source"]]),
      ...file.calledBy.map(target => [target, ["source", "test"]]),
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
      documentationReferenceCount: 0, testOwnershipCount: 0, callEdgeCount: 0,
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
    callEdgeCount: countRelations(index.files, "calls"),
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
    let callSpecifiers = [];
    let docLinkTargets = [];
    if (cached && cached.sha256 === source.sha256 && cached.size === source.bytes && cached.role === role) {
      symbols = cached.symbols;
      importSpecifiers = cached.importSpecifiers;
      callSpecifiers = cached.callSpecifiers || [];
      docLinkTargets = cached.docLinkTargets;
      reusedFiles += 1;
    } else {
      symbols = role === "documentation" ? [] : parseDeclarations(candidate.path, source.content);
      importSpecifiers = role === "documentation" ? [] : extractImportSpecifiers(candidate.path, source.content);
      callSpecifiers = role === "documentation" ? [] : extractCallSpecifiers(candidate.path, source.content);
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
      importSpecifiers, callSpecifiers, imports: [], importedBy: [], calls: [], calledBy: [],
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
    for (const callName of file.callSpecifiers) {
      const matchingImports = file.imports.filter(target =>
        fileByPath.get(target)?.symbols.some(symbol => symbol.name === callName)
      );
      if (matchingImports.length === 1) {
        const target = matchingImports[0];
        if (!file.calls.includes(target) && file.calls.length < MAX_RELATIONS_PER_FILE) file.calls.push(target);
        const linked = fileByPath.get(target);
        if (linked && !linked.calledBy.includes(file.path) && linked.calledBy.length < MAX_RELATIONS_PER_FILE) {
          linked.calledBy.push(file.path);
        }
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
    file.calls.sort();
    file.calledBy.sort();
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
    callEdgeCount: countRelations(nextFiles, "calls"),
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
      if (await isWorkspacePathExcluded(root, relativePath)) {
        validatedPaths.set(relativePath, false);
        staleFiles += 1;
        return false;
      }
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
      calls: await currentRelations(file.calls, 6),
      calledBy: await currentRelations(file.calledBy, 6),
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


function buildSemanticContentChunks(content) {
  const lines = String(content).split(/\r?\n/);
  const chunks = [];
  let start = 0;

  while (start < lines.length) {
    let cursor = start;
    let characterCount = 0;
    const selectedLines = [];
    while (cursor < lines.length && cursor - start < MAX_SEMANTIC_CHUNK_LINES) {
      const line = lines[cursor];
      const separatorLength = selectedLines.length ? 1 : 0;
      const remaining = MAX_SEMANTIC_CHUNK_CHARS - characterCount - separatorLength;
      if (remaining <= 0) break;
      if (line.length > remaining) {
        if (!selectedLines.length) {
          selectedLines.push(line.slice(0, MAX_SEMANTIC_CHUNK_CHARS));
          cursor += 1;
        }
        break;
      }
      selectedLines.push(line);
      characterCount += line.length + separatorLength;
      cursor += 1;
    }

    if (!selectedLines.length) {
      selectedLines.push(lines[start].slice(0, MAX_SEMANTIC_CHUNK_CHARS));
      cursor = start + 1;
    }
    chunks.push({
      lineStart: start + 1,
      lineEnd: Math.max(start + 1, cursor),
      source: selectedLines.join("\n").slice(0, MAX_SEMANTIC_CHUNK_CHARS)
    });
    if (cursor >= lines.length) break;
    start = Math.max(start + 1, cursor - SEMANTIC_CHUNK_OVERLAP_LINES);
  }

  if (chunks.length <= MAX_SEMANTIC_CHUNKS_PER_FILE) {
    return { chunks, totalChunks: chunks.length, sampled: false };
  }
  const sampled = Array.from({ length: MAX_SEMANTIC_CHUNKS_PER_FILE }, (_, index) => {
    const selectedIndex = Math.round(index * (chunks.length - 1) / (MAX_SEMANTIC_CHUNKS_PER_FILE - 1));
    return chunks[selectedIndex];
  });
  return { chunks: sampled, totalChunks: chunks.length, sampled: true };
}

function semanticEmbeddingText(file, chunk) {
  const nearbySymbols = file.symbols
    .filter(symbol => symbol.line >= chunk.lineStart && symbol.line <= chunk.lineEnd)
    .slice(0, 24)
    .map(symbol => symbol.kind + " " + symbol.name + " (line " + symbol.line + ")")
    .join("\n");
  const symbols = nearbySymbols || file.symbols.slice(0, 12)
    .map(symbol => symbol.kind + " " + symbol.name)
    .join("\n");
  const relations = [
    "Imports: " + file.imports.slice(0, 12).join(", "),
    "Callers: " + file.calledBy.slice(0, 12).join(", "),
    "Tests: " + file.ownedTests.slice(0, 8).join(", ")
  ].join("\n");
  return [
    "Path: " + file.path,
    "Role: " + file.role,
    "Language: " + path.posix.extname(file.path).slice(1),
    "Chunk lines: " + chunk.lineStart + "-" + chunk.lineEnd,
    "Symbols: " + symbols,
    relations,
    "Source:\n" + chunk.source
  ].join("\n");
}

function validSemanticVector(value, dimension = null) {
  return Array.isArray(value)
    && value.length >= 8 && value.length <= 4096
    && (dimension === null || value.length === dimension)
    && value.every(item => typeof item === "number" && Number.isFinite(item));
}

function semanticTextHash(value) {
  return crypto.createHash("sha256").update(value, "utf8").digest("hex");
}

async function semanticCachePath(root, create = false) {
  const directory = create
    ? await ensureWorkspaceStateDirectory(root)
    : await getWorkspaceStateDirectory(root);
  return path.join(directory, SEMANTIC_CACHE_FILENAME);
}

async function readSemanticCache(root, model) {
  const target = await semanticCachePath(root, false);
  let info;
  try { info = await fs.lstat(target); }
  catch (error) {
    if (error?.code === "ENOENT") return { dimension: null, entries: [] };
    throw error;
  }
  if (info.isSymbolicLink() || !info.isFile()) {
    throw new Error("The semantic source cache must be a regular file, not a symbolic link.");
  }
  if (info.size > MAX_SEMANTIC_CACHE_BYTES) {
    throw new Error("The semantic source cache exceeds its 20 MB storage limit.");
  }
  let value;
  try { value = JSON.parse(await fs.readFile(target, "utf8")); }
  catch {
    throw new Error("The semantic source cache contains invalid JSON and was not overwritten.");
  }
  if (!value || typeof value !== "object" || Array.isArray(value)
    || value.schemaVersion !== 2 || value.root !== root
    || typeof value.model !== "string" || !Array.isArray(value.entries)
    || value.entries.length > MAX_SEMANTIC_CHUNKS) {
    throw new Error("The semantic source cache is invalid or belongs to another workspace.");
  }
  if (value.model !== model) return { dimension: null, entries: [] };
  const dimension = Number.isInteger(value.dimension) && value.dimension >= 8 && value.dimension <= 4096
    ? value.dimension : null;
  const seen = new Set();
  const entries = [];
  for (const entry of value.entries) {
    const key = entry && typeof entry.path === "string" && Number.isInteger(entry.chunkIndex)
      ? entry.path + "\\0" + entry.chunkIndex : "";
    if (!entry || typeof entry !== "object" || !safeIndexedPath(entry.path)
      || !Number.isInteger(entry.chunkIndex) || entry.chunkIndex < 0
      || entry.chunkIndex >= MAX_SEMANTIC_CHUNKS_PER_FILE
      || seen.has(key) || !HASH_PATTERN.test(entry.textHash || "")
      || !validSemanticVector(entry.embedding, dimension)) {
      throw new Error("The semantic source cache contains an invalid chunk vector entry.");
    }
    seen.add(key);
    entries.push({
      path: entry.path, chunkIndex: entry.chunkIndex,
      textHash: entry.textHash, embedding: entry.embedding
    });
  }
  return { dimension, entries };
}
async function writeSemanticCache(root, model, dimension, entries) {
  if (!Array.isArray(entries) || entries.length > MAX_SEMANTIC_CHUNKS) {
    throw new Error("The semantic source cache can contain at most 256 chunk vectors.");
  }
  const target = await semanticCachePath(root, true);
  let existing;
  try {
    existing = await fs.lstat(target);
    if (existing.isSymbolicLink() || !existing.isFile()) {
      throw new Error("The semantic source cache must be a regular file, not a symbolic link.");
    }
  } catch (error) {
    if (error?.code !== "ENOENT") throw error;
  }
  const serialized = JSON.stringify({
    schemaVersion: 2, root, model, dimension,
    updatedAt: new Date().toISOString(), entries
  });
  if (Buffer.byteLength(serialized, "utf8") > MAX_SEMANTIC_CACHE_BYTES) {
    throw new Error("The semantic source cache exceeded its 20 MB storage limit.");
  }
  const temporary = target + "." + process.pid + "." + crypto.randomBytes(4).toString("hex") + ".tmp";
  await fs.writeFile(temporary, serialized + "\n", { encoding: "utf8", flag: "wx" });
  try { await fs.rename(temporary, target); }
  catch (error) {
    await fs.rm(temporary, { force: true });
    throw error;
  }
}

const NATIVE_SEMANTIC_EMBEDDING_MODEL = "syntra-native";

async function requestSourceEmbeddings(fetchImpl, model, inputs) {
  if (model !== NATIVE_SEMANTIC_EMBEDDING_MODEL) {
    throw new Error("SYNTRA source indexing accepts only embeddings from its own trained checkpoint.");
  }
  if (!Array.isArray(inputs) || inputs.length < 1 || inputs.length > MAX_EMBEDDING_BATCH) {
    throw new Error("Native semantic embedding batches must contain 1-32 texts.");
  }
  const aiPort = Number(process.env.SYNTRA_AI_PORT || 3002);
  const response = await fetchImpl(`http://127.0.0.1:${aiPort}/model/embed`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    redirect: "error",
    body: JSON.stringify({ texts: inputs, device: "auto" }),
    signal: AbortSignal.timeout(120000)
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(String(payload.error || "SYNTRA native embedding service returned HTTP " + response.status + "."));
  if (!/^[a-f0-9]{64}$/i.test(payload.checkpoint_identity || "")) {
    throw new Error("SYNTRA native runtime did not identify the checkpoint that generated these embeddings.");
  }
  if (!payload || !Array.isArray(payload.embeddings) || payload.embeddings.length !== inputs.length) {
    throw new Error("SYNTRA native runtime returned an invalid semantic embedding response.");
  }
  const dimension = payload.embeddings[0]?.length;
  if (!Number.isInteger(dimension) || dimension < 8 || dimension > 4096
    || !payload.embeddings.every(vector => validSemanticVector(vector, dimension))) {
    throw new Error("SYNTRA native runtime returned invalid or inconsistent semantic vectors.");
  }
  return { dimension, embeddings: payload.embeddings, checkpointIdentity: payload.checkpoint_identity };
}

function semanticCosineSimilarity(left, right) {
  if (!validSemanticVector(left) || !validSemanticVector(right) || left.length !== right.length) return null;
  let dot = 0, leftNorm = 0, rightNorm = 0;
  for (let index = 0; index < left.length; index += 1) {
    dot += left[index] * right[index];
    leftNorm += left[index] * left[index];
    rightNorm += right[index] * right[index];
  }
  if (!leftNorm || !rightNorm) return null;
  return dot / Math.sqrt(leftNorm * rightNorm);
}

function lexicalSourceScore(file, query) {
  const words = [...new Set(query.toLowerCase().match(/[\p{L}\p{N}_$-]{2,}/gu) || [])].slice(0, 16);
  if (!words.length) return 0;
  const haystack = [
    file.path, ...file.symbols.map(symbol => symbol.name + " " + symbol.signature),
    ...file.imports, ...file.importedBy, ...file.calls, ...file.calledBy
  ].join(" ").toLowerCase();
  return words.reduce((score, word) => score + Math.min(5, haystack.split(word).length - 1), 0);
}

function semanticMetadata(file) {
  return {
    path: file.path,
    role: file.role,
    language: path.posix.extname(file.path).slice(1) || "text",
    symbols: file.symbols.slice(0, 12).map(symbol => ({
      name: symbol.name, kind: symbol.kind, line: symbol.line
    })),
    imports: file.imports.slice(0, 8),
    importedBy: file.importedBy.slice(0, 8),
    calls: file.calls.slice(0, 8),
    calledBy: file.calledBy.slice(0, 8),
    ownedTests: file.ownedTests.slice(0, 6),
    testOwners: file.testOwners.slice(0, 4)
  };
}

export async function searchWorkspaceSemantic(inputRoot, query, limit = 8, options = {}) {
  const root = await resolveWorkspace(inputRoot);
  if (typeof query !== "string" || query.trim().length < 2 || query.length > 300) {
    throw new Error("Semantic source search query must contain 2-300 characters.");
  }
  if (!Number.isInteger(limit) || limit < 1 || limit > 20) {
    throw new Error("Semantic source search limit must be an integer from 1 to 20.");
  }
  const normalized = query.trim();
  const model = String(options.embedModel ?? NATIVE_SEMANTIC_EMBEDDING_MODEL).trim();
  if (model !== NATIVE_SEMANTIC_EMBEDDING_MODEL) {
    throw new Error("SYNTRA semantic source search uses only the native checkpoint embedding model.");
  }
  const fetchImpl = options.fetchImpl ?? globalThis.fetch;
  const index = await readIndex(root);
  if (!index) {
    const lexical = await searchWorkspace(root, normalized, limit);
    return {
      available: false, root, query: normalized, retrievalMode: "lexical_fallback",
      embeddingModel: model, semanticCoverage: 0, semanticChunkCoverage: 0,
      reason: "workspace_index_unavailable", results: lexical.results
    };
  }

  const sourceFiles = index.files.filter(file => file.role === "source" || file.role === "test");
  const candidates = [...sourceFiles].sort((left, right) => {
    const leftStructure = left.symbols.length + left.imports.length * 2
      + left.calledBy.length * 2 + left.ownedTests.length;
    const rightStructure = right.symbols.length + right.imports.length * 2
      + right.calledBy.length * 2 + right.ownedTests.length;
    const leftScore = lexicalSourceScore(left, normalized) * 6 + Math.min(40, leftStructure);
    const rightScore = lexicalSourceScore(right, normalized) * 6 + Math.min(40, rightStructure);
    return rightScore - leftScore || left.path.localeCompare(right.path);
  }).slice(0, MAX_SEMANTIC_FILES);
  if (!candidates.length) {
    return {
      available: true, root, query: normalized, retrievalMode: "semantic",
      embeddingModel: model, semanticCoverage: 0, semanticChunkCoverage: 0,
      results: [], truncated: false
    };
  }

  try {
    const currentFiles = [];
    let staleFiles = 0;
    for (const indexed of candidates) {
      if (!(await safeIndexedPath(indexed.path)) || await isWorkspacePathExcluded(root, indexed.path)) {
        staleFiles += 1;
        continue;
      }
      try {
        const current = await readWorkspaceFile(root, indexed.path);
        if (current.sha256 !== indexed.sha256 || current.bytes !== indexed.size) {
          staleFiles += 1;
          continue;
        }
        const built = buildSemanticContentChunks(current.content);
        const chunks = built.chunks.map((chunk, chunkIndex) => {
          const embeddingText = semanticEmbeddingText(indexed, chunk);
          return {
            path: indexed.path, indexed, sha256: current.sha256, content: current.content,
            chunkIndex, lineStart: chunk.lineStart, lineEnd: chunk.lineEnd,
            source: chunk.source, textHash: semanticTextHash(embeddingText), embeddingText
          };
        });
        currentFiles.push({
          indexed, sha256: current.sha256, chunks,
          sampledChunks: built.sampled, totalChunks: built.totalChunks
        });
      } catch {
        staleFiles += 1;
      }
    }

    // Round-robin chunks across files so a few very large modules cannot consume
    // the entire embedding budget before smaller, relevant files are considered.
    const selectedChunks = [];
    const maximumRounds = Math.max(0, ...currentFiles.map(file => file.chunks.length));
    for (let round = 0; round < maximumRounds && selectedChunks.length < MAX_SEMANTIC_CHUNKS; round += 1) {
      for (const file of currentFiles) {
        const chunk = file.chunks[round];
        if (chunk) selectedChunks.push(chunk);
        if (selectedChunks.length >= MAX_SEMANTIC_CHUNKS) break;
      }
    }
    if (!selectedChunks.length) {
      return {
        available: true, root, query: normalized, retrievalMode: "semantic",
        embeddingModel: model, semanticCoverage: 0, semanticChunkCoverage: 0,
        results: [], staleFiles, truncated: sourceFiles.length > candidates.length
      };
    }

    // Embed the query first so it identifies the exact native checkpoint
    // before any cached source vectors are considered for reuse.
    const queryBatch = await requestSourceEmbeddings(fetchImpl, model, [normalized]);
    const cacheModel = model + ":" + queryBatch.checkpointIdentity;
    const cache = await readSemanticCache(root, cacheModel);
    if (cache.dimension !== null && cache.dimension !== queryBatch.dimension) {
      throw new Error("The native checkpoint embedding dimension changed without a matching checkpoint identity.");
    }
    const cacheByKey = new Map(cache.entries.map(entry => [
      entry.path + "\\0" + entry.chunkIndex, entry
    ]));
    const vectorsByKey = new Map();
    const missing = [];
    for (const item of selectedChunks) {
      const key = item.path + "\\0" + item.chunkIndex;
      const cached = cacheByKey.get(key);
      if (cached && cached.textHash === item.textHash && cache.dimension
        && validSemanticVector(cached.embedding, cache.dimension)) {
        vectorsByKey.set(key, cached.embedding);
      } else {
        missing.push(item);
      }
    }

    const generated = [];
    let dimension = queryBatch.dimension;
    for (let offset = 0; offset < missing.length; offset += MAX_EMBEDDING_BATCH) {
      const batch = await requestSourceEmbeddings(
        fetchImpl, model, missing.slice(offset, offset + MAX_EMBEDDING_BATCH).map(item => item.embeddingText)
      );
      if (batch.checkpointIdentity !== queryBatch.checkpointIdentity) {
        throw new Error("The native checkpoint changed during semantic indexing; retry the search.");
      }
      if (dimension !== batch.dimension) {
        throw new Error("The native semantic model returned a different vector dimension within one search.");
      }
      generated.push(...batch.embeddings);
    }
    if (generated.length !== missing.length) {
      throw new Error("SYNTRA native semantic embedding generation returned an incomplete vector set.");
    }

    missing.forEach((item, index) => {
      vectorsByKey.set(item.path + "\\0" + item.chunkIndex, generated[index]);
    });
    const queryEmbedding = queryBatch.embeddings[0];
    const serializedEntries = selectedChunks.filter(item =>
      vectorsByKey.has(item.path + "\\0" + item.chunkIndex)
    ).map(item => ({
      path: item.path, chunkIndex: item.chunkIndex,
      textHash: item.textHash, embedding: vectorsByKey.get(item.path + "\\0" + item.chunkIndex)
    }));
    await writeSemanticCache(root, cacheModel, dimension, serializedEntries);

    const bestByPath = new Map();
    for (const item of selectedChunks) {
      const key = item.path + "\\0" + item.chunkIndex;
      const cosine = semanticCosineSimilarity(queryEmbedding, vectorsByKey.get(key));
      const score = cosine === null ? 0 : Number(cosine.toFixed(6));
      if (score <= 0) continue;
      const previous = bestByPath.get(item.path);
      if (previous && previous.score >= score) continue;
      bestByPath.set(item.path, {
        ...semanticMetadata(item.indexed),
        sha256: item.sha256,
        score,
        lineStart: item.lineStart,
        lineEnd: item.lineEnd,
        excerpt: item.source.slice(0, 1600)
      });
    }
    const results = [...bestByPath.values()]
      .sort((left, right) => right.score - left.score || left.path.localeCompare(right.path))
      .slice(0, limit);
    const semanticFileCoverage = new Set(serializedEntries.map(item => item.path)).size;
    const totalAvailableChunks = currentFiles.reduce((sum, file) => sum + file.totalChunks, 0);
    return {
      available: true, root, query: normalized, retrievalMode: "semantic",
      embeddingModel: model,
      checkpointIdentity: queryBatch.checkpointIdentity,
      semanticCoverage: semanticFileCoverage,
      semanticChunkCoverage: serializedEntries.length,
      indexedCandidates: sourceFiles.length,
      indexedFiles: currentFiles.length,
      indexedChunks: selectedChunks.length,
      staleFiles,
      truncated: sourceFiles.length > candidates.length
        || currentFiles.some(file => file.sampledChunks)
        || totalAvailableChunks > selectedChunks.length,
      results
    };
  } catch (error) {
    const lexical = await searchWorkspace(root, normalized, limit);
    return {
      available: true, root, query: normalized, retrievalMode: "lexical_fallback",
      embeddingModel: model, semanticCoverage: 0, semanticChunkCoverage: 0,
      semanticError: (error instanceof Error ? error.message : String(error)).slice(0, 240),
      results: lexical.results
    };
  }
}

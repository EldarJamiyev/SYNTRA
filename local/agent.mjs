import fs from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import crypto from "node:crypto";
import {
  inspectWorkspace, proposeFileChange, readWorkspaceFile, resolveWorkspace, runProjectCheck
} from "./workspace.mjs";

const SOURCE_EXTENSIONS = new Set([
  ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".mts", ".cts", ".py", ".pyi",
  ".html", ".css", ".scss", ".sql", ".json", ".md", ".yaml", ".yml", ".toml",
  ".sh", ".ps1", ".go", ".rs", ".java", ".cs", ".vue", ".svelte", ".astro"
]);
const SKIP_DIRS = new Set([
  ".git", ".syntra", ".venv", "venv", "env", "node_modules", "__pycache__",
  ".pytest_cache", ".mypy_cache", ".ruff_cache", "dist", "build", "coverage",
  ".next", ".nuxt", ".idea", ".vscode", "target", "vendor", "obj", "secrets",
  "credentials", "private_keys", "certificates"
]);
const INCLUDED_HIDDEN_DIRS = new Set([
  ".github", ".gitlab", ".circleci", ".husky", ".storybook", ".changeset", ".devcontainer"
]);
const SECRET_NAMES = new Set([
  ".env", ".env.local", ".env.production", ".npmrc", ".pypirc", "credentials",
  "credentials.json", "secrets.json", "id_rsa", "id_ed25519", "known_hosts"
]);
const SECRET_SUFFIXES = new Set([".pem", ".key", ".p12", ".pfx", ".keystore", ".secret"]);
const MAX_GRAPH_FILES = 5000;
const MAX_GRAPH_BYTES = 24 * 1024 * 1024;
const MAX_CANDIDATE_BYTES = 1_500_000_000;
const MAX_CANDIDATE_ENTRIES = 100_000;
const MAX_ITERATIONS = 5;
const CHECK_TIMEOUT_MS = 120_000;
const MODEL_PROMPT = [
  "You are SYNTRA's repository-aware software engineering agent.",
  "Act only on the supplied task, repository map, source excerpts, plan, and tool observations.",
  "Return exactly one JSON object; do not use markdown fences or explanations outside JSON.",
  "Schema: {\"summary\":\"...\",\"edits\":[{\"path\":\"relative/path\",\"find\":\"exact existing text\",\"replace\":\"replacement text\",\"reason\":\"...\"}]}",
  "Make a minimal, targeted patch. Each existing-file find string must exactly match once in that file.",
  "To create a new text file, use an unused relative path, set find to the empty string, and set replace to the entire new file content.",
  "Never output shell commands, install commands, scripts to execute, absolute paths, secrets, or .env changes.",
  "Do not claim tests passed. SYNTRA will run checks separately and feed failure output back to you.",
  "Use only file paths inside the selected project. Make no more than 6 file edits and keep each replacement under 12,000 characters."
].join("\n");

function sha256(text) {
  return crypto.createHash("sha256").update(text, "utf8").digest("hex");
}

function within(root, target) {
  const relative = path.relative(root, target);
  return relative === "" || (!relative.startsWith(".." + path.sep) && relative !== ".." && !path.isAbsolute(relative));
}

function isSecretPath(relative) {
  const parts = relative.split("/").map(part => part.toLowerCase());
  const base = parts.at(-1) || "";
  const extension = path.posix.extname(base);
  return parts.some(part => SECRET_NAMES.has(part) || ["secrets", "credentials", "private_keys", "certificates"].includes(part))
    || (base.startsWith(".env.") && base !== ".env.example")
    || SECRET_SUFFIXES.has(extension) || base.endsWith(".secret");
}

function normalizeAgentPath(value) {
  if (typeof value !== "string" || !value.trim()) throw new Error("Patch edit path must be a non-empty relative path.");
  const raw = value.replace(/\\/g, "/");
  if (raw.startsWith("/") || /^[a-zA-Z]:\//.test(raw)) throw new Error("Absolute patch paths are not permitted.");
  const parts = raw.split("/");
  if (parts.some(part => !part || part === "." || part === "..")) throw new Error("Patch path traversal is not permitted.");
  const normalized = parts.join("/");
  if (normalized === ".syntra" || normalized.startsWith(".syntra/")) throw new Error("The .syntra state directory is reserved.");
  if (isSecretPath(normalized)) throw new Error("Patch is blocked because the path looks secret or credential-related: " + normalized);
  const extension = path.posix.extname(normalized).toLowerCase();
  const baseName = path.posix.basename(normalized).toLowerCase();
  if (!extension && !["dockerfile", "makefile", ".gitignore", ".editorconfig"].includes(baseName)) {
    throw new Error("Agent edits must target recognized text/source/configuration files.");
  }
  if (extension && !SOURCE_EXTENSIONS.has(extension) && ![".gitignore", ".editorconfig", ".env.example"].includes(path.posix.basename(normalized).toLowerCase())) {
    throw new Error("Agent edits are restricted to recognized text/source/configuration files.");
  }
  if (path.posix.basename(normalized).toLowerCase() === ".env") throw new Error("Environment secret files are never edited by the agent.");
  return normalized;
}

function extractJsonObject(raw) {
  if (typeof raw !== "string") throw new Error("Model returned no text.");
  const start = raw.indexOf("{");
  if (start < 0) throw new Error("Model output did not contain a JSON object.");
  let depth = 0;
  let inString = false;
  let escaped = false;
  for (let index = start; index < raw.length; index += 1) {
    const char = raw[index];
    if (inString) {
      if (escaped) escaped = false;
      else if (char === "\\") escaped = true;
      else if (char === '"') inString = false;
      continue;
    }
    if (char === '"') inString = true;
    else if (char === "{") depth += 1;
    else if (char === "}") {
      depth -= 1;
      if (depth === 0) {
        try { return JSON.parse(raw.slice(start, index + 1)); }
        catch (error) { throw new Error("Model patch JSON is malformed: " + (error instanceof Error ? error.message : String(error))); }
      }
    }
  }
  throw new Error("Model output ended before the JSON object was complete.");
}

async function walkSourceFiles(root) {
  const files = [];
  let totalBytes = 0;
  async function visit(directory, depth) {
    if (depth > 12 || files.length >= MAX_GRAPH_FILES || totalBytes >= MAX_GRAPH_BYTES) return;
    let entries;
    try { entries = await fs.readdir(directory, { withFileTypes: true }); } catch { return; }
    entries.sort((a, b) => a.name.localeCompare(b.name));
    for (const entry of entries) {
      if (files.length >= MAX_GRAPH_FILES || totalBytes >= MAX_GRAPH_BYTES) break;
      const absolute = path.join(directory, entry.name);
      const relative = path.relative(root, absolute).split(path.sep).join("/");
      if (entry.isSymbolicLink()) continue;
      if (entry.isDirectory()) {
        const directoryName = entry.name.toLowerCase();
        if (!SKIP_DIRS.has(directoryName)
          && (!directoryName.startsWith(".") || INCLUDED_HIDDEN_DIRS.has(directoryName))) {
          await visit(absolute, depth + 1);
        }
        continue;
      }
      if (!entry.isFile()) continue;
      const extension = path.extname(entry.name).toLowerCase();
      if (!SOURCE_EXTENSIONS.has(extension) && ![".gitignore", ".editorconfig"].includes(entry.name.toLowerCase()) && entry.name.toLowerCase() !== "dockerfile" && entry.name.toLowerCase() !== "makefile") continue;
      if (isSecretPath(relative)) continue;
      const stat = await fs.stat(absolute).catch(() => null);
      if (!stat || stat.size <= 0 || stat.size > 256 * 1024 || totalBytes + stat.size > MAX_GRAPH_BYTES) continue;
      files.push({ path: relative, size: stat.size, extension });
      totalBytes += stat.size;
    }
  }
  await visit(root, 0);
  return { files, totalBytes, truncated: files.length >= MAX_GRAPH_FILES || totalBytes >= MAX_GRAPH_BYTES };
}

function parseImports(relative, content) {
  const result = [];
  const extension = path.posix.extname(relative).toLowerCase();
  if ([".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".vue", ".svelte", ".astro"].includes(extension)) {
    const pattern = /\bfrom\s*["']([^"']+)["']|\bimport\s*(?:\(\s*)?["']([^"']+)["']|\brequire\s*\(\s*["']([^"']+)["']/g;
    for (const match of content.matchAll(pattern)) {
      const specifier = match[1] || match[2] || match[3];
      if (specifier && specifier.startsWith(".")) result.push(specifier);
    }
    const css = /@import\s+(?:url\()?["']([^"']+)["']/g;
    for (const match of content.matchAll(css)) if (match[1] && match[1].startsWith(".")) result.push(match[1]);
  } else if (extension === ".py" || extension === ".pyi") {
    const pattern = /^\s*(?:from\s+([.\w]+)\s+import\b|import\s+([\w.]+))/gm;
    for (const match of content.matchAll(pattern)) {
      const specifier = match[1] || match[2];
      if (specifier) result.push(specifier);
    }
  }
  return [...new Set(result)].slice(0, 80);
}

function resolveImport(source, specifier, fileSet) {
  const sourceExt = path.posix.extname(source).toLowerCase();
  let base;
  if (sourceExt === ".py" || sourceExt === ".pyi") {
    if (!specifier.startsWith(".")) {
      const modulePath = specifier.replace(/\./g, "/");
      const candidates = [modulePath + ".py", modulePath + "/__init__.py"];
      return candidates.find(item => fileSet.has(item)) || null;
    }
    const dots = (specifier.match(/^\.+/) || ["."])[0].length;
    const tail = specifier.slice(dots).replace(/\./g, "/");
    let directory = path.posix.dirname(source);
    for (let count = 1; count < dots; count += 1) directory = path.posix.dirname(directory);
    base = path.posix.join(directory, tail);
    return [base + ".py", base + "/__init__.py", base + ".pyi"].find(item => fileSet.has(item)) || null;
  }
  base = path.posix.normalize(path.posix.join(path.posix.dirname(source), specifier));
  const candidates = [
    base, base + ".ts", base + ".tsx", base + ".js", base + ".jsx",
    base + ".mjs", base + ".cjs", base + ".css", base + ".scss", base + ".json",
    base + ".py", path.posix.join(base, "index.ts"), path.posix.join(base, "index.tsx"),
    path.posix.join(base, "index.js"), path.posix.join(base, "index.jsx")
  ];
  return candidates.find(item => fileSet.has(item)) || null;
}

export async function buildRepositoryGraph(inputRoot) {
  const root = await resolveWorkspace(inputRoot);
  const walked = await walkSourceFiles(root);
  const fileSet = new Set(walked.files.map(file => file.path));
  const nodes = new Map();
  for (const item of walked.files) {
    let file;
    try { file = await readWorkspaceFile(root, item.path); } catch { continue; }
    const node = {
      path: item.path, size: file.bytes, sha256: file.sha256,
      language: item.extension.slice(1) || "text", imports: [], importedBy: [], content: file.content
    };
    nodes.set(item.path, node);
  }
  for (const node of nodes.values()) {
    for (const specifier of parseImports(node.path, node.content)) {
      const resolved = resolveImport(node.path, specifier, fileSet);
      if (resolved && resolved !== node.path && nodes.has(resolved)) node.imports.push(resolved);
    }
  }
  for (const node of nodes.values()) for (const imported of node.imports) nodes.get(imported)?.importedBy.push(node.path);
  return {
    root, nodes: [...nodes.values()].map(({ content, ...node }) => node), contents: nodes,
    totalFiles: walked.files.length, totalBytes: walked.totalBytes, truncated: walked.truncated
  };
}

const STOP_WORDS = new Set([
  "the", "and", "for", "with", "from", "this", "that", "into", "make", "build",
  "change", "please", "using", "when", "where", "what", "should", "could", "would",
  "then", "than", "file", "files", "code", "project", "application", "function",
  "existing", "implementation", "current", "ensure", "update", "create", "add"
]);

function termsOf(text) {
  return [...new Set((text.toLowerCase().match(/[\p{L}\p{N}_-]{3,}/gu) || []).filter(term => !STOP_WORDS.has(term)))].slice(0, 30);
}

function makeExcerpt(content, query, maxChars = 5200) {
  if (content.length <= maxChars) return content;
  const lines = content.split(/\r?\n/);
  const terms = termsOf(query);
  const scored = lines.map((line, index) => ({
    index, score: terms.reduce((sum, term) => sum + (line.toLowerCase().includes(term) ? 1 : 0), 0)
  })).filter(line => line.score > 0).sort((a, b) => b.score - a.score).slice(0, 10);
  const keep = new Set();
  if (scored.length) {
    for (const line of scored) for (let index = Math.max(0, line.index - 3); index <= Math.min(lines.length - 1, line.index + 4); index += 1) keep.add(index);
  } else {
    for (let index = 0; index < Math.min(24, lines.length); index += 1) keep.add(index);
    for (let index = Math.max(0, lines.length - 24); index < lines.length; index += 1) keep.add(index);
  }
  let output = "";
  let previous = -2;
  for (const index of [...keep].sort((a, b) => a - b)) {
    const piece = (previous === index - 1 ? "" : "\n// … context omitted …\n") + lines[index] + "\n";
    if (output.length + piece.length > maxChars) break;
    output += piece;
    previous = index;
  }
  return output || content.slice(-maxChars);
}

function rankGraph(graph, goal, requestedPath = "") {
  const terms = termsOf(goal);
  const byPath = new Map(graph.nodes.map(node => [node.path, node]));
  const scores = new Map();
  for (const node of graph.nodes) {
    const file = graph.contents.get(node.path);
    const lowerPath = node.path.toLowerCase();
    const lowerContent = (file?.content || "").toLowerCase();
    let score = 0;
    for (const term of terms) {
      if (lowerPath.includes(term)) score += 8;
      score += Math.min(lowerContent.split(term).length - 1, 6) * 1.25;
    }
    if (requestedPath && node.path === requestedPath) score += 100;
    if (/\b(App|main|index|server|routes?|router|service|store|db|database|api|page|layout)\b/i.test(node.path)) score += 1.5;
    scores.set(node.path, score);
  }
  for (const node of graph.nodes) {
    const score = scores.get(node.path) || 0;
    if (score <= 0) continue;
    for (const related of [...node.imports, ...node.importedBy]) scores.set(related, (scores.get(related) || 0) + Math.min(8, score * 0.22));
  }
  return [...scores.entries()].map(([filePath, score]) => ({
    ...byPath.get(filePath), score, content: graph.contents.get(filePath)?.content || ""
  })).filter(item => item.content).sort((a, b) => b.score - a.score || a.path.localeCompare(b.path));
}

function createPlan(goal, inventory, ranked, graph) {
  const likelyFiles = ranked.slice(0, 8).map(item => item.path);
  const checks = [];
  if (inventory.scripts?.build) checks.push("build");
  if (inventory.scripts?.test || inventory.manifests?.pyprojectToml || inventory.manifests?.requirementsTxt) checks.push("test");
  if (inventory.scripts?.lint) checks.push("lint");
  if (inventory.scripts?.typecheck || inventory.scripts?.check) checks.push("typecheck");
  return {
    goal,
    mode: "repository-aware deterministic plan + constrained local-model patch generation",
    steps: [
      { id: "discover", title: "Discover architecture", status: "completed", detail: "Inspected " + inventory.fileCount + " workspace files; import graph contains " + graph.nodes.length + " readable source/config files." },
      { id: "scope", title: "Select change surfaces", status: "completed", detail: likelyFiles.length ? likelyFiles.join(", ") : "No relevant source files ranked yet." },
      { id: "patch", title: "Generate structured edits", status: "pending", detail: "Request exact find/replace operations in a JSON schema; reject ambiguous or unsafe edits." },
      { id: "verify", title: "Run checks and repair", status: checks.length ? "pending" : "not_available", detail: checks.length ? "Planned checks: " + checks.join(", ") + "." : "No configured build/test check was discovered." },
      { id: "review", title: "Stage final proposals", status: "pending", detail: "Stage hash-bound proposals for user review; do not modify the original project." }
    ],
    plannedChecks: checks,
    likelyFiles
  };
}

function makePatchPrompt(goal, inventory, plan, graph, ranked, feedback) {
  const map = ranked.slice(0, 12).map(node => ({
    path: node.path, imports: node.imports.slice(0, 10),
    importedBy: node.importedBy.slice(0, 10), score: Number(node.score.toFixed(2))
  }));
  const snippets = ranked.slice(0, 12).map(node => ({
    path: node.path, sha256: node.sha256, imports: node.imports.slice(0, 10),
    importedBy: node.importedBy.slice(0, 10), source: makeExcerpt(node.content, goal)
  }));
  return [
    MODEL_PROMPT,
    "\nTASK:\n" + goal,
    "\nPROJECT:\n" + JSON.stringify({
      name: inventory.name, root: "[selected local workspace]",
      packageManager: inventory.packageManager, frameworks: inventory.frameworks,
      scripts: inventory.scripts, manifests: inventory.manifests,
      graph: { totalFiles: graph.totalFiles, totalBytes: graph.totalBytes, truncated: graph.truncated, nodes: map }
    }),
    "\nPLAN (operational checklist):\n" + JSON.stringify(plan.steps.map(({ id, title, detail }) => ({ id, title, detail }))),
    "\nSOURCE EXCERPTS (retrieved and relevance-ranked):\n" + JSON.stringify(snippets),
    feedback ? "\nLAST ATTEMPT / VERIFICATION FEEDBACK. Correct these concrete issues; do not repeat the failed patch:\n" + feedback.slice(-6500) : "",
    "\nReturn one JSON object now. Use exact source anchors and do not alter unrelated files."
  ].filter(Boolean).join("\n");
}

async function modelJson(modelRequest, route, body, timeoutMs = 15000) {
  return modelRequest("POST", route, body, timeoutMs);
}

async function validatePatch(root, payload) {
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) throw new Error("Structured patch must be a JSON object.");
  if (typeof payload.summary !== "string" || !payload.summary.trim()) throw new Error("Structured patch is missing a short summary.");
  if (!Array.isArray(payload.edits) || payload.edits.length < 1 || payload.edits.length > 6) throw new Error("Patch must contain between 1 and 6 file edits.");
  const seen = new Set();
  const changes = [];
  for (const edit of payload.edits) {
    if (!edit || typeof edit !== "object") throw new Error("Each patch edit must be an object.");
    const relative = normalizeAgentPath(edit.path);
    if (seen.has(relative)) throw new Error("Use one deterministic edit per file; duplicate path: " + relative);
    seen.add(relative);
    if (typeof edit.find !== "string" || typeof edit.replace !== "string") throw new Error("Each edit requires string find and replace fields.");
    if (edit.find.length > 12000 || edit.replace.length > 12000) throw new Error("Each find/replace block is limited to 12,000 characters.");
    if (Buffer.byteLength(edit.replace, "utf8") > 40 * 1024) throw new Error("An individual new file or replacement may not exceed 40 KB.");
    let before = null;
    try { before = await readWorkspaceFile(root, relative); }
    catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      if (!message.startsWith("File does not exist:")) throw error;
    }
    if (before === null) {
      if (edit.find !== "" || !edit.replace.trim()) throw new Error("A new file requires an empty find and non-empty replace.");
      changes.push({
        path: relative, beforeContent: null, beforeHash: null, existedBefore: false,
        content: edit.replace, reason: typeof edit.reason === "string" ? edit.reason.slice(0, 500) : "",
        matches: 0
      });
      continue;
    }
    if (!edit.find) throw new Error("Existing-file edits must use a non-empty exact find anchor; full-file rewrite is rejected.");
    const matches = before.content.split(edit.find).length - 1;
    if (matches !== 1) throw new Error("Exact anchor for " + relative + " must occur exactly once; found " + matches + " matches.");
    const nextContent = before.content.replace(edit.find, () => edit.replace);
    if (nextContent === before.content) throw new Error("Patch for " + relative + " does not change the file.");
    if (Buffer.byteLength(nextContent, "utf8") > 512 * 1024) throw new Error("Patched file exceeds the 512 KB proposal limit.");
    changes.push({
      path: relative, beforeContent: before.content, beforeHash: before.sha256, existedBefore: true,
      content: nextContent, reason: typeof edit.reason === "string" ? edit.reason.slice(0, 500) : "",
      matches: 1
    });
  }
  return { summary: payload.summary.slice(0, 2000), changes };
}

async function copySandbox(root) {
  const destination = await fs.mkdtemp(path.join(os.tmpdir(), "syntra-agent-"));
  let count = 0;
  let bytes = 0;
  const moduleDir = path.join(root, "node_modules");
  async function copyDirectory(source, target, relative, depth = 0) {
    if (depth > 20) return;
    const entries = await fs.readdir(source, { withFileTypes: true }).catch(() => []);
    await fs.mkdir(target, { recursive: true });
    for (const entry of entries) {
      const sourcePath = path.join(source, entry.name);
      const targetPath = path.join(target, entry.name);
      const relativePath = relative ? relative + "/" + entry.name : entry.name;
      const lower = entry.name.toLowerCase();
      if (entry.isSymbolicLink()) {
        if (!relativePath.replace(/\\/g, "/").startsWith("node_modules/")) continue;
        const real = await fs.realpath(sourcePath).catch(() => null);
        if (!real || !within(moduleDir, real)) continue;
        const targetStat = await fs.stat(real).catch(() => null);
        if (!targetStat) continue;
        if (targetStat.isFile()) {
          if (++count > MAX_CANDIDATE_ENTRIES || bytes + targetStat.size > MAX_CANDIDATE_BYTES) throw new Error("Candidate sandbox exceeds configured file/byte limits.");
          await fs.copyFile(real, targetPath);
          bytes += targetStat.size;
        } else if (targetStat.isDirectory()) await copyDirectory(real, targetPath, relativePath, depth + 1);
        continue;
      }
      if (entry.isDirectory()) {
        if ((SKIP_DIRS.has(lower) && lower !== "node_modules") || lower === ".git" || lower === ".syntra") continue;
        await copyDirectory(sourcePath, targetPath, relativePath, depth + 1);
        continue;
      }
      if (!entry.isFile() || isSecretPath(relativePath)) continue;
      const stat = await fs.stat(sourcePath).catch(() => null);
      if (!stat || stat.size > 100 * 1024 * 1024) continue;
      if (++count > MAX_CANDIDATE_ENTRIES || bytes + stat.size > MAX_CANDIDATE_BYTES) throw new Error("Candidate sandbox exceeds configured file/byte limits.");
      await fs.copyFile(sourcePath, targetPath);
      bytes += stat.size;
    }
  }
  try {
    await copyDirectory(root, destination, "");
    return { root: destination, filesCopied: count, bytesCopied: bytes };
  } catch (error) {
    await fs.rm(destination, { recursive: true, force: true }).catch(() => {});
    throw error;
  }
}

async function resetSandboxFiles(sandboxRoot, previous, originalChanges) {
  for (const relative of previous) {
    const original = originalChanges.get(relative);
    const target = path.resolve(sandboxRoot, relative.split("/").join(path.sep));
    if (!within(sandboxRoot, target)) throw new Error("Sandbox patch path escaped the temporary workspace.");
    if (!original || !original.existedBefore) await fs.rm(target, { force: true }).catch(() => {});
    else {
      await fs.mkdir(path.dirname(target), { recursive: true });
      await fs.writeFile(target, original.beforeContent, "utf8");
    }
  }
}

async function writeCandidateChanges(sandboxRoot, changes) {
  for (const change of changes) {
    const target = path.resolve(sandboxRoot, change.path.split("/").join(path.sep));
    if (!within(sandboxRoot, target)) throw new Error("Candidate path escaped the sandbox.");
    await fs.mkdir(path.dirname(target), { recursive: true });
    await fs.writeFile(target, change.content, "utf8");
  }
}

function truncateOutput(output) {
  const value = String(output || "");
  return value.length > 8000 ? value.slice(-8000) : value;
}


function packageScriptsChanged(changes) {
  const change = changes.find(item => item.path === "package.json");
  if (!change || typeof change.beforeContent !== "string") return false;
  try {
    const before = JSON.parse(change.beforeContent);
    const after = JSON.parse(change.content);
    const sortedScripts = value => {
      const scripts = value && value.scripts && typeof value.scripts === "object" && !Array.isArray(value.scripts)
        ? value.scripts : {};
      return JSON.stringify(Object.fromEntries(Object.keys(scripts).sort().map(key => [key, scripts[key]])));
    };
    return sortedScripts(before) !== sortedScripts(after);
  } catch {
    // Never execute candidate-controlled checks if the package manifest was made invalid.
    return true;
  }
}

function verificationReviewReasons(changes) {
  const reasons = [];
  for (const change of changes) {
    const relative = String(change.path || "").replace(/\\/g, "/");
    const base = path.posix.basename(relative).toLowerCase();
    if (/(^|\/)(?:test|tests|__tests__|spec|specs)(\/|$)/i.test(relative)
      || /\.(?:test|spec)\.[^.]+$/i.test(base)) {
      reasons.push("Patch changes test/spec source (" + relative + "); review the test diff before applying.");
    }
    if (/^(?:\.github\/workflows\/|\.gitlab-ci\.yml$|\.circleci\/|azure-pipelines(?:\.[^/]+)?$|jenkinsfile$)/i.test(relative)
      || /(^|\/)(?:vite|vitest|jest|playwright|cypress|webpack|rollup|eslint|prettier|biome|babel|tsconfig)(?:\.config)?\.[^/]+$/i.test(relative)
      || /^(?:pytest\.ini|tox\.ini|noxfile\.py|pyproject\.toml|Makefile|justfile)$/i.test(base)) {
      reasons.push("Patch changes build/test/CI configuration (" + relative + "); review the verification setup before applying.");
    }
  }
  return [...new Set(reasons)];
}

export async function runAgentTask({
  root: inputRoot, goal, maxIterations = 3, runChecks = false, requestedPath = "", modelRequest,
  onProgress = () => {}
}) {
  const root = await resolveWorkspace(inputRoot);
  if (typeof goal !== "string" || !goal.trim() || goal.length > 5000) throw new Error("Enter a specific application/code task (maximum 5,000 characters).");
  if (typeof modelRequest !== "function") throw new Error("The local model request bridge is required.");
  maxIterations = Math.max(1, Math.min(MAX_ITERATIONS, Number(maxIterations) || 3));
  const modelStatus = await modelRequest("GET", "/model/status", undefined, 15000);
  if (!modelStatus.checkpoint_exists) throw new Error("No usable local model is available. Start a local Ollama model or train SYNTRA's native model before requesting an agent patch.");
  const contextWindow = Number(modelStatus.context_tokens || modelStatus.config?.seq_len || 4096);
  const contextBudget = Math.max(128, Math.floor(contextWindow * 0.68));
  const outputBudget = Math.max(128, Math.min(1024, Math.floor(contextWindow * 0.22)));
  const inventory = await inspectWorkspace(root);
  onProgress({ phase: "repository-map", iteration: 0, message: "Inspecting files and building the local import graph." });
  const graph = await buildRepositoryGraph(root);
  const ranked = rankGraph(graph, goal, requestedPath);
  const plan = createPlan(goal, inventory, ranked, graph);
  const attempts = [];
  let feedback = "";
  let finalChanges = null;
  let finalVerification = { status: "not_run", checks: [], passed: false };
  let snapshot = null;
  let previouslyTouched = [];
  const originalByPath = new Map();
  const proposals = [];
  if (!ranked.length) throw new Error("No readable source/configuration files were found for repository context.");

  try {
    for (let iteration = 1; iteration <= maxIterations; iteration += 1) {
      onProgress({ phase: "planning", iteration, message: "Ranking dependencies and allocating a token budget for relevant source files." });
      let selected = ranked.slice(0, 12);
      let prompt = makePatchPrompt(goal, inventory, plan, graph, selected, feedback);
      let tokenInfo = await modelJson(modelRequest, "/model/tokenize", { text: prompt }, 30000);
      while (tokenInfo.token_count > contextBudget && selected.length > 1) {
        selected = selected.slice(0, -1);
        prompt = makePatchPrompt(goal, inventory, plan, graph, selected, feedback);
        tokenInfo = await modelJson(modelRequest, "/model/tokenize", { text: prompt }, 30000);
      }
      if (tokenInfo.token_count > contextBudget) {
        feedback = "Context budget exceeded: need " + tokenInfo.token_count + " tokens but only " + contextBudget + " are available. Shorten the plan and focus on one file.";
        attempts.push({ iteration, status: "context_too_large", promptTokens: tokenInfo.token_count, contextBudget, feedback });
        continue;
      }

      plan.steps.find(step => step.id === "patch").status = "running";
      onProgress({ phase: "generating-patch", iteration, message: "Requesting structured, exact-anchor patch JSON from the selected local model." });
      let generated;
      try {
        generated = await modelJson(modelRequest, "/model/generate", {
          prompt, max_new_tokens: outputBudget, temperature: 0.15,
          top_k: Math.min(40, Number(modelStatus.tokenizer_vocab_size || modelStatus.config?.vocab_size || 1024)),
          device: "auto"
        }, 300000);
      } catch (error) {
        feedback = "The selected model call failed: " + (error instanceof Error ? error.message : String(error));
        attempts.push({ iteration, status: "generation_failed", feedback });
        continue;
      }

      let candidate;
      try {
        const payload = extractJsonObject(generated.text);
        candidate = await validatePatch(root, payload);
      } catch (error) {
        feedback = "Patch validation failed: " + (error instanceof Error ? error.message : String(error)) +
          "\nRequired schema: {summary:string, edits:[{path:string, find:string, replace:string, reason:string}]}. The next output must be valid JSON, not source continuation.";
        attempts.push({
          iteration, status: "invalid_patch", promptTokens: tokenInfo.token_count,
          outputTokens: generated.generated_tokens, modelStep: generated.trained_steps, feedback
        });
        continue;
      }

      onProgress({ phase: "patch-validated", iteration, message: "Patch schema, paths, file hashes and exact anchors validated." });
      attempts.push({
        iteration, status: runChecks ? "patch_validated" : "patch_validated_not_run",
        summary: candidate.summary, changedFiles: candidate.changes.map(change => change.path),
        promptTokens: tokenInfo.token_count, outputTokens: generated.generated_tokens,
        modelStep: generated.trained_steps
      });
      finalChanges = candidate;
      plan.steps.find(step => step.id === "patch").status = "completed";
      if (!runChecks) {
        finalVerification = {
          status: "not_run", passed: false, checks: [],
          note: "User did not authorize build/test execution in the temporary candidate workspace."
        };
        break;
      }

      if (packageScriptsChanged(candidate.changes)) {
        const note = "Automatic verification was blocked because the patch changes package.json scripts. Running candidate-controlled build/test scripts would make the checks self-modifying. Review the manifest diff, then type /apply only if you accept the script changes.";
        finalVerification = {
          status: "verification_blocked", passed: false, checks: [{
            task: "verification-script-integrity", passed: false, skipped: true, timedOut: false,
            output: note
          }], note
        };
        feedback = note;
        attempts[attempts.length - 1].status = "verification_blocked";
        onProgress({ phase: "verification-blocked", iteration, message: note });
        break;
      }

      if (!snapshot) {
        onProgress({ phase: "sandbox-prepare", iteration, message: "Creating a temporary project copy for safe build/test runs." });
        try { snapshot = await copySandbox(root); }
        catch (error) {
          finalVerification = {
            status: "sandbox_unavailable", passed: false, checks: [],
            note: "Could not create an isolated project copy: " + (error instanceof Error ? error.message : String(error))
          };
          feedback = finalVerification.note;
          attempts[attempts.length - 1].status = "sandbox_unavailable";
          break;
        }
      }
      for (const change of candidate.changes) if (!originalByPath.has(change.path)) originalByPath.set(change.path, change);
      await resetSandboxFiles(snapshot.root, previouslyTouched, originalByPath);
      await writeCandidateChanges(snapshot.root, candidate.changes);
      previouslyTouched = candidate.changes.map(change => change.path);

      const checks = [];
      const scripts = inventory.scripts || {};
      const hasTestsDirectory = await fs.stat(path.join(root, "tests")).then(stat => stat.isDirectory()).catch(() => false);
      const tasks = [];
      if (scripts.build) tasks.push("build");
      if (scripts.test || inventory.manifests?.pyprojectToml || inventory.manifests?.requirementsTxt || hasTestsDirectory) tasks.push("test");
      if (scripts.lint) tasks.push("lint");
      if (scripts.typecheck || scripts.check) tasks.push("typecheck");
      for (const task of tasks) {
        onProgress({ phase: "verification", iteration, message: "Running " + task + " in the temporary project copy." });
        try {
          const result = await runProjectCheck(snapshot.root, task, { timeoutMs: CHECK_TIMEOUT_MS });
          checks.push({
            task, passed: result.passed, timedOut: result.timedOut, exitCode: result.exitCode,
            durationMs: result.durationMs, command: result.command, output: truncateOutput(result.output)
          });
        } catch (error) {
          checks.push({ task, passed: false, error: error instanceof Error ? error.message : String(error), output: "" });
        }
        if (checks.at(-1)?.passed !== true) break;
      }
      const allPassed = tasks.length > 0 && checks.length === tasks.length && checks.every(item => item.passed);
      const reviewReasons = verificationReviewReasons(candidate.changes);
      const verificationStatus = allPassed
        ? reviewReasons.length ? "passed_review_required" : "passed"
        : tasks.length ? "failed" : "no_checks_configured";
      finalVerification = {
        status: verificationStatus,
        passed: allPassed, checks,
        sandbox: { filesCopied: snapshot.filesCopied, bytesCopied: snapshot.bytesCopied },
        ...(reviewReasons.length ? { reviewRequiredReasons: reviewReasons } : {})
      };
      attempts[attempts.length - 1].status = allPassed
        ? reviewReasons.length ? "verification_passed_review_required" : "verification_passed"
        : "verification_failed";
      if (allPassed) {
        onProgress({
          phase: reviewReasons.length ? "verification-review-required" : "verification-passed",
          iteration,
          message: reviewReasons.length
            ? "Checks passed, but this patch changes tests or verification configuration and requires review before application."
            : "Every discovered check passed in the temporary copy."
        });
        break;
      }
      if (tasks.length === 0) {
        feedback = "No supported build/test/lint/typecheck command was discovered; the patch cannot be automatically verified.";
        onProgress({ phase: "verification-unavailable", iteration, message: feedback });
        break;
      }
      onProgress({ phase: "repair", iteration, message: "A check failed; the failure output will be fed into the next bounded repair attempt." });
      feedback = "Candidate verification failed. Address these results; do not assert success:\n" +
        checks.map(check => "[" + check.task + "] passed=" + check.passed + ", timedOut=" + (check.timedOut || false) +
          ", exitCode=" + (check.exitCode ?? "unknown") + "\n" + (check.error || "") + "\n" + (check.output || "")).join("\n\n");
    }

    if (finalChanges) {
      onProgress({ phase: "staging-proposals", iteration: attempts.length, message: "Staging hash-bound proposals without modifying project files." });
      for (const change of finalChanges.changes) proposals.push(await proposeFileChange(root, change.path, change.content, change.beforeHash));
      plan.steps.find(step => step.id === "verify").status = finalVerification.status === "passed" ? "completed" : finalVerification.status;
      plan.steps.find(step => step.id === "review").status = "completed";
    }

    onProgress({ phase: finalChanges ? "completed" : "generation-failed", iteration: attempts.length, message: finalChanges ? "Agent loop finished; review staged proposals." : "Agent did not produce a schema-valid patch." });
    return {
      status: !finalChanges ? "generation_failed"
        : finalVerification.status === "passed" ? "verified_proposals_ready"
          : finalVerification.status === "passed_review_required" ? "verified_proposals_review_required"
            : "proposals_ready_unverified",
      root, goal, plan,
      repository: {
        name: inventory.name, fileCount: inventory.fileCount,
        graphNodes: graph.nodes.length, graphBytes: graph.totalBytes, graphTruncated: graph.truncated,
        contextWindowTokens: contextWindow, contextBudgetTokens: contextBudget,
        tokenizer: modelStatus.tokenizer || "byte-BPE",
        tokenizerVocabulary: modelStatus.tokenizer_vocab_size || modelStatus.config?.vocab_size
      },
      selectedContext: ranked.slice(0, 12).map(({ path: filePath, score, imports, importedBy, sha256 }) => ({
        path: filePath, score: Number(score.toFixed(2)), imports, importedBy, sha256
      })),
      patchSummary: finalChanges?.summary || null,
      changedFiles: finalChanges?.changes.map(change => ({
        path: change.path, existedBefore: change.existedBefore, baseHash: change.beforeHash,
        afterHash: sha256(change.content), reason: change.reason,
        preview: change.content.split(/\r?\n/).slice(0, 32).join("\n")
      })) || [],
      proposals, verification: finalVerification, attempts,
      notices: [
        "Patch text came from the selected selected local model; a schema-valid patch is not automatically a correct patch.",
        "Original project files were not modified by this agent run. Changes are staged as hash-bound proposals for explicit review and apply.",
        finalVerification.status === "passed"
          ? "All discovered checks passed in a temporary copy. Review source changes and outputs before applying."
          : finalVerification.status === "passed_review_required"
            ? "All discovered checks passed, but this patch touches test or verification configuration. Review it before applying."
            : "Changes were not verified successfully. Review the failure details; do not treat these proposals as production-ready."
      ]
    };
  } finally {
    if (snapshot?.root) await fs.rm(snapshot.root, { recursive: true, force: true }).catch(() => {});
  }
}

export { extractJsonObject as parsePatchOutput, validatePatch as validateStructuredPatch };

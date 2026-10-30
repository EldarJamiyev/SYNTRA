import fs from "node:fs/promises";
import fsSync from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { spawn } from "node:child_process";

const IGNORED_DIRS = new Set([
  ".git", ".syntra", ".venv", "venv", "env", "node_modules", "__pycache__",
  ".pytest_cache", ".mypy_cache", ".ruff_cache", "dist", "build", "coverage",
  ".next", ".nuxt", ".idea", ".vscode", "target", "vendor", "obj"
]);
const TEXT_EXTENSIONS = new Set([
  ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".mts", ".cts", ".py", ".pyi",
  ".java", ".kt", ".go", ".rs", ".c", ".h", ".cpp", ".hpp", ".cs", ".php",
  ".rb", ".swift", ".html", ".css", ".scss", ".sass", ".less", ".sql", ".json",
  ".md", ".mdx", ".txt", ".xml", ".yaml", ".yml", ".toml", ".ini", ".csproj",
  ".fsproj", ".vbproj", ".sln", ".props", ".targets", ".sh",
  ".ps1", ".bat", ".cmd", ".dockerfile", ".graphql", ".gql", ".vue", ".svelte",
  ".astro", ".env.example", ".gitignore", ".editorconfig"
]);
const SENSITIVE_NAMES = new Set([
  ".env", ".env.local", ".env.production", ".npmrc", ".pypirc", "credentials",
  "credentials.json", "secrets.json", "id_rsa", "id_ed25519", "known_hosts"
]);
const SENSITIVE_EXTENSIONS = new Set([".pem", ".key", ".p12", ".pfx", ".keystore"]);
const MAX_TEXT_BYTES = 512 * 1024;
const MAX_FILES = 5000;
const MAX_DEPTH = 10;
const MAX_OUTPUT = 50000;

function sha256(value) {
  return crypto.createHash("sha256").update(value).digest("hex");
}

function ensureAbsoluteRoot(input) {
  if (typeof input !== "string" || !input.trim()) {
    throw new Error("Provide an absolute project directory path.");
  }
  if (!path.isAbsolute(input)) {
    throw new Error("Project path must be absolute.");
  }
  const resolved = path.resolve(input);
  if (resolved === path.parse(resolved).root) {
    throw new Error("Refusing to use a filesystem root as a project workspace.");
  }
  return resolved;
}

export async function resolveWorkspace(input, options = {}) {
  const root = ensureAbsoluteRoot(input);
  if (options.create) await fs.mkdir(root, { recursive: true });
  const stat = await fs.stat(root).catch(() => null);
  if (!stat || !stat.isDirectory()) {
    throw new Error("Project directory does not exist or is not a directory: " + root);
  }
  const realRoot = await fs.realpath(root);
  if (realRoot === path.parse(realRoot).root) {
    throw new Error("Refusing to use a filesystem root as a project workspace.");
  }
  return realRoot;
}

function isWithinRoot(root, target) {
  const relative = path.relative(root, target);
  return relative === "" || (!relative.startsWith(".." + path.sep) && relative !== ".." && !path.isAbsolute(relative));
}

function normalizeRelativePath(relativePath) {
  if (typeof relativePath !== "string" || !relativePath.trim()) {
    throw new Error("A relative file path is required.");
  }
  if (path.isAbsolute(relativePath) || /^[a-zA-Z]:[\\/]/.test(relativePath)) {
    throw new Error("Absolute file paths are not accepted for workspace operations.");
  }
  const normalized = relativePath.replace(/\\/g, "/");
  if (normalized.split("/").some(part => part === ".." || part === "." || part === "")) {
    throw new Error("Relative path traversal or empty path segments are not allowed.");
  }
  if (normalized.startsWith(".syntra/") || normalized === ".syntra") {
    throw new Error("The .syntra state directory is reserved by SYNTRA.");
  }
  return normalized;
}

function isSensitiveRelativePath(relativePath) {
  const parts = relativePath.split("/").map(part => part.toLowerCase());
  const base = parts[parts.length - 1] || "";
  const ext = path.posix.extname(base);
  return parts.some(part => part === ".git" || part === "secrets" || part === "credentials")
    || SENSITIVE_NAMES.has(base)
    || (base.startsWith(".env.") && base !== ".env.example")
    || SENSITIVE_EXTENSIONS.has(ext)
    || base.endsWith(".secret");
}

async function resolveFile(root, relativePath, options = {}) {
  const rel = normalizeRelativePath(relativePath);
  if (isSensitiveRelativePath(rel)) {
    throw new Error("SYNTRA blocks access to likely secret/credential files: " + rel);
  }
  const target = path.resolve(root, rel.split("/").join(path.sep));
  if (!isWithinRoot(root, target) || target === root) {
    throw new Error("Path escapes the selected workspace.");
  }
  let cursor = root;
  for (const segment of path.relative(root, target).split(path.sep)) {
    cursor = path.join(cursor, segment);
    const component = await fs.lstat(cursor).catch(() => null);
    if (component?.isSymbolicLink()) {
      throw new Error("Symbolic link path components are not allowed inside a workspace operation.");
    }
    if (!component) break;
    if (cursor !== target && !component.isDirectory()) {
      throw new Error("A parent path component is not a directory.");
    }
  }
  const parent = path.dirname(target);
  const existingParent = await fs.realpath(parent).catch(() => null);
  if (existingParent && !isWithinRoot(root, existingParent)) {
    throw new Error("Resolved parent directory escapes the selected workspace.");
  }
  const stat = await fs.lstat(target).catch(() => null);
  if (stat?.isSymbolicLink()) {
    throw new Error("Symbolic links are not read or modified by SYNTRA workspace tools.");
  }
  if (stat && !stat.isFile() && !options.allowDirectory) {
    throw new Error("Path is not a regular file: " + rel);
  }
  if (!stat && !options.allowMissing) {
    throw new Error("File does not exist: " + rel);
  }
  return { target, relativePath: rel, stat };
}

function isTextCandidate(filePath) {
  const base = path.basename(filePath).toLowerCase();
  if (base.endsWith(".env.example")) return true;
  if (base === "dockerfile" || base === "makefile" || base === ".gitignore" || base === ".editorconfig") return true;
  if (isSensitiveRelativePath(filePath)) return false;
  return TEXT_EXTENSIONS.has(path.extname(base));
}

async function walkFiles(root) {
  const files = [];
  async function walk(directory, depth) {
    if (depth > MAX_DEPTH || files.length >= MAX_FILES) return;
    let entries;
    try {
      entries = await fs.readdir(directory, { withFileTypes: true });
    } catch {
      return;
    }
    entries.sort((a, b) => a.name.localeCompare(b.name));
    for (const entry of entries) {
      if (files.length >= MAX_FILES) break;
      if (entry.isSymbolicLink()) continue;
      if (entry.isDirectory()) {
        if (!IGNORED_DIRS.has(entry.name.toLowerCase())) await walk(path.join(directory, entry.name), depth + 1);
        continue;
      }
      if (!entry.isFile()) continue;
      const absolute = path.join(directory, entry.name);
      const relative = path.relative(root, absolute).split(path.sep).join("/");
      if (!isTextCandidate(relative)) continue;
      let stat;
      try { stat = await fs.stat(absolute); } catch { continue; }
      if (stat.size <= 0 || stat.size > MAX_TEXT_BYTES) continue;
      files.push({ absolute, relative, size: stat.size, extension: path.extname(entry.name).toLowerCase() });
    }
  }
  await walk(root, 0);
  return files;
}

function detectLanguage(extension, filename) {
  if (filename.toLowerCase() === "dockerfile") return "Docker";
  const mapping = {
    ".ts": "TypeScript", ".tsx": "TypeScript React", ".js": "JavaScript",
    ".jsx": "JavaScript React", ".mjs": "JavaScript", ".py": "Python",
    ".rs": "Rust", ".go": "Go", ".java": "Java", ".cs": "C#",
    ".cpp": "C++", ".c": "C", ".html": "HTML", ".css": "CSS",
    ".scss": "SCSS", ".json": "JSON", ".md": "Markdown", ".sql": "SQL",
    ".yaml": "YAML", ".yml": "YAML", ".toml": "TOML", ".sh": "Shell",
    ".ps1": "PowerShell", ".vue": "Vue", ".svelte": "Svelte", ".astro": "Astro"
  };
  return mapping[extension] || "Other";
}

async function readJsonIfPresent(root, relativePath) {
  try {
    const resolved = await resolveFile(root, relativePath);
    return JSON.parse(await fs.readFile(resolved.target, "utf8"));
  } catch {
    return null;
  }
}

export async function inspectWorkspace(inputRoot) {
  const root = await resolveWorkspace(inputRoot);
  const files = await walkFiles(root);
  const languageCounts = {};
  for (const file of files) {
    const language = detectLanguage(file.extension, path.basename(file.relative));
    languageCounts[language] = (languageCounts[language] || 0) + 1;
  }
  const packageJson = await readJsonIfPresent(root, "package.json");
  const exists = async name => fs.access(path.join(root, name)).then(() => true).catch(() => false);
  const pyprojectExists = await exists("pyproject.toml");
  const requirementsExists = await exists("requirements.txt");
  const cargoExists = await exists("Cargo.toml");
  const goModExists = await exists("go.mod");
  const rootEntries = await fs.readdir(root).catch(() => []);
  const dotnetSolutionExists = rootEntries.some(name => name.toLowerCase().endsWith(".sln"));
  const csproj = files.find(file => file.relative.toLowerCase().endsWith(".csproj"));
  const readme = files.find(file => /^readme([.]|$)/i.test(path.basename(file.relative)));
  let readmeExcerpt = "";
  if (readme) {
    try { readmeExcerpt = (await fs.readFile(readme.absolute, "utf8")).slice(0, 1500); } catch {}
  }
  const scripts = packageJson && packageJson.scripts && typeof packageJson.scripts === "object"
    ? Object.fromEntries(Object.entries(packageJson.scripts).filter(([key, value]) =>
        ["build", "test", "lint", "typecheck", "check"].includes(key) && typeof value === "string"))
    : {};
  const frameworkHints = [];
  if (packageJson?.dependencies || packageJson?.devDependencies) {
    const deps = { ...(packageJson.dependencies || {}), ...(packageJson.devDependencies || {}) };
    for (const [name, label] of [
      ["react", "React"], ["vite", "Vite"], ["next", "Next.js"], ["express", "Express"],
      ["fastify", "Fastify"], ["electron", "Electron"], ["typescript", "TypeScript"],
      ["vue", "Vue"], ["svelte", "Svelte"], ["@nestjs/core", "NestJS"]
    ]) {
      if (deps[name]) frameworkHints.push(label);
    }
  }
  if (pyprojectExists || requirementsExists) frameworkHints.push("Python");
  if (cargoExists) frameworkHints.push("Rust/Cargo");
  if (goModExists) frameworkHints.push("Go");
  if (csproj || dotnetSolutionExists) frameworkHints.push(".NET");
  const packageManager = packageJson
    ? (await exists("pnpm-lock.yaml") ? "pnpm" : await exists("yarn.lock") ? "yarn" : "npm")
    : pyprojectExists || requirementsExists ? "Python"
      : cargoExists ? "Cargo" : goModExists ? "Go" : (csproj || dotnetSolutionExists) ? ".NET" : "Unknown";

  return {
    root,
    name: packageJson?.name || path.basename(root),
    fileCount: files.length,
    truncated: files.length >= MAX_FILES,
    languageCounts: Object.fromEntries(Object.entries(languageCounts).sort((a, b) => b[1] - a[1])),
    packageManager,
    frameworks: [...new Set(frameworkHints)],
    scripts,
    manifests: {
      packageJson: Boolean(packageJson),
      pyprojectToml: pyprojectExists,
      requirementsTxt: requirementsExists,
      cargoToml: cargoExists,
      goMod: goModExists,
      dotnetProject: Boolean(csproj),
      dotnetSolution: dotnetSolutionExists
    },
    readmeExcerpt,
    files: files.slice(0, 250).map(file => ({
      path: file.relative, size: file.size,
      language: detectLanguage(file.extension, path.basename(file.relative))
    }))
  };
}

export async function readWorkspaceFile(inputRoot, relativePath) {
  const root = await resolveWorkspace(inputRoot);
  const resolved = await resolveFile(root, relativePath);
  if (resolved.stat.size > MAX_TEXT_BYTES) throw new Error("File exceeds the 512 KB text-inspection limit.");
  const content = await fs.readFile(resolved.target, "utf8");
  return {
    root, path: resolved.relativePath, bytes: Buffer.byteLength(content, "utf8"),
    sha256: sha256(content), content
  };
}

export async function searchWorkspace(inputRoot, query, limit = 20) {
  const root = await resolveWorkspace(inputRoot);
  if (typeof query !== "string" || query.trim().length < 2 || query.length > 300) {
    throw new Error("Search query must contain 2-300 characters.");
  }
  const words = [...new Set(query.toLowerCase().match(/[a-z0-9_./-]{2,}/g) || [])].slice(0, 16);
  if (!words.length) throw new Error("Search query did not contain searchable terms.");
  const files = await walkFiles(root);
  const results = [];
  for (const file of files) {
    let content;
    try { content = await fs.readFile(file.absolute, "utf8"); } catch { continue; }
    const lower = content.toLowerCase();
    let score = 0;
    for (const word of words) {
      const count = lower.split(word).length - 1;
      if (count) score += Math.min(count, 8) * (word === query.trim().toLowerCase() ? 3 : 1);
    }
    if (!score) continue;
    const lines = content.split(/\r?\n/);
    const matched = [];
    for (let index = 0; index < lines.length && matched.length < 4; index++) {
      const lowerLine = lines[index].toLowerCase();
      if (words.some(word => lowerLine.includes(word))) {
        matched.push({ line: index + 1, text: lines[index].slice(0, 260) });
      }
    }
    results.push({ path: file.relative, score, size: file.size, matches: matched });
  }
  results.sort((a, b) => b.score - a.score || a.path.localeCompare(b.path));
  return {
    root, query, terms: words, searchedFiles: files.length,
    results: results.slice(0, Math.max(1, Math.min(50, limit)))
  };
}

function stateDirectory(root) {
  const directory = path.join(root, ".syntra");
  let info;
  try {
    info = fsSync.lstatSync(directory);
  } catch (error) {
    if (error && error.code === "ENOENT") return directory;
    throw error;
  }
  if (info.isSymbolicLink()) {
    throw new Error("The workspace .syntra state directory must not be a symbolic link; refusing to read or write state outside the workspace.");
  }
  if (!info.isDirectory()) {
    throw new Error("The workspace .syntra state path exists but is not a directory.");
  }
  const realDirectory = fsSync.realpathSync(directory);
  if (!isWithinRoot(root, realDirectory)) {
    throw new Error("The workspace .syntra state directory resolves outside the selected workspace.");
  }

  for (const name of ["proposals", "backups", "plans", "runs"]) {
    const nested = path.join(directory, name);
    let nestedInfo;
    try {
      nestedInfo = fsSync.lstatSync(nested);
    } catch (error) {
      if (error && error.code === "ENOENT") continue;
      throw error;
    }
    if (nestedInfo.isSymbolicLink()) {
      throw new Error("The workspace .syntra/" + name + " state directory must not be a symbolic link.");
    }
    if (!nestedInfo.isDirectory()) {
      throw new Error("The workspace .syntra/" + name + " state path exists but is not a directory.");
    }
    if (!isWithinRoot(root, fsSync.realpathSync(nested))) {
      throw new Error("The workspace .syntra/" + name + " state directory resolves outside the selected workspace.");
    }
  }

  const auditPath = path.join(directory, "actions.jsonl");
  try {
    const auditInfo = fsSync.lstatSync(auditPath);
    if (auditInfo.isSymbolicLink() || !auditInfo.isFile()) {
      throw new Error("The workspace .syntra/actions.jsonl audit log must be a regular file, not a symbolic link.");
    }
  } catch (error) {
    if (error && error.code !== "ENOENT") throw error;
  }
  return directory;
}

async function assertRegularStateFile(target, label) {
  let info;
  try {
    info = await fs.lstat(target);
  } catch (error) {
    if (error && error.code === "ENOENT") return false;
    throw error;
  }
  if (info.isSymbolicLink() || !info.isFile()) {
    throw new Error(label + " must be a regular file, not a symbolic link.");
  }
  return true;
}

async function writeFileAtomic(target, content) {
  await fs.mkdir(path.dirname(target), { recursive: true });
  const temporary = target + "." + process.pid + "." + crypto.randomBytes(4).toString("hex") + ".tmp";
  await fs.writeFile(temporary, content, { encoding: "utf8", flag: "wx" });
  try {
    await fs.rename(temporary, target);
  } catch (error) {
    await fs.rm(temporary, { force: true });
    throw error;
  }
}

async function appendAudit(root, entry) {
  const auditPath = path.join(stateDirectory(root), "actions.jsonl");
  await fs.mkdir(path.dirname(auditPath), { recursive: true });
  await fs.appendFile(auditPath, JSON.stringify({ timestamp: new Date().toISOString(), ...entry }) + "\n", "utf8");
}

export async function recordWorkspaceToolEvent(inputRoot, event) {
  const root = await resolveWorkspace(inputRoot);
  if (!event || typeof event !== "object" || Array.isArray(event)) {
    throw new Error("A structured tool event is required.");
  }
  const runId = typeof event.runId === "string" && /^[a-f0-9-]{36}$/i.test(event.runId) ? event.runId : null;
  const tool = typeof event.tool === "string" && /^[a-z][a-z0-9._-]{2,80}$/.test(event.tool) ? event.tool : null;
  const permission = ["read_only", "local_state_write", "execute", "source_write", "destructive_state_write"].includes(event.permission)
    ? event.permission : null;
  const status = ["started", "completed", "failed", "cancelled", "approval_required"].includes(event.status) ? event.status : null;
  if (!runId || !tool || !permission || !status || typeof event.approved !== "boolean") {
    throw new Error("Tool audit event metadata is invalid.");
  }
  const inputSummary = event.inputSummary && typeof event.inputSummary === "object" && !Array.isArray(event.inputSummary)
    ? event.inputSummary : {};
  const outputSummary = event.outputSummary && typeof event.outputSummary === "object" && !Array.isArray(event.outputSummary)
    ? event.outputSummary : {};
  const serializeBounded = (value, label) => {
    const serialized = JSON.stringify(value);
    if (serialized.length > 3000) throw new Error(label + " exceeds the audit size limit.");
    return value;
  };
  const durationMs = Number.isFinite(event.durationMs) ? Math.max(0, Math.min(3_600_000, Math.floor(event.durationMs))) : 0;
  await appendAudit(root, {
    action: "tool_runtime_event",
    runId,
    tool,
    permission,
    approved: event.approved,
    status,
    durationMs,
    inputSummary: serializeBounded(inputSummary, "Tool input summary"),
    outputSummary: serializeBounded(outputSummary, "Tool output summary"),
    error: typeof event.error === "string" ? event.error.slice(0, 1000) : undefined
  });
  return { root, recorded: true, runId, status };
}


const MAX_AGENT_RUNS = 40;
const MAX_AGENT_RUN_BYTES = 160 * 1024;
const AGENT_RUN_STATUSES = new Set(["queued", "running", "completed", "failed", "cancelled", "interrupted"]);
const AGENT_RUN_TRANSITIONS = {
  queued: new Set(["running", "interrupted", "failed", "cancelled"]),
  running: new Set(["running", "interrupted", "completed", "failed", "cancelled"]),
  interrupted: new Set(["interrupted", "running"]),
  completed: new Set(["completed"]),
  failed: new Set(["failed"]),
  cancelled: new Set(["cancelled"])
};

function validAgentRunId(value) {
  return typeof value === "string"
    && /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(value);
}

async function agentRunDirectory(root) {
  const directory = path.join(stateDirectory(root), "runs");
  await fs.mkdir(directory, { recursive: true });
  const stat = await fs.lstat(directory);
  if (stat.isSymbolicLink() || !stat.isDirectory()) {
    throw new Error("The workspace .syntra/runs state directory must be a regular directory.");
  }
  const real = await fs.realpath(directory);
  if (!isWithinRoot(root, real)) throw new Error("The workspace .syntra/runs directory resolves outside the workspace.");
  return real;
}

async function readAgentRunRecord(root, runId) {
  if (!validAgentRunId(runId)) throw new Error("A valid agent run ID is required.");
  const directory = await agentRunDirectory(root);
  const target = path.join(directory, runId + ".json");
  if (!(await assertRegularStateFile(target, "Agent run record"))) return null;
  const stat = await fs.stat(target);
  if (stat.size > MAX_AGENT_RUN_BYTES) throw new Error("Agent run record exceeds the size limit.");
  let run;
  try { run = JSON.parse(await fs.readFile(target, "utf8")); }
  catch { throw new Error("Agent run record is malformed and cannot be recovered."); }
  if (!run || run.schemaVersion !== 1 || run.id !== runId || path.resolve(String(run.root || "")) !== root) {
    throw new Error("Agent run record does not match this workspace.");
  }
  return run;
}

async function writeAgentRunRecord(root, run) {
  const directory = await agentRunDirectory(root);
  const target = path.join(directory, run.id + ".json");
  if (await assertRegularStateFile(target, "Agent run record")) {
    const stat = await fs.lstat(target);
    if (stat.isSymbolicLink() || !stat.isFile()) throw new Error("Agent run record must be a regular file.");
  }
  const serialized = JSON.stringify(run, null, 2) + "\n";
  if (Buffer.byteLength(serialized, "utf8") > MAX_AGENT_RUN_BYTES) {
    throw new Error("Agent run checkpoint exceeds the 160 KB storage limit.");
  }
  await writeFileAtomic(target, serialized);
}

function summarizeAgentVerification(value) {
  if (!value || typeof value !== "object") return null;
  return {
    status: typeof value.status === "string" ? value.status.slice(0, 64) : "unknown",
    passed: value.passed === true,
    checks: Array.isArray(value.checks) ? value.checks.slice(0, 8).map(check => ({
      task: String(check.task || "").slice(0, 40),
      passed: check.passed === true,
      exitCode: Number.isInteger(check.exitCode) ? check.exitCode : null,
      timedOut: check.timedOut === true,
      durationMs: Number.isFinite(check.durationMs) ? Math.max(0, Math.floor(check.durationMs)) : null
    })) : [],
    reviewRequiredReasons: Array.isArray(value.reviewRequiredReasons)
      ? value.reviewRequiredReasons.slice(0, 8).map(item => String(item).slice(0, 300))
      : []
  };
}

async function pruneAgentRunRecords(root) {
  const directory = await agentRunDirectory(root);
  const entries = await fs.readdir(directory, { withFileTypes: true });
  const records = [];
  for (const entry of entries) {
    if (!entry.isFile() || !/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\.json$/i.test(entry.name)) continue;
    const target = path.join(directory, entry.name);
    if (!(await assertRegularStateFile(target, "Agent run record"))) continue;
    const stat = await fs.stat(target);
    records.push({ path: target, modifiedAt: stat.mtimeMs });
  }
  records.sort((a, b) => b.modifiedAt - a.modifiedAt);
  for (const record of records.slice(MAX_AGENT_RUNS)) await fs.rm(record.path);
}

export async function createAgentRun(inputRoot, goal) {
  const root = await resolveWorkspace(inputRoot);
  if (typeof goal !== "string" || !goal.trim() || goal.length > 5000) {
    throw new Error("Agent run goal must contain 1-5,000 characters.");
  }
  const now = new Date().toISOString();
  const run = {
    schemaVersion: 1,
    id: crypto.randomUUID(),
    root,
    goal: goal.trim(),
    status: "running",
    phase: "queued",
    message: "Agent task queued.",
    iteration: 0,
    createdAt: now,
    updatedAt: now,
    cancelRequested: false,
    model: null,
    planSteps: [],
    selectedContext: [],
    attempts: [],
    toolCalls: [],
    verification: null,
    changedFiles: [],
    proposalIds: [],
    checkpoint: { stage: "created", iteration: 0 },
    recoveryPatch: null,
    events: [{ at: now, phase: "queued", message: "Agent task created." }],
    error: null
  };
  await writeAgentRunRecord(root, run);
  await pruneAgentRunRecords(root);
  await appendAudit(root, { action: "agent_run_created", runId: run.id, goalLength: run.goal.length });
  return run;
}

export async function getAgentRun(inputRoot, runId) {
  const root = await resolveWorkspace(inputRoot);
  return await readAgentRunRecord(root, runId);
}

async function updateAgentRunUnlocked(inputRoot, runId, patch = {}) {
  const root = await resolveWorkspace(inputRoot);
  if (!patch || typeof patch !== "object" || Array.isArray(patch)) throw new Error("Agent run update must be an object.");
  const run = await readAgentRunRecord(root, runId);
  if (!run) throw new Error("Agent run was not found.");
  const next = { ...run, updatedAt: new Date().toISOString() };

  if (typeof patch.status === "string") {
    if (!AGENT_RUN_STATUSES.has(patch.status)) throw new Error("Agent run status is invalid.");
    if (!AGENT_RUN_TRANSITIONS[run.status]?.has(patch.status)) {
      throw new Error("Agent run cannot transition from " + run.status + " to " + patch.status + ".");
    }
    next.status = patch.status;
  }
  if (patch.model && typeof patch.model === "object" && !Array.isArray(patch.model)) {
    next.model = {
      provider: String(patch.model.provider || "unknown").slice(0, 80),
      name: String(patch.model.name || "unknown").slice(0, 120),
      contextWindowTokens: Number.isFinite(patch.model.contextWindowTokens)
        ? Math.max(0, Math.min(1_000_000, Math.floor(patch.model.contextWindowTokens))) : 0,
      contextBudgetTokens: Number.isFinite(patch.model.contextBudgetTokens)
        ? Math.max(0, Math.min(1_000_000, Math.floor(patch.model.contextBudgetTokens))) : 0,
      outputBudgetTokens: Number.isFinite(patch.model.outputBudgetTokens)
        ? Math.max(0, Math.min(100_000, Math.floor(patch.model.outputBudgetTokens))) : 0
    };
  }
  if (typeof patch.phase === "string") next.phase = patch.phase.slice(0, 64);
  if (typeof patch.message === "string") next.message = patch.message.replace(/[\r\n]+/g, " ").slice(0, 500);
  if (Number.isInteger(patch.iteration)) next.iteration = Math.max(0, Math.min(20, patch.iteration));
  if (typeof patch.cancelRequested === "boolean") next.cancelRequested = patch.cancelRequested;
  if (typeof patch.error === "string" || patch.error === null) next.error = patch.error ? patch.error.slice(0, 1200) : null;
  if (Array.isArray(patch.planSteps)) next.planSteps = patch.planSteps.slice(0, 12).map(step => ({
    id: String(step.id || "").slice(0, 40),
    title: String(step.title || "").slice(0, 120),
    status: String(step.status || "pending").slice(0, 40),
    detail: String(step.detail || "").slice(0, 400)
  }));
  if (Array.isArray(patch.selectedContext)) next.selectedContext = patch.selectedContext.slice(0, 20).map(item => ({
    path: String(item.path || "").slice(0, 1024),
    score: Number.isFinite(item.score) ? Number(item.score) : 0,
    sha256: typeof item.sha256 === "string" ? item.sha256.slice(0, 64) : ""
  }));
  if (Array.isArray(patch.attempts)) next.attempts = patch.attempts.slice(-12).map(item => ({
    iteration: Number.isInteger(item.iteration) ? item.iteration : 0,
    status: String(item.status || "unknown").slice(0, 64),
    promptTokens: Number.isFinite(item.promptTokens) ? item.promptTokens : null,
    outputTokens: Number.isFinite(item.outputTokens) ? item.outputTokens : null,
    changedFiles: Array.isArray(item.changedFiles) ? item.changedFiles.slice(0, 8).map(value => String(value).slice(0, 1024)) : []
  }));
  if (Array.isArray(patch.toolCalls)) next.toolCalls = patch.toolCalls.slice(-60).map(item => ({
    runId: String(item.runId || "").slice(0, 64),
    name: String(item.name || "").slice(0, 80),
    status: String(item.status || "").slice(0, 32),
    permission: String(item.permission || "").slice(0, 40),
    durationMs: Number.isFinite(item.durationMs) ? Math.max(0, Math.floor(item.durationMs)) : 0,
    auditRecorded: item.auditRecorded !== false
  }));
  if (patch.verification !== undefined) next.verification = summarizeAgentVerification(patch.verification);
  if (Array.isArray(patch.changedFiles)) next.changedFiles = patch.changedFiles.slice(0, 12).map(item => ({
    path: String(item.path || "").slice(0, 1024),
    existedBefore: item.existedBefore === true,
    baseHash: typeof item.baseHash === "string" ? item.baseHash.slice(0, 64) : null,
    afterHash: typeof item.afterHash === "string" ? item.afterHash.slice(0, 64) : "",
    reason: String(item.reason || "").slice(0, 500)
  }));
  if (Array.isArray(patch.proposalIds)) next.proposalIds = patch.proposalIds.slice(0, 24).filter(validAgentRunId);
  if (patch.checkpoint && typeof patch.checkpoint === "object" && !Array.isArray(patch.checkpoint)) {
    next.checkpoint = {
      stage: String(patch.checkpoint.stage || next.checkpoint?.stage || "unknown").slice(0, 64),
      iteration: Number.isInteger(patch.checkpoint.iteration)
        ? Math.max(0, Math.min(20, patch.checkpoint.iteration))
        : next.iteration
    };
  }
  if (patch.recoveryPatch === null) {
    next.recoveryPatch = null;
  } else if (patch.recoveryPatch && typeof patch.recoveryPatch === "object" && !Array.isArray(patch.recoveryPatch)) {
    let incomingPatchBytes;
    try {
      incomingPatchBytes = Buffer.byteLength(JSON.stringify(patch.recoveryPatch), "utf8");
    } catch {
      throw new Error("The saved patch checkpoint must be serializable JSON.");
    }
    if (!Number.isFinite(incomingPatchBytes) || incomingPatchBytes > 100 * 1024) {
      throw new Error("The saved patch checkpoint exceeds the 100 KB input limit.");
    }
    const recoveryPatch = {
      summary: String(patch.recoveryPatch.summary || "").slice(0, 1000),
      edits: Array.isArray(patch.recoveryPatch.edits) ? patch.recoveryPatch.edits.slice(0, 6).map(edit => ({
        path: String(edit.path || "").slice(0, 1024),
        find: typeof edit.find === "string" ? edit.find.slice(0, 30_000) : "",
        replace: typeof edit.replace === "string" ? edit.replace.slice(0, 30_000) : "",
        reason: String(edit.reason || "").slice(0, 500)
      })) : []
    };
    if (!recoveryPatch.edits.length || recoveryPatch.edits.some(edit => !edit.path || (!edit.find && !edit.replace))) {
      throw new Error("The saved patch checkpoint is incomplete.");
    }
    if (Buffer.byteLength(JSON.stringify(recoveryPatch), "utf8") > 100 * 1024) {
      throw new Error("The saved patch checkpoint exceeds the 100 KB limit.");
    }
    next.recoveryPatch = recoveryPatch;
  }
  if (patch.event && typeof patch.event === "object" && !Array.isArray(patch.event)) {
    next.events = [...(Array.isArray(next.events) ? next.events : []), {
      at: new Date().toISOString(),
      phase: String(patch.event.phase || next.phase).slice(0, 64),
      message: String(patch.event.message || "").replace(/[\r\n]+/g, " ").slice(0, 500),
      iteration: Number.isInteger(patch.event.iteration) ? Math.max(0, patch.event.iteration) : next.iteration
    }].slice(-80);
  }

  await writeAgentRunRecord(root, next);
  return next;
}

const agentRunUpdateQueues = new Map();

async function withAgentRunUpdateQueue(key, action) {
  const previous = agentRunUpdateQueues.get(key) || Promise.resolve();
  const current = previous.catch(() => {}).then(action);
  agentRunUpdateQueues.set(key, current);
  try {
    return await current;
  } finally {
    if (agentRunUpdateQueues.get(key) === current) agentRunUpdateQueues.delete(key);
  }
}

export async function updateAgentRun(inputRoot, runId, patch = {}) {
  const root = await resolveWorkspace(inputRoot);
  if (!validAgentRunId(runId)) throw new Error("A valid agent run ID is required.");
  return withAgentRunUpdateQueue(root + "::" + runId, () =>
    updateAgentRunUnlocked(root, runId, patch)
  );
}

export async function listAgentRuns(inputRoot, limit = 20) {
  const root = await resolveWorkspace(inputRoot);
  const directory = await agentRunDirectory(root);
  const boundedLimit = Math.max(1, Math.min(40, Math.floor(Number(limit) || 20)));
  const entries = await fs.readdir(directory, { withFileTypes: true });
  const runs = [];
  for (const entry of entries) {
    if (!entry.isFile() || !/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\.json$/i.test(entry.name)) continue;
    const runId = entry.name.slice(0, -5);
    try {
      const run = await readAgentRunRecord(root, runId);
      if (run) runs.push(run);
    } catch {
      // One damaged record must not hide other recoverable task history.
    }
  }
  runs.sort((a, b) => String(b.updatedAt).localeCompare(String(a.updatedAt)));
  return runs.slice(0, boundedLimit);
}

export async function recoverInterruptedAgentRuns(inputRoot) {
  const root = await resolveWorkspace(inputRoot);
  const runs = await listAgentRuns(root, MAX_AGENT_RUNS);
  const recovered = [];
  for (const run of runs) {
    if (run.status !== "running" && run.status !== "queued") continue;
    const updated = await updateAgentRun(root, run.id, {
      status: "interrupted",
      phase: "interrupted",
      message: "The local service stopped before this task completed. Inspect its checkpoint before retrying.",
      event: { phase: "interrupted", message: "Recovered an unfinished run after a service restart." }
    });
    recovered.push(updated);
    await appendAudit(root, { action: "agent_run_interrupted", runId: run.id, checkpoint: run.checkpoint?.stage || "unknown" });
  }
  return recovered;
}

export async function requestAgentRunCancellation(inputRoot, runId) {
  const root = await resolveWorkspace(inputRoot);
  if (!validAgentRunId(runId)) throw new Error("A valid agent run ID is required.");
  return withAgentRunUpdateQueue(root + "::" + runId, async () => {
    const run = await readAgentRunRecord(root, runId);
    if (!run) throw new Error("Agent run was not found.");
    if (run.status !== "running" && run.status !== "queued") return run;
    return updateAgentRunUnlocked(root, runId, {
      cancelRequested: true,
      phase: "cancelling",
      message: "Cancellation requested.",
      event: { phase: "cancelling", message: "User requested cancellation." }
    });
  });
}

export async function createProjectPlan(inputRoot, goal) {
  const root = await resolveWorkspace(inputRoot);
  if (typeof goal !== "string" || goal.trim().length < 8 || goal.length > 4000) {
    throw new Error("Project goal must contain 8-4000 characters.");
  }
  const inspection = await inspectWorkspace(root);
  const id = crypto.randomUUID();
  const plan = {
    id,
    createdAt: new Date().toISOString(),
    root,
    goal: goal.trim(),
    planningMode: "deterministic-template",
    notice: "This is a workflow skeleton, not an AI-reasoned implementation plan. Review and refine task details before execution.",
    projectSnapshot: {
      name: inspection.name,
      fileCount: inspection.fileCount,
      frameworks: inspection.frameworks,
      scripts: inspection.scripts
    },
    tasks: [
      { id: "T1", phase: "discover", title: "Map the existing project", description: "Inspect manifests, source modules, dependencies, entry points, tests, and existing conventions.", status: "pending", acceptance: ["Project map reviewed", "Relevant files identified"] },
      { id: "T2", phase: "specify", title: "Translate the goal into requirements", description: "Write behavior, edge cases, UI states, constraints, and measurable acceptance criteria before editing.", status: "pending", acceptance: ["Requirements are concrete", "Acceptance criteria are testable"] },
      { id: "T3", phase: "design", title: "Design the smallest safe change set", description: "Identify affected modules, API contracts, data flow, and backwards-compatibility risks.", status: "pending", acceptance: ["Files and interfaces mapped", "Risks and dependencies recorded"] },
      { id: "T4", phase: "implement", title: "Implement one vertical slice at a time", description: "Propose focused file changes, review diffs, apply approved changes, and preserve a recoverable backup.", status: "pending", acceptance: ["Every change has a reviewable proposal", "No unrelated files modified"] },
      { id: "T5", phase: "verify", title: "Run build, tests, and static checks", description: "Execute available project scripts, inspect failures, fix regressions, and repeat checks.", status: "pending", acceptance: ["Relevant checks have recorded results", "Failures are resolved or documented"] },
      { id: "T6", phase: "review", title: "Review security, UX, and documentation", description: "Inspect error states, permissions, path handling, dependency changes, accessibility, and user documentation.", status: "pending", acceptance: ["Security-sensitive changes reviewed", "Documentation matches observed behavior"] }
    ]
  };
  const target = path.join(stateDirectory(root), "plans", "current.json");
  await writeFileAtomic(target, JSON.stringify(plan, null, 2) + "\n");
  await appendAudit(root, { action: "create_project_plan", planId: id, fileCount: inspection.fileCount });
  return { planPath: path.relative(root, target).split(path.sep).join("/"), ...plan };
}

function summarizeChange(before, after) {
  const oldLines = before === null ? [] : before.split(/\r?\n/);
  const newLines = after.split(/\r?\n/);
  let prefix = 0;
  while (prefix < oldLines.length && prefix < newLines.length && oldLines[prefix] === newLines[prefix]) prefix++;
  let oldSuffix = oldLines.length;
  let newSuffix = newLines.length;
  while (oldSuffix > prefix && newSuffix > prefix && oldLines[oldSuffix - 1] === newLines[newSuffix - 1]) {
    oldSuffix--;
    newSuffix--;
  }
  return {
    oldLines: oldLines.length,
    newLines: newLines.length,
    addedLines: before === null ? newLines.length : Math.max(0, newSuffix - prefix),
    removedLines: before === null ? 0 : Math.max(0, oldSuffix - prefix),
    preview: newLines.slice(Math.max(0, prefix - 2), Math.min(newLines.length, prefix + 10)).join("\n").slice(0, 2000)
  };
}

export async function proposeFileChange(inputRoot, relativePath, newContent, expectedBaseHash = undefined) {
  const root = await resolveWorkspace(inputRoot);
  if (typeof newContent !== "string" || Buffer.byteLength(newContent, "utf8") > MAX_TEXT_BYTES) {
    throw new Error("Proposed file content must be text no larger than 512 KB.");
  }
  const resolved = await resolveFile(root, relativePath, { allowMissing: true });
  if (resolved.stat && resolved.stat.size > MAX_TEXT_BYTES) {
    throw new Error("The existing target file exceeds the 512 KB proposal limit.");
  }
  const before = resolved.stat ? await fs.readFile(resolved.target, "utf8") : null;
  const beforeHash = before === null ? null : sha256(before);
  if (expectedBaseHash !== undefined && beforeHash !== expectedBaseHash) {
    throw new Error("The target file changed after SYNTRA read it for generation. Reload the file and generate/review a new proposal.");
  }
  const id = crypto.randomUUID();
  const proposal = {
    id,
    createdAt: new Date().toISOString(),
    root,
    path: resolved.relativePath,
    beforeHash,
    afterHash: sha256(newContent),
    existedBefore: before !== null,
    content: newContent,
    summary: summarizeChange(before, newContent)
  };
  const proposalPath = path.join(stateDirectory(root), "proposals", id + ".json");
  await writeFileAtomic(proposalPath, JSON.stringify(proposal, null, 2) + "\n");
  await appendAudit(root, {
    action: "propose_file_change",
    proposalId: id,
    path: resolved.relativePath,
    beforeHash,
    afterHash: proposal.afterHash
  });
  return {
    id, createdAt: proposal.createdAt, path: proposal.path,
    existedBefore: proposal.existedBefore, beforeHash,
    afterHash: proposal.afterHash, summary: proposal.summary,
    proposalPath: path.relative(root, proposalPath).split(path.sep).join("/")
  };
}

export async function listProposals(inputRoot) {
  const root = await resolveWorkspace(inputRoot);
  const directory = path.join(stateDirectory(root), "proposals");
  const entries = await fs.readdir(directory, { withFileTypes: true }).catch(() => []);
  const proposals = [];
  for (const entry of entries) {
    if (!entry.isFile() || !/^[a-f0-9-]+\.json$/i.test(entry.name)) continue;
    try {
      const proposalPath = path.join(directory, entry.name);
      if (!(await assertRegularStateFile(proposalPath, "Saved proposal"))) continue;
      const proposal = JSON.parse(await fs.readFile(proposalPath, "utf8"));
      proposals.push({
        id: proposal.id, createdAt: proposal.createdAt, path: proposal.path,
        existedBefore: proposal.existedBefore, beforeHash: proposal.beforeHash,
        afterHash: proposal.afterHash, summary: proposal.summary
      });
    } catch {}
  }
  proposals.sort((a, b) => String(b.createdAt).localeCompare(String(a.createdAt)));
  return { root, proposals };
}


/**
 * Rebuild the actionable proposal queue from disk. A proposal is pending only
 * while its original file hash still matches; already-applied and stale
 * proposals are kept out of the apply queue and reported separately.
 */
export async function listPendingProposals(inputRoot) {
  const root = await resolveWorkspace(inputRoot);
  const directory = path.join(stateDirectory(root), "proposals");
  const entries = await fs.readdir(directory, { withFileTypes: true }).catch(() => []);
  const pending = [];
  const stale = [];

  for (const entry of entries) {
    if (!entry.isFile() || !/^[a-f0-9-]{36}\.json$/i.test(entry.name)) continue;
    const id = entry.name.slice(0, -5);
    const proposalPath = path.join(directory, entry.name);
    try {
      if (!(await assertRegularStateFile(proposalPath, "Saved proposal"))) continue;
      const proposal = JSON.parse(await fs.readFile(proposalPath, "utf8"));
      if (!proposal || proposal.id !== id || path.resolve(String(proposal.root || "")) !== root) continue;

      const metadata = {
        id,
        createdAt: proposal.createdAt,
        path: proposal.path,
        existedBefore: proposal.existedBefore,
        beforeHash: proposal.beforeHash,
        afterHash: proposal.afterHash,
        summary: proposal.summary
      };
      const resolved = await resolveFile(root, proposal.path, { allowMissing: true });
      if (resolved.stat && resolved.stat.size > MAX_TEXT_BYTES) {
        stale.push({ ...metadata, reason: "The source file now exceeds the proposal size limit." });
        continue;
      }
      const currentContent = resolved.stat ? await fs.readFile(resolved.target, "utf8") : null;
      const currentHash = currentContent === null ? null : sha256(currentContent);

      if (typeof proposal.content !== "string"
        || Buffer.byteLength(proposal.content, "utf8") > MAX_TEXT_BYTES
        || typeof proposal.afterHash !== "string"
        || sha256(proposal.content) !== proposal.afterHash) {
        stale.push({ ...metadata, reason: "The saved proposal failed its content-integrity check." });
        continue;
      }

      if (currentHash === proposal.afterHash) continue;
      if (currentHash !== proposal.beforeHash) {
        stale.push({ ...metadata, reason: "The source file changed after this proposal was staged." });
        continue;
      }
      pending.push(metadata);
    } catch (error) {
      if (error instanceof Error && error.message.includes("must be a regular file, not a symbolic link")) throw error;
      // A malformed or unreadable proposal must never be treated as applicable.
    }
  }

  const sortByDate = (a, b) => String(b.createdAt).localeCompare(String(a.createdAt));
  pending.sort(sortByDate);
  stale.sort(sortByDate);
  return { root, pending, stale };
}

/**
 * Explicitly discard saved proposal records. This does not modify project files.
 */
export async function discardFileProposals(inputRoot, proposalIds, confirmation) {
  const root = await resolveWorkspace(inputRoot);
  if (confirmation !== true) throw new Error("Discarding proposals requires explicit confirmation.");
  if (!Array.isArray(proposalIds) || proposalIds.length < 1 || proposalIds.length > 24) {
    throw new Error("A discard operation must contain between 1 and 24 proposal IDs.");
  }
  if (proposalIds.some(id => typeof id !== "string" || !/^[a-f0-9-]{36}$/i.test(id))) {
    throw new Error("Every proposal identifier must be a valid UUID.");
  }
  if (new Set(proposalIds).size !== proposalIds.length) {
    throw new Error("A proposal may appear only once in a discard operation.");
  }

  const staged = [];
  for (const id of proposalIds) {
    const proposalPath = path.join(stateDirectory(root), "proposals", id + ".json");
    if (!(await assertRegularStateFile(proposalPath, "Saved proposal"))) {
      throw new Error("Proposal " + id + " is missing; no proposals were discarded.");
    }
    let proposal;
    try {
      proposal = JSON.parse(await fs.readFile(proposalPath, "utf8"));
    } catch {
      throw new Error("Proposal " + id + " is unreadable; no proposals were discarded.");
    }
    if (!proposal || proposal.id !== id || path.resolve(String(proposal.root || "")) !== root) {
      throw new Error("Proposal " + id + " belongs to a different workspace; no proposals were discarded.");
    }
    staged.push({ id, path: proposal.path, proposalPath });
  }

  for (const item of staged) await fs.rm(item.proposalPath);
  await appendAudit(root, {
    action: "discard_file_proposals",
    proposalIds: staged.map(item => item.id),
    paths: staged.map(item => item.path)
  });
  return { root, discarded: staged.map(item => ({ id: item.id, path: item.path })) };
}

export async function applyFileProposal(inputRoot, proposalId, confirmation) {
  const transaction = await applyFileProposals(inputRoot, [proposalId], confirmation);
  return {
    ...transaction.applied[0],
    transactionId: transaction.transactionId,
    backupDirectory: transaction.backupDirectory
  };
}


/**
 * Apply a reviewed set of proposals as one transaction. Every path and source hash is
 * checked before the first write; all existing contents are backed up together; and
 * a later write conflict triggers best-effort rollback without overwriting a concurrent
 * edit made by someone else.
 */
export async function applyFileProposals(inputRoot, proposalIds, confirmation) {
  const root = await resolveWorkspace(inputRoot);
  if (confirmation !== true) {
    throw new Error("File changes require explicit confirmation: set confirm=true after reviewing the proposals.");
  }
  if (!Array.isArray(proposalIds) || proposalIds.length < 1 || proposalIds.length > 6) {
    throw new Error("A transaction must contain between 1 and 6 proposal IDs.");
  }
  if (proposalIds.some(id => typeof id !== "string" || !/^[a-f0-9-]{36}$/i.test(id))) {
    throw new Error("Every proposal identifier must be a valid UUID.");
  }
  if (new Set(proposalIds).size !== proposalIds.length) {
    throw new Error("A proposal may appear only once in a transaction.");
  }

  const staged = [];
  const seenPaths = new Set();
  for (const id of proposalIds) {
    const proposalPath = path.join(stateDirectory(root), "proposals", id + ".json");
    let proposal;
    try {
      if (!(await assertRegularStateFile(proposalPath, "Saved proposal"))) {
        throw new Error("Proposal is missing.");
      }
      proposal = JSON.parse(await fs.readFile(proposalPath, "utf8"));
    } catch (error) {
      if (error instanceof Error && error.message.includes("must be a regular file, not a symbolic link")) throw error;
      throw new Error("Proposal " + id + " is missing or unreadable; no files were changed.");
    }
    if (!proposal || path.resolve(String(proposal.root || "")) !== root) {
      throw new Error("Proposal " + id + " belongs to a different workspace.");
    }
    if (typeof proposal.content !== "string"
      || Buffer.byteLength(proposal.content, "utf8") > MAX_TEXT_BYTES
      || typeof proposal.afterHash !== "string"
      || sha256(proposal.content) !== proposal.afterHash) {
      throw new Error("Proposal " + id + " failed its content-integrity check; no files were changed.");
    }

    const resolved = await resolveFile(root, proposal.path, { allowMissing: true });
    if (seenPaths.has(resolved.relativePath)) {
      throw new Error("A transaction cannot contain multiple proposals for the same file: " + resolved.relativePath);
    }
    seenPaths.add(resolved.relativePath);
    const beforeContent = resolved.stat ? await fs.readFile(resolved.target, "utf8") : null;
    const beforeHash = beforeContent === null ? null : sha256(beforeContent);
    if (beforeHash !== proposal.beforeHash) {
      throw new Error("Refusing to apply " + resolved.relativePath + ": the source changed after the proposal was staged. No files were changed.");
    }
    if (beforeHash === proposal.afterHash) {
      throw new Error("Proposal " + id + " makes no content change; recreate it against the latest file.");
    }
    staged.push({ id, proposal, relativePath: resolved.relativePath, target: resolved.target, beforeContent, beforeHash });
  }

  const transactionId = new Date().toISOString().replace(/[:.]/g, "-") + "-" + crypto.randomUUID();
  const backupRoot = path.join(stateDirectory(root), "backups", transactionId);
  // Stage every backup before touching any project file.
  for (const item of staged) {
    if (item.beforeContent === null) continue;
    const backupPath = path.resolve(backupRoot, item.relativePath.split("/").join(path.sep));
    if (!isWithinRoot(backupRoot, backupPath)) {
      throw new Error("Backup path validation failed for " + item.relativePath + "; no files were changed.");
    }
    await writeFileAtomic(backupPath, item.beforeContent);
  }

  const applied = [];
  try {
    for (const item of staged) {
      // Close the most obvious time-of-check/time-of-use gap before each write.
      const currentTarget = await resolveFile(root, item.relativePath, { allowMissing: true });
      const currentContent = currentTarget.stat ? await fs.readFile(currentTarget.target, "utf8") : null;
      const currentHash = currentContent === null ? null : sha256(currentContent);
      if (currentHash !== item.beforeHash) {
        throw new Error("Refusing to apply " + item.relativePath + ": the source changed during this transaction.");
      }
      await writeFileAtomic(currentTarget.target, item.proposal.content);
      applied.push({ ...item, target: currentTarget.target });
    }

    await appendAudit(root, {
      action: staged.length === 1 ? "apply_file_proposal" : "apply_file_proposals_transaction",
      transactionId,
      proposalIds: staged.map(item => item.id),
      files: staged.map(item => ({
        path: item.relativePath,
        beforeHash: item.beforeHash,
        afterHash: item.proposal.afterHash,
        backupCreated: item.beforeContent !== null
      }))
    });
  } catch (error) {
    const rollbackErrors = [];
    for (const item of [...applied].reverse()) {
      try {
        const currentTarget = await resolveFile(root, item.relativePath, { allowMissing: true });
        const currentContent = currentTarget.stat ? await fs.readFile(currentTarget.target, "utf8") : null;
        const currentHash = currentContent === null ? null : sha256(currentContent);
        // Do not overwrite a third-party edit that raced with the transaction.
        if (currentHash !== item.proposal.afterHash) {
          rollbackErrors.push(item.relativePath + " changed again during rollback; left untouched");
          continue;
        }
        if (item.beforeContent === null) {
          if (currentTarget.stat) await fs.rm(currentTarget.target);
        } else {
          await writeFileAtomic(currentTarget.target, item.beforeContent);
        }
      } catch (rollbackError) {
        rollbackErrors.push(item.relativePath + ": " + (rollbackError instanceof Error ? rollbackError.message : String(rollbackError)));
      }
    }
    await appendAudit(root, {
      action: "rollback_file_proposals_transaction",
      transactionId,
      proposalIds: staged.map(item => item.id),
      rolledBackPaths: applied.map(item => item.relativePath),
      rollbackErrors
    }).catch(() => {});

    const message = error instanceof Error ? error.message : String(error);
    if (rollbackErrors.length) {
      throw new Error("Proposal transaction failed: " + message
        + ". Automatic rollback was incomplete; inspect backups at " + path.relative(root, backupRoot).split(path.sep).join("/")
        + ". Details: " + rollbackErrors.join("; "));
    }
    throw new Error("Proposal transaction failed and applied changes were rolled back: " + message
      + ". Backups remain at " + path.relative(root, backupRoot).split(path.sep).join("/"));
  }

  return {
    transactionId,
    backupDirectory: path.relative(root, backupRoot).split(path.sep).join("/"),
    applied: staged.map(item => ({
      applied: true,
      proposalId: item.id,
      path: item.relativePath,
      beforeHash: item.beforeHash,
      afterHash: item.proposal.afterHash,
      backupCreated: item.beforeContent !== null
    }))
  };
}

function pythonTestFiles(files) {
  return files.filter(file => file.extension.toLowerCase() === ".py"
    && (/^test_.+\.py$/i.test(path.posix.basename(file.relative))
      || /.+_test\.py$/i.test(path.posix.basename(file.relative))));
}

function nodeTestFiles(files) {
  return files.filter(file => [".js", ".mjs", ".cjs"].includes(file.extension.toLowerCase())
    && (/(^|\/)(?:test|tests|__tests__|__specs__)\//i.test(file.relative)
      || /\.(?:test|spec)\.(?:js|mjs|cjs)$/i.test(file.relative)));
}

function commonPythonTestDirectory(files) {
  const counts = new Map();
  for (const file of files) {
    const parent = path.posix.dirname(file.relative);
    counts.set(parent, (counts.get(parent) || 0) + 1);
  }
  return [...counts.entries()].sort((a, b) => b[1] - a[1] || a[0].length - b[0].length)[0]?.[0] || ".";
}

async function hasPytestConfiguration(root, files) {
  if (["pytest.ini", "tox.ini", "setup.cfg"].some(name => fsSync.existsSync(path.join(root, name)))) return true;
  const indicators = files.filter(file => ["pyproject.toml", "requirements.txt", "requirements-dev.txt"].includes(path.posix.basename(file.relative).toLowerCase()));
  for (const file of indicators) {
    try {
      const content = await fs.readFile(path.join(root, file.relative.split("/").join(path.sep)), "utf8");
      if (/\[tool\.pytest|(?:^|\s)pytest(?:\s|[<=>;\[]|$)/im.test(content)) return true;
    } catch {}
  }
  return false;
}

function executableCheck(kind, executable, args, label) {
  return { kind, executable, args, command: [executable, ...args].join(" "), script: null, label };
}

function pythonExecutable(root) {
  const candidates = process.platform === "win32"
    ? [[".venv", "Scripts", "python.exe"], ["venv", "Scripts", "python.exe"], ["env", "Scripts", "python.exe"]]
    : [[".venv", "bin", "python"], ["venv", "bin", "python"], ["env", "bin", "python"]];
  for (const parts of candidates) {
    const candidate = path.join(root, ...parts);
    try {
      if (fsSync.statSync(candidate).isFile()) return candidate;
    } catch {}
  }
  return process.platform === "win32" ? "python" : "python3";
}

function terminateProcessTree(child, force = false) {
  if (!child || child.exitCode !== null || !child.pid) return;
  if (process.platform === "win32") {
    try {
      const killer = spawn("taskkill.exe", ["/pid", String(child.pid), "/T", "/F"], {
        windowsHide: true, stdio: "ignore"
      });
      killer.once("error", () => {
        try { child.kill(force ? "SIGKILL" : "SIGTERM"); } catch {}
      });
      killer.unref();
    } catch {
      try { child.kill(force ? "SIGKILL" : "SIGTERM"); } catch {}
    }
    return;
  }
  try {
    // All verification commands are placed in their own process group below.
    process.kill(-child.pid, force ? "SIGKILL" : "SIGTERM");
  } catch {
    try { child.kill(force ? "SIGKILL" : "SIGTERM"); } catch {}
  }
}

export async function chooseCheck(root, task, packageJson) {
  const scripts = packageJson && packageJson.scripts && typeof packageJson.scripts === "object"
    ? packageJson.scripts : {};
  const aliases = {
    build: ["build"],
    test: ["test"],
    lint: ["lint"],
    typecheck: ["typecheck", "check"]
  };
  const allowed = aliases[task];
  if (!allowed) throw new Error("Check task must be one of: build, test, lint, typecheck.");
  const packageScript = allowed.find(name => typeof scripts[name] === "string" && scripts[name].trim());
  if (packageScript) return { kind: "npm-script", command: "npm run " + packageScript, script: packageScript };

  const exists = name => fsSync.existsSync(path.join(root, name));
  const topFiles = await fs.readdir(root).catch(() => []);
  const hasCargo = exists("Cargo.toml");
  const hasGo = exists("go.mod");
  const dotnetSolution = topFiles.some(name => name.toLowerCase().endsWith(".sln"));
  const topDotnetProjects = topFiles.filter(name => name.toLowerCase().endsWith(".csproj"));
  const sourceFiles = (task === "test" || (!hasCargo && !hasGo && !dotnetSolution && topDotnetProjects.length === 0))
    ? await walkFiles(root) : [];
  const dotnetProjects = [...new Set([
    ...topDotnetProjects,
    ...sourceFiles.filter(file => file.relative.toLowerCase().endsWith(".csproj")).map(file => file.relative)
  ])];
  const pythonFiles = pythonTestFiles(sourceFiles);
  const jsFiles = nodeTestFiles(sourceFiles);

  if (task === "build") {
    const builders = [];
    if (hasCargo) builders.push(executableCheck("command", "cargo", ["build"], "Rust/Cargo"));
    if (hasGo) builders.push(executableCheck("command", "go", ["build", "./..."], "Go"));
    if (dotnetSolution) builders.push(executableCheck("command", "dotnet", ["build"], ".NET solution"));
    else if (dotnetProjects.length === 1) builders.push(executableCheck("command", "dotnet", ["build", dotnetProjects[0]], ".NET project"));
    if (builders.length === 1) return builders[0];
    if (builders.length > 1) throw new Error("Multiple build systems were detected. Add an explicit build script to package.json so SYNTRA can run the intended build.");
  }

  if (task === "test") {
    const testRunners = [];
    const isPythonProject = ["pyproject.toml", "requirements.txt", "requirements-dev.txt", "pytest.ini", "setup.py"].some(exists);
    const isPythonTestProject = isPythonProject && pythonFiles.length > 0;
    if (isPythonTestProject) {
      const interpreter = pythonExecutable(root);
      if (await hasPytestConfiguration(root, sourceFiles)) {
        testRunners.push(executableCheck("python-pytest", interpreter, ["-m", "pytest"], "Python/pytest"));
      } else {
        const startDirectory = commonPythonTestDirectory(pythonFiles);
        testRunners.push(executableCheck(
          "python-unittest",
          interpreter,
          ["-m", "unittest", "discover", "-s", startDirectory, "-v"],
          "Python/unittest"
        ));
      }
    }
    if (hasCargo) testRunners.push(executableCheck("command", "cargo", ["test"], "Rust/Cargo"));
    if (hasGo) testRunners.push(executableCheck("command", "go", ["test", "./..."], "Go"));
    const dotnetTestProjects = [];
    for (const projectPath of dotnetProjects) {
      const looksLikeTestProject = /(?:\.tests?|\.(?:test|spec))\.csproj$/i.test(path.posix.basename(projectPath));
      let containsTestSdk = false;
      if (!looksLikeTestProject) {
        try {
          const projectText = await fs.readFile(path.join(root, projectPath.split("/").join(path.sep)), "utf8");
          containsTestSdk = /Microsoft\.NET\.Test\.Sdk|\bxunit\b|\bnunit\b|\bmstest\b/i.test(projectText);
        } catch {}
      }
      if (looksLikeTestProject || containsTestSdk) dotnetTestProjects.push(projectPath);
    }
    if (dotnetSolution && dotnetTestProjects.length > 0) {
      testRunners.push(executableCheck("command", "dotnet", ["test"], ".NET solution"));
    } else if (!dotnetSolution) {
      for (const testProject of dotnetTestProjects) {
        testRunners.push(executableCheck("command", "dotnet", ["test", testProject], ".NET test project"));
      }
    }
    if (jsFiles.length > 0) testRunners.push(executableCheck("node-test", "node", ["--test"], "Node.js"));

    if (testRunners.length === 1) return testRunners[0];
    if (testRunners.length > 1) {
      throw new Error("Multiple test ecosystems were detected without a configured test script. Add a package.json test script or specify one test runner to avoid reporting a partial test pass as full verification.");
    }
    if (isPythonProject && pythonFiles.length === 0) {
      throw new Error("A Python project was detected, but no test_*.py or *_test.py files were found. SYNTRA cannot call this a verified test pass.");
    }
    throw new Error("No supported test runner and matching test files were found. Add an explicit test script or test configuration; an empty tests directory is not verification.");
  }

  throw new Error("No supported " + task + " command was found in the project manifest.");
}

export async function discoverProjectChecks(inputRoot) {
  const root = await resolveWorkspace(inputRoot);
  const packageJson = await readJsonIfPresent(root, "package.json");
  const checks = [];
  const unavailable = [];
  for (const task of ["build", "test", "lint", "typecheck"]) {
    try {
      const runner = await chooseCheck(root, task, packageJson);
      checks.push({
        task,
        kind: String(runner.kind || "unknown").slice(0, 40),
        label: String(runner.label || (runner.script ? "package.json " + runner.script : runner.command || task)).slice(0, 120),
        command: String(runner.command || "").slice(0, 500)
      });
    } catch (error) {
      unavailable.push({
        task,
        reason: (error instanceof Error ? error.message : String(error)).slice(0, 400)
      });
    }
  }
  return { root, checks, unavailable };
}

export async function runProjectCheck(inputRoot, task, options = {}) {
  const root = await resolveWorkspace(inputRoot);
  const packageJson = await readJsonIfPresent(root, "package.json");
  const selected = await chooseCheck(root, task, packageJson);
  const timeoutMs = Math.max(1000, Math.min(180000, Number(options.timeoutMs) || 120000));
  if (options.signal?.aborted) {
    const error = new Error("Project check was cancelled before it started.");
    error.name = "AbortError";
    throw error;
  }
  const outputParts = [];
  let outputSize = 0;
  let timedOut = false;
  const startedAt = Date.now();

  return await new Promise((resolve, reject) => {
    let child;
    try {
      const spawnOptions = {
        cwd: root,
        windowsHide: true,
        detached: process.platform !== "win32"
      };
      if (selected.kind === "npm-script" && process.platform === "win32") {
        child = spawn("cmd.exe", ["/d", "/s", "/c", selected.command], spawnOptions);
      } else if (selected.kind === "npm-script") {
        child = spawn("npm", ["run", selected.script], spawnOptions);
      } else if (selected.kind === "python-pytest" || selected.kind === "python-unittest") {
        child = spawn(selected.executable, selected.args, spawnOptions);
      } else if (selected.kind === "node-test" || selected.kind === "command") {
        child = spawn(selected.executable, selected.args, spawnOptions);
      } else {
        reject(new Error("Unsupported verification runner: " + String(selected.kind)));
        return;
      }
    } catch (error) {
      reject(error);
      return;
    }
    const collect = chunk => {
      const text = chunk.toString("utf8");
      if (outputSize < MAX_OUTPUT) {
        const clipped = text.slice(0, MAX_OUTPUT - outputSize);
        outputParts.push(clipped);
        outputSize += clipped.length;
      }
    };
    child.stdout.on("data", collect);
    child.stderr.on("data", collect);
    let forceKillTimer;
    let aborted = false;
    const signal = options.signal;
    const cleanup = () => {
      clearTimeout(timer);
      clearTimeout(forceKillTimer);
      signal?.removeEventListener("abort", abortProcess);
    };
    const abortProcess = () => {
      if (aborted) return;
      aborted = true;
      terminateProcessTree(child, false);
      forceKillTimer = setTimeout(() => terminateProcessTree(child, true), 1500);
    };
    const timer = setTimeout(() => {
      timedOut = true;
      terminateProcessTree(child, false);
      forceKillTimer = setTimeout(() => terminateProcessTree(child, true), 1500);
    }, timeoutMs);
    if (signal?.aborted) abortProcess();
    else signal?.addEventListener("abort", abortProcess, { once: true });
    child.on("error", error => {
      cleanup();
      reject(error);
    });
    child.on("close", code => {
      cleanup();
      if (aborted) {
        const error = new Error("Project check was cancelled.");
        error.name = "AbortError";
        reject(error);
        return;
      }
      const result = {
        root, task, command: selected.command, exitCode: code,
        passed: code === 0 && !timedOut, timedOut,
        durationMs: Date.now() - startedAt, output: outputParts.join("")
      };
      appendAudit(root, {
        action: "run_project_check", task,
        command: selected.command, exitCode: code, timedOut
      }).catch(() => {});
      resolve(result);
    });
  });
}

export const workspaceLimits = {
  maxFileBytes: MAX_TEXT_BYTES,
  maxFiles: MAX_FILES,
  maxDepth: MAX_DEPTH,
  ignoredDirectories: [...IGNORED_DIRS]
};

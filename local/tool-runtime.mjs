import crypto from "node:crypto";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import {
  applyFileProposal,
  applyFileProposals,
  createProjectPlan,
  discardFileProposals,
  discoverProjectChecks,
  inspectWorkspace,
  listPendingProposals,
  proposeFileChange,
  readWorkspaceFile,
  recordWorkspaceToolEvent,
  resolveWorkspace,
  runProjectCheck,
  searchWorkspace
} from "./workspace.mjs";

const execFileAsync = promisify(execFile);
const MAX_READ_LINES = 1000;
const MAX_READ_CHARS = 50_000;

const TOOL_DEFINITIONS = [
  {
    name: "workspace.inspect",
    description: "Inspect project metadata, supported languages, manifests, scripts, frameworks, and bounded file inventory.",
    permission: "read_only",
    approvalRequired: false,
    inputSchema: { type: "object", properties: {}, additionalProperties: false }
  },
  {
    name: "workspace.checks.discover",
    description: "Discover build, test, lint, and type-check runners that are actually configured for this workspace without executing them.",
    permission: "read_only",
    approvalRequired: false,
    inputSchema: { type: "object", properties: {}, additionalProperties: false }
  },
  {
    name: "workspace.search",
    description: "Search readable project files and return ranked paths with bounded matching line excerpts.",
    permission: "read_only",
    approvalRequired: false,
    inputSchema: {
      type: "object", required: ["query"],
      properties: { query: { type: "string", minLength: 2, maxLength: 300 }, limit: { type: "integer", minimum: 1, maximum: 50 } },
      additionalProperties: false
    }
  },
  {
    name: "workspace.read",
    description: "Read a bounded line range from one permitted text file. Secret paths and symbolic links remain blocked.",
    permission: "read_only",
    approvalRequired: false,
    inputSchema: {
      type: "object", required: ["path"],
      properties: {
        path: { type: "string", minLength: 1, maxLength: 1024 },
        startLine: { type: "integer", minimum: 1, maximum: 10_000_000 },
        endLine: { type: "integer", minimum: 1, maximum: 10_000_000 }
      },
      additionalProperties: false
    }
  },
  {
    name: "workspace.plan",
    description: "Create a local, deterministic project-workflow plan under .syntra/plans. This is a planning template, not an AI-generated implementation plan.",
    permission: "local_state_write",
    approvalRequired: false,
    inputSchema: {
      type: "object", required: ["goal"],
      properties: { goal: { type: "string", minLength: 8, maxLength: 4000 } },
      additionalProperties: false
    }
  },
  {
    name: "workspace.proposals.list",
    description: "List pending and stale hash-checked proposals for the selected workspace.",
    permission: "read_only",
    approvalRequired: false,
    inputSchema: { type: "object", properties: {}, additionalProperties: false }
  },
  {
    name: "workspace.proposals.stage",
    description: "Stage a proposed text change in .syntra without modifying the original project file.",
    permission: "local_state_write",
    approvalRequired: false,
    inputSchema: {
      type: "object", required: ["path", "content"],
      properties: {
        path: { type: "string", minLength: 1, maxLength: 1024 },
        content: { type: "string", maxLength: 524288 },
        expectedBaseHash: { type: "string", pattern: "^[a-f0-9]{64}$" }
      },
      additionalProperties: false
    }
  },
  {
    name: "workspace.proposals.apply",
    description: "Apply one saved proposal after explicit user approval. Original source hashes, backups, and transactional safeguards are rechecked.",
    permission: "source_write",
    approvalRequired: true,
    inputSchema: {
      type: "object", required: ["proposalId"],
      properties: { proposalId: { type: "string", pattern: "^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$" } },
      additionalProperties: false
    }
  },
  {
    name: "workspace.proposals.apply-many",
    description: "Apply multiple saved proposals as one transaction after explicit user approval. Every source hash is preflighted and backups are created before writes.",
    permission: "source_write",
    approvalRequired: true,
    inputSchema: {
      type: "object", required: ["proposalIds"],
      properties: {
        proposalIds: {
          type: "array", minItems: 1, maxItems: 24,
          items: { type: "string", pattern: "^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$" }
        }
      },
      additionalProperties: false
    }
  },
  {
    name: "workspace.proposals.discard",
    description: "Delete saved proposal records without modifying project files. Requires explicit approval because discarding a review record is irreversible.",
    permission: "destructive_state_write",
    approvalRequired: true,
    inputSchema: {
      type: "object", required: ["proposalIds"],
      properties: {
        proposalIds: {
          type: "array", minItems: 1, maxItems: 24,
          items: { type: "string", pattern: "^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$" }
        }
      },
      additionalProperties: false
    }
  },
  {
    name: "workspace.check",
    description: "Run one discovered project check with a hard timeout. This may execute repository-defined code and always requires explicit approval.",
    permission: "execute",
    approvalRequired: true,
    inputSchema: {
      type: "object", required: ["task"],
      properties: {
        task: { type: "string", enum: ["build", "test", "lint", "typecheck"] },
        timeoutMs: { type: "integer", minimum: 1000, maximum: 180000 }
      },
      additionalProperties: false
    }
  },
  {
    name: "git.status",
    description: "Read Git branch and working-tree status without changing the repository or contacting a remote.",
    permission: "read_only",
    approvalRequired: false,
    inputSchema: { type: "object", properties: {}, additionalProperties: false }
  },
  {
    name: "git.diff-summary",
    description: "Read tracked working-tree and index diff statistics only; file contents are not returned.",
    permission: "read_only",
    approvalRequired: false,
    inputSchema: { type: "object", properties: {}, additionalProperties: false }
  }
];

const definitionsByName = new Map(TOOL_DEFINITIONS.map(definition => [definition.name, definition]));

export function listWorkspaceTools() {
  return TOOL_DEFINITIONS.map(definition => ({
    ...definition,
    inputSchema: JSON.parse(JSON.stringify(definition.inputSchema))
  }));
}

function isPlainObject(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

function validateInput(definition, input) {
  if (!isPlainObject(input)) throw new Error("Tool input must be a JSON object.");
  const schema = definition.inputSchema;
  const allowed = new Set(Object.keys(schema.properties || {}));
  const extra = Object.keys(input).filter(key => !allowed.has(key));
  if (extra.length) throw new Error("Unknown input field(s) for " + definition.name + ": " + extra.join(", "));
  for (const required of schema.required || []) {
    if (!(required in input)) throw new Error("Missing required input field: " + required);
  }

  for (const [key, value] of Object.entries(input)) {
    const rule = schema.properties[key];
    if (rule.type === "string") {
      if (typeof value !== "string" || value.length < (rule.minLength || 0) || value.length > (rule.maxLength ?? Infinity)) {
        throw new Error("Tool input field " + key + " must be a string within its allowed length.");
      }
      if (rule.pattern && !(new RegExp(rule.pattern).test(value))) {
        throw new Error("Tool input field " + key + " has an invalid format.");
      }
      if (rule.enum && !rule.enum.includes(value)) {
        throw new Error("Tool input field " + key + " has an unsupported value.");
      }
    } else if (rule.type === "integer") {
      if (!Number.isInteger(value)
        || (rule.minimum !== undefined && value < rule.minimum)
        || (rule.maximum !== undefined && value > rule.maximum)) {
        throw new Error("Tool input field " + key + " is outside its allowed integer range.");
      }
    } else if (rule.type === "array") {
      if (!Array.isArray(value) || value.length < rule.minItems || value.length > rule.maxItems
        || value.some(item => typeof item !== "string" || !(new RegExp(rule.items.pattern).test(item)))) {
        throw new Error("Tool input field " + key + " must contain 1-24 valid proposal IDs.");
      }
      if (new Set(value).size !== value.length) throw new Error("Tool input field " + key + " must not contain duplicate IDs.");
    }
  }
}

function summarizeInput(name, input) {
  switch (name) {
    case "workspace.search": return { queryLength: input.query.length, limit: input.limit || 20 };
    case "workspace.read": return { path: input.path, startLine: input.startLine || 1, endLine: input.endLine || null };
    case "workspace.plan": return { goalLength: input.goal.length };
    case "workspace.proposals.stage": return {
      path: input.path, contentBytes: Buffer.byteLength(input.content, "utf8"),
      expectedBaseHashProvided: typeof input.expectedBaseHash === "string"
    };
    case "workspace.proposals.apply": return { proposalId: input.proposalId };
    case "workspace.proposals.discard": return { proposalCount: input.proposalIds.length };
    case "workspace.check": return { task: input.task, timeoutMs: input.timeoutMs || 120000 };
    default: return {};
  }
}

async function runGit(root, args) {
  const result = await execFileAsync("git", ["-c", "core.fsmonitor=false", "--no-optional-locks", ...args], {
    cwd: root, windowsHide: true, timeout: 8000, maxBuffer: 32 * 1024, encoding: "utf8",
    env: {
      ...process.env,
      GIT_OPTIONAL_LOCKS: "0",
      GIT_TERMINAL_PROMPT: "0",
      GIT_PAGER: "cat"
    }
  });
  return String(result.stdout || "").slice(0, 28000).trimEnd();
}

async function invokeTool(root, name, input, signal) {
  switch (name) {
    case "workspace.inspect":
      return inspectWorkspace(root);
    case "workspace.checks.discover":
      return discoverProjectChecks(root);
    case "workspace.search":
      return searchWorkspace(root, input.query, input.limit || 20);
    case "workspace.read": {
      const result = await readWorkspaceFile(root, input.path);
      const lines = result.content.split(/\r?\n/);
      const startLine = input.startLine || 1;
      const endLine = Math.min(lines.length, input.endLine || (startLine + 199), startLine + MAX_READ_LINES - 1);
      if (startLine > lines.length) throw new Error("Start line is beyond the end of the file.");
      if (endLine < startLine) throw new Error("End line must be greater than or equal to start line.");
      let content = lines.slice(startLine - 1, endLine).join("\n");
      let truncated = false;
      if (content.length > MAX_READ_CHARS) {
        content = content.slice(0, MAX_READ_CHARS);
        truncated = true;
      }
      return {
        root: result.root, path: result.path, sha256: result.sha256,
        bytes: result.bytes, totalLines: lines.length, startLine, endLine,
        truncated: truncated || endLine < lines.length, content
      };
    }
    case "workspace.plan":
      return createProjectPlan(root, input.goal);
    case "workspace.proposals.list":
      return listPendingProposals(root);
    case "workspace.proposals.stage":
      return proposeFileChange(root, input.path, input.content, input.expectedBaseHash);
    case "workspace.proposals.apply":
      return applyFileProposal(root, input.proposalId, true);
    case "workspace.proposals.apply-many":
      return applyFileProposals(root, input.proposalIds, true);
    case "workspace.proposals.discard":
      return discardFileProposals(root, input.proposalIds, true);
    case "workspace.check":
      return runProjectCheck(root, input.task, { timeoutMs: input.timeoutMs || 120000, signal });
    case "git.status":
      return { root, status: await runGit(root, ["status", "--short", "--branch"]) };
    case "git.diff-summary": {
      const [workingTree, index] = await Promise.all([
        runGit(root, ["diff", "--stat", "--no-ext-diff"]),
        runGit(root, ["diff", "--cached", "--stat", "--no-ext-diff"])
      ]);
      return { root, workingTree, index };
    }
    default:
      throw new Error("No handler is registered for tool: " + name);
  }
}

function summarizeOutput(name, output) {
  switch (name) {
    case "workspace.inspect":
      return { name: output.name, fileCount: output.fileCount, frameworks: output.frameworks };
    case "workspace.checks.discover":
      return {
        supportedTasks: output.checks.map(check => check.task),
        runnerKinds: output.checks.map(check => check.kind),
        unavailableTasks: output.unavailable.map(item => item.task)
      };
    case "workspace.search":
      return { searchedFiles: output.searchedFiles, resultCount: output.results.length, topPaths: output.results.slice(0, 8).map(item => item.path) };
    case "workspace.read":
      return { path: output.path, bytes: output.bytes, totalLines: output.totalLines, startLine: output.startLine, endLine: output.endLine };
    case "workspace.plan":
      return { planPath: output.planPath, taskCount: output.tasks?.length || 0 };
    case "workspace.proposals.list":
      return { pendingCount: output.pending.length, staleCount: output.stale.length };
    case "workspace.proposals.stage":
      return { proposalId: output.id, path: output.path, afterHash: output.afterHash };
    case "workspace.proposals.apply":
      return { proposalId: output.id, path: output.path, applied: output.applied };
    case "workspace.proposals.apply-many":
      return { appliedCount: output.applied?.length || 0, paths: (output.applied || []).map(item => item.path) };
    case "workspace.proposals.discard":
      return { discardedCount: output.discarded.length };
    case "workspace.check":
      return { task: output.task, passed: output.passed, exitCode: output.exitCode, timedOut: output.timedOut };
    case "git.status":
      return { statusCharacters: output.status.length };
    case "git.diff-summary":
      return { workingTreeCharacters: output.workingTree.length, indexCharacters: output.index.length };
    default:
      return {};
  }
}

async function executeWorkspaceToolUnlocked({ root: inputRoot, auditRoot: inputAuditRoot, name, input = {}, approval = false, signal }) {
  const definition = definitionsByName.get(name);
  if (!definition) throw new Error("Unknown SYNTRA tool: " + String(name));
  validateInput(definition, input);

  const root = await resolveWorkspace(inputRoot);
  const auditRoot = await resolveWorkspace(inputAuditRoot || inputRoot);
  const runId = crypto.randomUUID();
  const startedAt = Date.now();
  const approved = approval === true;
  const inputSummary = summarizeInput(name, input);
  const base = {
    runId, tool: name, permission: definition.permission,
    approved, inputSummary
  };

  if (definition.approvalRequired && !approved) {
    await recordWorkspaceToolEvent(auditRoot, {
      ...base, status: "approval_required", durationMs: Date.now() - startedAt,
      outputSummary: { message: "Explicit approval is required; the tool was not executed." }
    });
    return {
      ok: false, runId, name, status: "approval_required",
      permission: definition.permission, approvalRequired: true,
      durationMs: Date.now() - startedAt,
      message: "This tool can execute project code, modify project files, or discard saved review records. Explicit user approval is required."
    };
  }

  try {
    await recordWorkspaceToolEvent(auditRoot, {
      ...base, status: "started", durationMs: 0,
      outputSummary: { message: "Tool execution started." }
    });
  } catch (error) {
    return {
      ok: false, runId, name, status: "audit_failed",
      permission: definition.permission, approvalRequired: definition.approvalRequired,
      durationMs: Date.now() - startedAt,
      error: "SYNTRA could not persist the tool audit event, so the operation was not executed: " +
        (error instanceof Error ? error.message : String(error))
    };
  }

  let output;
  let failure = null;
  try {
    if (signal?.aborted) {
      const cancelled = new Error("Tool execution was cancelled.");
      cancelled.name = "AbortError";
      throw cancelled;
    }
    output = await invokeTool(root, name, input, signal);
    if (signal?.aborted) {
      const cancelled = new Error("Tool execution was cancelled.");
      cancelled.name = "AbortError";
      throw cancelled;
    }
  } catch (error) {
    failure = error instanceof Error ? error.message : String(error);
  }

  const durationMs = Date.now() - startedAt;
  const cancelled = Boolean(signal?.aborted);
  const status = cancelled ? "cancelled" : failure ? "failed" : "completed";
  let auditRecorded = true;
  try {
    await recordWorkspaceToolEvent(auditRoot, {
      ...base, status, durationMs,
      outputSummary: failure ? {} : summarizeOutput(name, output),
      error: failure || undefined
    });
  } catch {
    auditRecorded = false;
  }

  if (failure) {
    return {
      ok: false, runId, name, status, permission: definition.permission,
      approvalRequired: definition.approvalRequired, durationMs,
      error: failure, auditRecorded
    };
  }
  return {
    ok: true, runId, name, status, permission: definition.permission,
    approvalRequired: definition.approvalRequired, durationMs,
    output, auditRecorded
  };
}


const workspaceExecutionQueues = new Map();

async function withWorkspaceQueue(key, action) {
  const previous = workspaceExecutionQueues.get(key) || Promise.resolve();
  let release;
  const gate = new Promise(resolve => { release = resolve; });
  const queued = previous.then(() => gate);
  workspaceExecutionQueues.set(key, queued);
  await previous;
  try {
    return await action();
  } finally {
    release();
    if (workspaceExecutionQueues.get(key) === queued) workspaceExecutionQueues.delete(key);
  }
}

/**
 * Serialize tools against one workspace so check/apply/stage/discard operations
 * cannot race each other or interleave their audit records.
 */
export async function executeWorkspaceTool(options = {}) {
  if (!isPlainObject(options)) throw new Error("Tool execution options must be a JSON object.");
  const inputRoot = await resolveWorkspace(options.root);
  const auditRoot = await resolveWorkspace(options.auditRoot || options.root);
  return withWorkspaceQueue(auditRoot, () => executeWorkspaceToolUnlocked({
    ...options,
    root: inputRoot,
    auditRoot
  }));
}

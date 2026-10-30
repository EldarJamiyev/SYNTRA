import http from "node:http";
import path from "node:path";
import { fileURLToPath, URL } from "node:url";
import crypto from "node:crypto";
import { runAgentTask } from "./agent.mjs";
import { executeWorkspaceTool, listWorkspaceTools } from "./tool-runtime.mjs";
import { isCodingTask } from "./chat-intent.mjs";
import { detectLocalModel, handleLocalModelRequest } from "./model-provider.mjs";
import { classifyNetworkDiagnostic, formatNetworkDiagnostic, runNetworkDiagnostic } from "./network-diagnostics.mjs";
import {
  createAgentRun,
  getAgentRun,
  listAgentRuns,
  listPendingProposals,
  readWorkspaceFile,
  recoverInterruptedAgentRuns,
  requestAgentRunCancellation,
  resolveWorkspace,
  updateAgentRun
} from "./workspace.mjs";

const HOST = "127.0.0.1";
const PORT = Number(process.env.SYNTRA_LOCAL_PORT || 3001);
const AI_BASE = "http://127.0.0.1:" + Number(process.env.SYNTRA_AI_PORT || 3002);
const ALLOWED_ORIGINS = new Set([
  "http://127.0.0.1:3000",
  "http://localhost:3000",
  "http://127.0.0.1:5173",
  "http://localhost:5173"
]);
const MAX_BODY_BYTES = 600 * 1024;
const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
let activeWorkspace = process.env.SYNTRA_WORKSPACE ? path.resolve(process.env.SYNTRA_WORKSPACE) : REPO_ROOT;
const chatJobs = new Map();
let activeChatJobId = null;


function sendJson(response, status, data) {
  response.writeHead(status, {
    "Content-Type": "application/json; charset=utf-8",
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer"
  });
  response.end(JSON.stringify(data));
}

async function parseBody(request) {
  const chunks = [];
  let total = 0;
  for await (const chunk of request) {
    total += chunk.length;
    if (total > MAX_BODY_BYTES) throw new Error("Request body exceeds the local API size limit.");
    chunks.push(chunk);
  }
  if (!total) return {};
  let value;
  try { value = JSON.parse(Buffer.concat(chunks).toString("utf8")); }
  catch { throw new Error("Request body must be valid JSON."); }
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("Request body must be a JSON object.");
  }
  return value;
}

async function aiRequest(method, route, body, timeoutMs = undefined, options = {}) {
  const onToken = typeof options?.onToken === "function" ? options.onToken : undefined;
  const signal = options?.signal;
  const modelRoute = (method === "GET" && route === "/model/status")
    || (method === "POST" && (route === "/model/tokenize" || route === "/model/generate"));
  if (modelRoute) {
    const provider = await detectLocalModel();
    if (provider) {
      const localResult = await handleLocalModelRequest(provider, method, route, body || {}, { timeoutMs, onToken, signal });
      if (localResult) return localResult;
    }
  }

  let response;
  try {
    response = await fetch(AI_BASE + route, {
      method,
      headers: method === "GET" ? {} : { "Content-Type": "application/json" },
      body: method === "GET" ? undefined : JSON.stringify(body || {}),
      signal: signal ? AbortSignal.any([signal, AbortSignal.timeout(timeoutMs || (method === "POST" && route === "/model/generate" ? 300000 : 30000))]) : AbortSignal.timeout(timeoutMs || (method === "POST" && route === "/model/generate" ? 300000 : 30000))
    });
  } catch (error) {
    if (signal?.aborted || error?.name === "AbortError") throw error;
    throw new Error("SYNTRA's native AI service is offline. Start it with 'npm run ai:server', or install and run a local Ollama model for stronger local inference.");
  }
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || "SYNTRA AI request failed with HTTP " + response.status);
  return payload;
}

async function generateAgentCompletion(body) {
  const root = body.root;
  const filePath = body.path;
  const goal = typeof body.goal === "string" ? body.goal.trim() : "";
  if (!goal || goal.length > 5000) throw new Error("Enter a specific task (maximum 5,000 characters).");
  const file = await readWorkspaceFile(root, filePath);
  if (file.content.length > 200000) {
    throw new Error("AI editing is limited to files of at most 200 KB; select a smaller module.");
  }
  // Bound prompt fields in UTF-8 bytes so the actual file tail fits the native model's
  // small context. Keep the source tail last: autoregressive generation continues from
  // the final prompt bytes, so instruction prose must not follow the code excerpt.
  const taskExcerpt = Buffer.from(goal, "utf8").subarray(0, 160).toString("utf8").replace(/\uFFFD$/u, "");
  let pathBytes = Buffer.from(filePath, "utf8").subarray(-100);
  while (pathBytes.length && (pathBytes[0] & 0xc0) === 0x80) pathBytes = pathBytes.subarray(1);
  const targetExcerpt = pathBytes.toString("utf8").replace(/^\uFFFD/u, "");
  let sourceBytes = Buffer.from(file.content, "utf8").subarray(-520);
  while (sourceBytes.length && (sourceBytes[0] & 0xc0) === 0x80) sourceBytes = sourceBytes.subarray(1);
  const excerpt = sourceBytes.toString("utf8").replace(/^\uFFFD/u, "");
  const prompt = [
    "SYNTRA_CODE_CONTINUATION",
    "Task: " + taskExcerpt,
    "Target: " + targetExcerpt,
    "Continue with source-code bytes only; do not add markdown or explanations.",
    "Existing source-file tail:",
    excerpt
  ].join("\n");
  const output = await aiRequest("POST", "/model/generate", {
    prompt,
    max_new_tokens: Math.min(1024, Math.max(32, Number(body.max_new_tokens) || 256)),
    temperature: 0.25,
    top_k: 30
  });
  return {
    ...output,
    sourceContent: file.content,
    targetPath: filePath,
    task: goal,
    sourceHash: file.sha256,
    sourceExcerptBytes: Buffer.byteLength(excerpt),
    promptBytes: Buffer.byteLength(prompt),
    generationMode: "trained-local-model-continuation",
    notice: "This is actual output from your checkpoint, not a hosted model. A raw continuation is not a guaranteed patch or a complete-file rewrite; review it, edit it, and run project checks before applying."
  };
}



function pruneChatJobs() {
  if (chatJobs.size <= 32) return;
  for (const [id, job] of chatJobs) {
    if (id !== activeChatJobId && job.status !== "running") chatJobs.delete(id);
    if (chatJobs.size <= 24) break;
  }
}

function finishChat(jobId, reply, details = {}) {
  const job = chatJobs.get(jobId);
  if (!job) return;
  Object.assign(job, { status: "completed", phase: "completed", reply, error: null, updatedAt: new Date().toISOString() }, details);
  if (activeChatJobId === jobId) activeChatJobId = null;
}

function failChat(jobId, error) {
  const job = chatJobs.get(jobId);
  if (!job) return;
  const message = error instanceof Error ? error.message : String(error);
  const partialReply = typeof job.reply === "string" ? job.reply : "";
  Object.assign(job, {
    status: "failed", phase: "failed", reply: message, error: message,
    partialReply, updatedAt: new Date().toISOString()
  });
  if (activeChatJobId === jobId) activeChatJobId = null;
}

function updateChat(jobId, value) {
  const job = chatJobs.get(jobId);
  if (job) Object.assign(job, value, { updatedAt: new Date().toISOString() });
}

function cancelChatJob(jobId, reply = "Task cancelled. Any proposals already staged remain in the workspace; type /proposals to inspect them.") {
  const job = chatJobs.get(jobId);
  if (!job) return;
  Object.assign(job, {
    status: "cancelled",
    phase: "cancelled",
    message: "Task cancelled.",
    reply,
    error: null,
    updatedAt: new Date().toISOString()
  });
  if (activeChatJobId === jobId) activeChatJobId = null;
}

function publicAgentRun(run) {
  if (!run) return null;
  const { recoveryPatch, ...visible } = run;
  return { ...visible, recoveryAvailable: Boolean(recoveryPatch) };
}

async function listCurrentAgentRuns() {
  const runs = await listAgentRuns(activeWorkspace, 12);
  return { root: activeWorkspace, runs: runs.map(publicAgentRun) };
}

async function readCurrentAgentRun(body) {
  const run = await getAgentRun(activeWorkspace, body.runId);
  if (!run) throw new Error("Agent run was not found in the active workspace.");
  return { run: publicAgentRun(run), recoveryPatch: run.recoveryPatch || null };
}

async function resumeAgentRun(body) {
  const run = await getAgentRun(activeWorkspace, body.runId);
  if (!run) throw new Error("Agent run was not found in the active workspace.");
  if (run.status !== "interrupted") {
    throw new Error("Only interrupted agent runs can be resumed. Current status: " + run.status + ".");
  }
  const saved = await listPendingProposals(activeWorkspace);
  if (saved.pending.length || saved.stale.length) {
    throw new Error("Resolve existing workspace proposals with /proposals and /apply or /discard before resuming this run.");
  }
  return startChat({ message: "/resume " + run.id, history: [] });
}

async function cancelChat(body) {
  const job = chatJobs.get(typeof body.jobId === "string" ? body.jobId : "");
  if (!job) throw new Error("Chat turn not found or expired.");
  if (job.status !== "running") return { ok: false, status: job.status, message: "This task is no longer running." };
  if (job.phase === "applying") {
    return { ok: false, status: job.status, message: "A transactional file apply is in progress and cannot be interrupted safely." };
  }
  // Change the in-memory phase and abort synchronously before awaiting durable
  // state writes. This closes the race with the agent entering transactional apply.
  updateChat(job.jobId, { phase: "cancelling", message: "Stopping the current task safely." });
  job.controller?.abort();
  if (job.agentRunId) {
    await requestAgentRunCancellation(job.workspace, job.agentRunId);
  }
  return { ok: true, status: "cancelling", agentRunId: job.agentRunId || null };
}



async function applyProposals(root, proposals, signal) {
  const toolRun = await executeWorkspaceTool({
    root, auditRoot: root, name: "workspace.proposals.apply-many",
    input: { proposalIds: proposals.map(proposal => proposal.id) },
    approval: true, signal
  });
  if (!toolRun.ok) throw new Error(toolRun.error || toolRun.message || "The proposal apply tool failed.");
  return {
    applied: toolRun.output.applied,
    toolCall: {
      runId: toolRun.runId, name: toolRun.name, status: toolRun.status,
      permission: toolRun.permission, durationMs: toolRun.durationMs,
      auditRecorded: toolRun.auditRecorded
    }
  };
}

async function readWorkspaceStatus(root = activeWorkspace) {
  const proposals = await listPendingProposals(root);
  return { root: proposals.root, pendingProposals: proposals.pending, staleProposals: proposals.stale };
}

async function switchWorkspace(body) {
  const currentJob = activeChatJobId ? chatJobs.get(activeChatJobId) : null;
  if (currentJob?.status === "running") {
    throw new Error("Finish the current chat task before switching workspaces.");
  }
  activeWorkspace = await resolveWorkspace(body.root);
  await recoverInterruptedAgentRuns(activeWorkspace);
  return readWorkspaceStatus(activeWorkspace);
}

function formatProposalList(status) {
  const lines = ["**Saved workspace proposals**", "", "Workspace: " + status.root];
  if (!status.pendingProposals.length && !status.staleProposals.length) {
    lines.push("", "No pending or stale proposals were found.");
    return lines.join("\n");
  }
  for (const proposal of status.pendingProposals) {
    const summary = proposal.summary || {};
    lines.push(
      "",
      "**Pending · `" + String(proposal.path || "unknown path").replaceAll("`", "") + "`**",
      "Added: " + Number(summary.addedLines || 0) + " · Removed: " + Number(summary.removedLines || 0)
    );
    if (summary.preview) {
      const preview = String(summary.preview).slice(0, 2000);
      const longestBacktickRun = Math.max(0, ...((preview.match(/`+/g) || []).map(run => run.length)));
      const fence = "`".repeat(Math.max(3, longestBacktickRun + 1));
      lines.push(fence + "text", preview, fence);
    }
  }
  for (const proposal of status.staleProposals) {
    lines.push(
      "",
      "**Stale · `" + String(proposal.path || "unknown path").replaceAll("`", "") + "`**",
      String(proposal.reason || "The original source no longer matches this proposal.")
    );
  }
  lines.push(
    "",
    "The full diff remains in the conversation where the proposal was created.",
    "Type `/apply` to apply pending proposals after reviewing them, or `/discard` to remove pending and stale proposal records without changing project files."
  );
  return lines.join("\n");
}

async function executeToolOutput(options) {
  const result = await executeWorkspaceTool(options);
  if (!result.ok) throw new Error(result.error || result.message || "SYNTRA tool execution failed.");
  return result.auditRecorded
    ? result.output
    : { ...result.output, auditWarning: "The tool completed, but its final audit record could not be persisted. Check .syntra/actions.jsonl." };
}

function compactAgentReply(result, applied) {
  const lines = [result.patchSummary || "Code task finished.", ""];
  if (applied.length) {
    lines.push("Changed files:");
    for (const item of applied) lines.push("- " + item.path);
  } else if (result.proposals?.length && result.verification?.status === "verification_blocked") {
    lines.push("I did not run candidate checks because this patch changes dependency/build manifests or lockfiles. The sandbox does not reinstall changed dependencies.");
    for (const item of result.proposals) lines.push("- " + item.path);
    lines.push("The original project files remain unchanged. Review the script changes, then type /apply only if you accept them.");
  } else if (result.proposals?.length && result.verification?.status === "passed_review_required") {
    lines.push("The checks passed in the temporary copy, but I did not auto-apply this patch because it changes tests or verification configuration.");
    for (const item of result.proposals) lines.push("- " + item.path);
    lines.push("The original project files remain unchanged. Review those diffs, then type /apply to apply the proposals.");
  } else if (result.proposals?.length) {
    lines.push("I prepared proposed changes, but did not apply them because verification did not pass or no checks were available.");
    for (const item of result.proposals) lines.push("- " + item.path);
    lines.push("The original project files remain unchanged. Type /apply to explicitly apply these unverified proposals.");
  } else {
    lines.push("No valid patch was produced. The original project files remain unchanged.");
  }
  lines.push("", "Verification: " + String(result.verification?.status || "not run").replaceAll("_", " "));
  for (const check of result.verification?.checks || []) {
    lines.push("- " + check.task + ": " + (check.passed ? "passed" : check.skipped ? "skipped" : check.timedOut ? "timed out" : "failed"));
    if (!check.passed && check.output) lines.push(check.output.slice(-1600));
  }
  const reviewFiles = result.changedFiles || [];
  if (reviewFiles.length) {
    lines.push("", "### Change review");
    for (const file of reviewFiles) {
      const safePath = String(file.path || "changed file").replaceAll("`", "");
      lines.push("", "**`" + safePath + "`**");
      if (file.reason) lines.push(String(file.reason).replace(/[\r\n]+/g, " ").slice(0, 500));
      const longestBacktickRun = Math.max(0, ...((String(file.diff || "").match(/`+/g) || []).map(run => run.length)));
      const fence = "`".repeat(Math.max(3, longestBacktickRun + 1));
      lines.push(fence + "diff", file.diff || "Diff preview is unavailable.", fence);
      if (file.diffTruncated) lines.push("*The changed section is large; this is a bounded preview, not the full file.*");
    }
  }
  if (result.proposals?.length && !applied.length) {
    lines.push("", "Review the proposed changes above before typing `/apply`. SYNTRA will recheck the original file hashes before writing.");
  }
  if (result.toolCalls?.length) {
    const toolNames = [...new Set(result.toolCalls.map(call => call.name))];
    lines.push("", "Tools used: " + toolNames.map(name => "`" + name + "`").join(", ") + ".");
    if (result.toolCalls.every(call => call.auditRecorded !== false)) {
      lines.push("Tool execution events are recorded in `.syntra/actions.jsonl`.");
    }
  }
  if (result.attempts?.length) lines.push("", "Repair attempts: " + result.attempts.length + ".");
  if (applied.length) lines.push("", "Backups are saved under .syntra.");
  return lines.join("\n");
}

async function runChatTurn(jobId, message, history, signal) {
  let agentRunId = null;
  try {
    updateChat(jobId, { phase: "connecting", message: "Connecting to the local model" });
    const workspaceCommand = message.match(/^\/workspace(?:\s+(.+))?$/is);
    if (workspaceCommand) {
      const requested = (workspaceCommand[1] || "").trim();
      if (!requested) {
        finishChat(jobId, "Current workspace:\n" + activeWorkspace + "\n\nSwitch project by typing /workspace followed by an absolute folder path.");
        return;
      }
      activeWorkspace = await resolveWorkspace(requested);
      finishChat(jobId, "Workspace changed to:\n" + activeWorkspace);
      return;
    }
    if (/^\/proposals\s*$/i.test(message.trim())
      || /^\/apply\s*$/i.test(message.trim())
      || /^\/discard\s*$/i.test(message.trim())) {
      const listRun = await executeWorkspaceTool({
        root: activeWorkspace, auditRoot: activeWorkspace,
        name: "workspace.proposals.list", input: {}
      });
      if (!listRun.ok) throw new Error(listRun.error || "The proposal list tool failed.");
      const saved = listRun.output;
      const listTrace = {
        runId: listRun.runId, name: listRun.name, status: listRun.status,
        permission: listRun.permission, durationMs: listRun.durationMs,
        auditRecorded: listRun.auditRecorded
      };

      if (/^\/proposals\s*$/i.test(message.trim())) {
        finishChat(jobId, formatProposalList({
          root: saved.root,
          pendingProposals: saved.pending,
          staleProposals: saved.stale
        }), { toolCalls: [listTrace] });
        return;
      }

      if (/^\/apply\s*$/i.test(message.trim())) {
        if (!saved.pending.length) {
          const staleNote = saved.stale.length
            ? " " + saved.stale.length + " stale proposal(s) were found; type /proposals to inspect them and /discard to remove their records."
            : "";
          finishChat(jobId, "There are no applicable pending changes for this workspace." + staleNote, { toolCalls: [listTrace] });
          return;
        }
        updateChat(jobId, { phase: "applying", message: "Rechecking saved proposal hashes and applying changes" });
        const transaction = await applyProposals(activeWorkspace, saved.pending);
        const auditNote = transaction.toolCall.auditRecorded
          ? "Backups and tool audit records are saved under .syntra."
          : "Backups are saved under .syntra, but the final tool audit event could not be written.";
        finishChat(jobId, "Applied the explicitly requested proposals:\n" + transaction.applied.map(item => "- " + item.path).join("\n") + "\n\n" + auditNote, {
          toolCalls: [listTrace, transaction.toolCall]
        });
        return;
      }

      const ids = [...saved.pending, ...saved.stale].map(item => item.id);
      if (!ids.length) {
        finishChat(jobId, "There are no saved proposal records to discard.", { toolCalls: [listTrace] });
        return;
      }
      const discardRun = await executeWorkspaceTool({
        root: activeWorkspace, auditRoot: activeWorkspace, name: "workspace.proposals.discard",
        input: { proposalIds: ids }, approval: true
      });
      if (!discardRun.ok) throw new Error(discardRun.error || "The proposal discard tool failed.");
      finishChat(jobId, "Discarded " + discardRun.output.discarded.length + " saved proposal record(s). Project files were not modified.", {
        toolCalls: [listTrace, {
          runId: discardRun.runId, name: discardRun.name, status: discardRun.status,
          permission: discardRun.permission, durationMs: discardRun.durationMs,
          auditRecorded: discardRun.auditRecorded
        }]
      });
      return;
    }

    const trimmedMessage = message.trim();
    const helpRequested = /^\/help\s*$/i.test(trimmedMessage)
      || /^(?:help|what can you do|what can syntra do|how do i use syntra|what commands are available)[?.!]*$/i.test(trimmedMessage);
    if (helpRequested) {
      finishChat(jobId, [
        "**SYNTRA help**",
        "",
        "Network checks (run locally, even without a trained model):",
        "- ping example.com or /ping example.com",
        "- DNS lookup for example.com or /dns example.com",
        "- check port 443 on example.com or /port example.com 443",
        "",
        "Coding and workspace:",
        "- Ask for a specific software change to invoke the repository-aware coding agent.",
        "- /workspace followed by an absolute folder path switches the active project.",
        "- /proposals lists saved proposals after a service restart.",
        "- /apply applies pending proposals only after you review and explicitly request it.",
        "- /discard removes pending/stale proposal records without changing project files.",
        "- /status checks whether the local model backend is reachable.",
        "- /clear clears this conversation and its saved browser history.",
        "",
        "Normal chat and code edits need a compatible local Ollama model or a trained SYNTRA checkpoint. No hosted AI API is used."
      ].join("\n"));
      return;
    }

    if (/^\/status\s*$/i.test(trimmedMessage)) {
      updateChat(jobId, { phase: "status", message: "Checking the local model backend" });
      try {
        const modelStatus = await aiRequest("GET", "/model/status", undefined, 5000);
        const provider = modelStatus.provider === "ollama-local"
          ? "Ollama local model"
          : modelStatus.provider || "Native local backend";
        const lines = [
          "**SYNTRA status**",
          "",
          "Workspace service: running",
          "Model backend: reachable (" + provider + ")",
          modelStatus.checkpoint_exists
            ? "Model/checkpoint: available"
            : "Model/checkpoint: not trained or not installed"
        ];
        if (modelStatus.model_name) lines.push("Model: " + modelStatus.model_name);
        const context = Number(modelStatus.context_tokens || modelStatus.context_window_tokens || modelStatus.config?.seq_len || 0);
        if (context > 0) lines.push("Context window: " + context.toLocaleString() + " tokens");
        if (Number.isFinite(Number(modelStatus.trained_steps)) && Number(modelStatus.trained_steps) > 0) {
          lines.push("Training steps: " + Number(modelStatus.trained_steps).toLocaleString());
        }
        if (modelStatus.pretrained_weights === true) lines.push("Weights: pretrained, installed locally");
        else if (modelStatus.checkpoint_exists) lines.push("Weights: local SYNTRA checkpoint");
        lines.push("", "Network diagnostics: available without a model.");
        if (!modelStatus.checkpoint_exists) lines.push("Install a compatible local Ollama model or train the native model to enable ordinary chat and code edits.");
        finishChat(jobId, lines.join("\n"), { modelStatus: "reachable" });
      } catch (error) {
        finishChat(jobId, [
          "**SYNTRA status**",
          "",
          "Workspace service: running",
          "Model backend: unavailable",
          "Network diagnostics: available without a model.",
          "",
          "Check the service windows opened by Start-SYNTRA.bat. Ordinary chat and code edits need a reachable local model backend.",
          error instanceof Error ? error.message : String(error)
        ].join("\n"), { modelStatus: "unavailable" });
      }
      return;
    }

    const diagnostic = classifyNetworkDiagnostic(message);
    if (diagnostic) {
      if (diagnostic.kind === "invalid") {
        finishChat(jobId, formatNetworkDiagnostic(diagnostic), { diagnostic: "invalid" });
        return;
      }
      updateChat(jobId, { phase: "diagnostics", message: "Running a local network check" });
      const result = await runNetworkDiagnostic(diagnostic);
      finishChat(jobId, formatNetworkDiagnostic(result), {
        diagnostic: result.kind,
        diagnosticStatus: result.status,
        diagnosticTarget: result.host + (result.port ? ":" + result.port : "")
      });
      return;
    }

    const modelStatus = await aiRequest("GET", "/model/status", undefined, 15000, { signal });
    if (!modelStatus.checkpoint_exists) {
      finishChat(jobId,
        "The local AI backend is running, but there is no trained checkpoint yet. Train once from the repository folder, then return here:\n\n"
        + "npm run ai:install\n"
        + "npm run ai:cli -- train --root \"" + activeWorkspace + "\" --seq-len 1024 --steps 1000\n"
        + "npm run ai:server\n\n"
        + "This is a from-scratch model: training is required and does not guarantee strong instruction-following.");
      return;
    }

    const resumeMatch = message.trim().match(/^\/resume\s+([0-9a-f-]{36})$/i);
    let resumeRecord = null;
    let taskGoal = message;
    if (resumeMatch) {
      resumeRecord = await getAgentRun(activeWorkspace, resumeMatch[1]);
      if (!resumeRecord) throw new Error("Agent run was not found in the active workspace.");
      if (resumeRecord.status !== "interrupted") {
        throw new Error("Only interrupted agent runs can be resumed. Current status: " + resumeRecord.status + ".");
      }
      taskGoal = resumeRecord.goal;
    }

    if (isCodingTask(message) || resumeRecord) {
      const saved = await listPendingProposals(activeWorkspace);
      if (saved.pending.length || saved.stale.length) {
        finishChat(jobId, "This workspace already has saved proposals. Type /proposals to inspect them, then use /apply or /discard before starting or resuming a code task.");
        return;
      }

      if (resumeRecord) {
        agentRunId = resumeRecord.id;
        await updateAgentRun(activeWorkspace, agentRunId, {
          status: "running",
          phase: "resuming",
          message: "Resuming from the last durable checkpoint.",
          cancelRequested: false,
          proposalIds: [],
          event: { phase: "resuming", message: resumeRecord.recoveryPatch
            ? "Restoring and revalidating the saved patch checkpoint."
            : "Restarting task discovery because no validated patch checkpoint was saved." }
        });
      } else {
        const createdRun = await createAgentRun(activeWorkspace, taskGoal);
        agentRunId = createdRun.id;
      }

      updateChat(jobId, {
        agentRunId,
        phase: resumeRecord ? "resuming" : "planning",
        message: resumeRecord ? "Restoring the saved agent checkpoint" : "Reading the project and planning a code change"
      });

      const result = await runAgentTask({
        root: activeWorkspace,
        goal: taskGoal,
        maxIterations: 3,
        maxSlices: 3,
        runChecks: true,
        signal,
        resumePatch: resumeRecord?.recoveryPatch || null,
        resumeIteration: resumeRecord?.checkpoint?.iteration || 1,
        resumeBaseHashes: resumeRecord?.recoveryPatch
          ? Object.fromEntries((resumeRecord.changedFiles || []).map(change => [change.path, change.baseHash ?? null]))
          : null,
        previousAttempts: resumeRecord?.attempts || [],
        previousToolCalls: resumeRecord?.toolCalls || [],
        modelToolSelection: true,
        modelRequest: (method, route, body, timeoutMs, options = {}) =>
          aiRequest(method, route, body, timeoutMs, { ...options, signal }),
        onProgress: async update => {
          const phase = update.phase || "working";
          const messageText = update.message || "Working on the requested change";
          const runPatch = {
            phase,
            iteration: update.iteration || 0,
            message: messageText,
            event: { phase, message: messageText, iteration: update.iteration || 0 }
          };
          for (const key of [
            "planSteps", "selectedContext", "model", "observations", "attempts", "toolCalls",
            "verification", "changedFiles", "proposalIds", "recoveryPatch"
          ]) {
            if (Object.prototype.hasOwnProperty.call(update, key)) runPatch[key] = update[key];
          }
          if (update.checkpoint) runPatch.checkpoint = update.checkpoint;
          await updateAgentRun(activeWorkspace, agentRunId, runPatch);
          updateChat(jobId, { agentRunId, phase, message: messageText, iteration: update.iteration || 0 });
        }
      });

      let applied = [];
      if (result.verification?.status === "passed" && result.proposals?.length) {
        if (signal?.aborted) {
          const cancelled = new Error("Agent run was cancelled before file application.");
          cancelled.name = "AbortError";
          throw cancelled;
        }
        updateChat(jobId, { phase: "applying", message: "Checks passed in a temporary copy; applying hash-checked proposals" });
        const transaction = await applyProposals(activeWorkspace, result.proposals, signal);
        applied = transaction.applied;
        result.toolCalls.push(transaction.toolCall);
      }

      const completedAt = new Date().toISOString();
      await updateAgentRun(activeWorkspace, agentRunId, {
        status: "completed",
        phase: "completed",
        message: applied.length ? "Verified changes were applied." : "Agent task finished; review any staged proposals.",
        planSteps: result.plan?.steps || [],
        attempts: result.attempts || [],
        toolCalls: result.toolCalls || [],
        observations: result.observations || [],
        verification: result.verification || null,
        changedFiles: result.changedFiles || [],
        proposalIds: (result.proposals || []).map(proposal => proposal.id),
        recoveryPatch: null,
        checkpoint: {
          stage: applied.length ? "applied" : result.proposals?.length ? "proposals_ready" : "completed",
          iteration: result.attempts?.length || 0
        },
        event: {
          phase: "completed",
          message: applied.length ? "Verified proposals were applied transactionally." : "Agent task completed.",
          iteration: result.attempts?.length || 0
        }
      });

      finishChat(jobId, compactAgentReply(result, applied), {
        agentRunId,
        verification: result.verification?.status || "not_run",
        changedFiles: result.changedFiles?.map(item => item.path) || []
      });
      return;
    }

    const recentHistory = Array.isArray(history)
      ? history.filter(item => item && (item.role === "user" || item.role === "assistant") && typeof item.content === "string")
          .slice(-8).map(item => ({ role: item.role, content: item.content.slice(-2500) }))
      : [];
    let userMessage = message.slice(-10000);
    let prompt = "";
    let tokenInfo = null;
    const contextTokens = Number(modelStatus.context_tokens || modelStatus.config?.seq_len || 1024);
    const contextLimit = Math.max(64, contextTokens - 64);
    while (true) {
      prompt = [
        "SYNTRA ASSISTANT",
        "Answer the user's latest message directly and clearly. Do not claim facts or tools you do not have.",
        ...recentHistory.map(item => (item.role === "user" ? "USER: " : "ASSISTANT: ") + item.content),
        "USER: " + userMessage,
        "ASSISTANT:"
      ].join("\n\n");
      tokenInfo = await aiRequest("POST", "/model/tokenize", { text: prompt }, 30000);
      if (tokenInfo.token_count <= contextLimit) break;
      if (recentHistory.length) recentHistory.shift();
      else if (userMessage.length > 1000) userMessage = userMessage.slice(-Math.max(1000, Math.floor(userMessage.length * 0.65)));
      else break;
    }
    if (tokenInfo.token_count >= contextTokens) {
      finishChat(jobId, "Your message exceeds the trained model's context window. Please split it into a shorter message.");
      return;
    }
    updateChat(jobId, { phase: "generating", message: "Generating a local response" });
    const generation = await aiRequest("POST", "/model/generate", {
      prompt,
      max_new_tokens: Math.min(512, Math.max(16, contextTokens - tokenInfo.token_count - 1)),
      temperature: 0.7,
      top_k: Math.min(40, Number(modelStatus.tokenizer_vocab_size || modelStatus.config?.vocab_size || 1024)),
      device: "auto"
    }, 300000, {
      onToken: (_delta, fullText) => updateChat(jobId, {
        phase: "streaming",
        message: "Responding",
        reply: fullText
      })
    });
    const reply = String(generation.text || "").trim();
    finishChat(jobId, reply || "The local model returned an empty response. This may happen with an early checkpoint; more training is needed.", {
      contextTokensUsed: tokenInfo.token_count, trainedSteps: generation.trained_steps
    });
  } catch (error) {
    const cancelled = Boolean(signal?.aborted || error?.name === "AbortError");
    if (agentRunId) {
      try {
        await updateAgentRun(activeWorkspace, agentRunId, {
          status: cancelled ? "cancelled" : "failed",
          phase: cancelled ? "cancelled" : "failed",
          message: cancelled
            ? "Task cancelled at a safe checkpoint. Inspect any staged proposals before continuing."
            : error instanceof Error ? error.message : String(error),
          cancelRequested: cancelled,
          error: cancelled ? null : (error instanceof Error ? error.message : String(error)),
          event: {
            phase: cancelled ? "cancelled" : "failed",
            message: cancelled ? "The user cancelled this task." : "The task stopped after an error.",
            iteration: 0
          }
        });
      } catch {}
    }
    if (cancelled) cancelChatJob(jobId);
    else failChat(jobId, error);
  }
}

function startChat(body) {
  const message = typeof body.message === "string" ? body.message.trim() : "";
  if (!message || message.length > 12000) throw new Error("Message must contain 1–12,000 characters.");
  if (activeChatJobId) {
    const active = chatJobs.get(activeChatJobId);
    if (active?.status === "running") throw new Error("SYNTRA is still answering the previous message.");
    activeChatJobId = null;
  }
  const jobId = crypto.randomUUID();
  const controller = new AbortController();
  const job = {
    jobId, status: "running", phase: "starting", message: "Starting local response",
    reply: "", error: null, workspace: activeWorkspace, controller,
    updatedAt: new Date().toISOString()
  };
  chatJobs.set(jobId, job);
  activeChatJobId = jobId;
  pruneChatJobs();
  void runChatTurn(jobId, message, Array.isArray(body.history) ? body.history : [], controller.signal);
  return { jobId, status: job.status, phase: job.phase };
}

function readChatStatus(body) {
  const job = chatJobs.get(typeof body.jobId === "string" ? body.jobId : "");
  if (!job) throw new Error("Chat turn not found or expired. Send your message again.");
  const { controller, ...visible } = job;
  return visible;
}

const routeHandlers = {
  "GET /local-api/health": async () => ({
    status: "ok",
    service: "SYNTRA local chat and coding agent",
    host: HOST,
    port: PORT,
    capabilities: ["local-chat", "local-network-diagnostics", "repository-import-graph", "structured-patches", "bounded-repair-loop", "sandbox-build-tests", "persistent-proposal-review", "permissioned-tool-runtime", "audited-tool-execution", "durable-agent-runs", "checkpoint-resume", "chat-cancellation", "evidence-driven-check-discovery", "candidate-integrity-observation", "model-selected-readonly-retrieval", "verified-implementation-slices"]
  }),
  "POST /local-api/chat": startChat,
  "POST /local-api/chat/status": readChatStatus,
  "POST /local-api/chat/cancel": cancelChat,
  "GET /local-api/agent/runs": listCurrentAgentRuns,
  "POST /local-api/agent/runs/get": readCurrentAgentRun,
  "POST /local-api/agent/runs/resume": resumeAgentRun,
  "GET /local-api/tools": async () => ({ tools: listWorkspaceTools() }),
  "POST /local-api/tools/execute": async body => executeWorkspaceTool({
    root: body.root || activeWorkspace,
    auditRoot: body.root || activeWorkspace,
    name: body.name,
    input: body.input || {},
    approval: body.approval === true
  }),
  "GET /local-api/workspace/status": async () => readWorkspaceStatus(),
  "POST /local-api/workspace/switch": switchWorkspace,
  "GET /local-api/ai/status": async () => aiRequest("GET", "/model/status"),
  "GET /local-api/ai/train/status": async () => aiRequest("GET", "/model/train/status"),
  "POST /local-api/ai/train": async body => aiRequest("POST", "/model/train", body),
  "POST /local-api/ai/finetune": async body => aiRequest("POST", "/model/finetune", body),
  "POST /local-api/agent/generate": generateAgentCompletion,
  "POST /local-api/workspace/inspect": async body => executeToolOutput({
    root: body.root || activeWorkspace, name: "workspace.inspect", input: {}
  }),
  "POST /local-api/workspace/search": async body => executeToolOutput({
    root: body.root || activeWorkspace, name: "workspace.search",
    input: { query: body.query, ...(body.limit === undefined ? {} : { limit: body.limit }) }
  }),
  "POST /local-api/workspace/read": async body => executeToolOutput({
    root: body.root || activeWorkspace, name: "workspace.read",
    input: { path: body.path, ...(body.startLine === undefined ? {} : { startLine: body.startLine }), ...(body.endLine === undefined ? {} : { endLine: body.endLine }) }
  }),
  "POST /local-api/workspace/plan": async body => executeToolOutput({
    root: body.root || activeWorkspace, name: "workspace.plan", input: { goal: body.goal }
  }),
  "POST /local-api/workspace/proposals": async body => executeToolOutput({
    root: body.root || activeWorkspace, name: "workspace.proposals.list", input: {}
  }),
  "POST /local-api/workspace/propose": async body => executeToolOutput({
    root: body.root || activeWorkspace, name: "workspace.proposals.stage",
    input: { path: body.path, content: body.content, ...(body.expectedSha256 ? { expectedBaseHash: body.expectedSha256 } : {}) }
  }),
  "POST /local-api/workspace/apply": async body => executeToolOutput({
    root: body.root || activeWorkspace, name: "workspace.proposals.apply",
    input: { proposalId: body.proposalId }, approval: body.confirm === true
  }),
  "POST /local-api/workspace/check": async body => executeToolOutput({
    root: body.root || activeWorkspace, auditRoot: body.root || activeWorkspace,
    name: "workspace.check",
    input: { task: body.task, ...(body.timeoutMs === undefined ? {} : { timeoutMs: body.timeoutMs }) },
    approval: body.confirm === true
  })
};

const server = http.createServer(async (request, response) => {
  const parsed = new URL(request.url || "/", "http://" + HOST + ":" + PORT);
  const key = (request.method || "GET") + " " + parsed.pathname;
  const handler = routeHandlers[key];
  if (!handler) {
    sendJson(response, 404, { error: "Route not found." });
    return;
  }
  if (request.method !== "GET") {
    const origin = request.headers.origin;
    if (origin && !ALLOWED_ORIGINS.has(origin)) {
      sendJson(response, 403, { error: "Origin is not allowed by the local workspace service." });
      return;
    }
    if (!origin && request.headers["sec-fetch-site"] === "cross-site") {
      sendJson(response, 403, { error: "Cross-site requests are not accepted." });
      return;
    }
  }
  try {
    const body = request.method === "GET" ? {} : await parseBody(request);
    const result = await handler(body);
    sendJson(response, 200, result);
  } catch (error) {
    sendJson(response, 400, {
      error: error instanceof Error ? error.message : "Unknown local workspace error."
    });
  }
});

try {
  const recoveredRuns = await recoverInterruptedAgentRuns(activeWorkspace);
  if (recoveredRuns.length) {
    process.stdout.write("Recovered " + recoveredRuns.length + " interrupted agent run(s) for the active workspace.\n");
  }
} catch (error) {
  process.stderr.write("SYNTRA could not recover prior agent runs: " +
    (error instanceof Error ? error.message : String(error)) + "\n");
}

server.listen(PORT, HOST, () => {
  process.stdout.write("SYNTRA local workspace service listening on http://" + HOST + ":" + PORT + "\n");
  process.stdout.write("This service is loopback-only. Chat and code-task execution use the selected local model.\n");
});
process.on("SIGINT", () => server.close(() => process.exit(0)));
process.on("SIGTERM", () => server.close(() => process.exit(0)));

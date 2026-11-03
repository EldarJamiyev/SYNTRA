import http from "node:http";
import path from "node:path";
import { fileURLToPath, URL } from "node:url";
import crypto from "node:crypto";
import { getHostRejection, getOriginRejection } from "./origin-policy.mjs";
import { runAgentTask } from "./agent.mjs";
import { executeWorkspaceTool, listWorkspaceTools } from "./tool-runtime.mjs";
import { isCodingTask } from "./chat-intent.mjs";
import {
  cancelBrainTraining,
  generateBrainText,
  getBrainStatus,
  getBrainTrainingStatus,
  startBrainTraining
} from "./brain-runtime.mjs";
import { classifyNetworkDiagnostic, formatNetworkDiagnostic, runNetworkDiagnostic } from "./network-diagnostics.mjs";
import {
  createAgentRun,
  getAgentCandidateCheckpoint,
  getAgentRun,
  listAgentRuns,
  listPendingProposals,
  getWorkspaceExclusions,
  updateWorkspaceExclusions,
  readWorkspaceFile,
  recoverInterruptedAgentRuns,
  requestAgentRunCancellation,
  resolveWorkspace,
  saveAgentCandidateCheckpoint,
  updateAgentRun
} from "./workspace.mjs";

const HOST = "127.0.0.1";
const PORT = Number(process.env.SYNTRA_LOCAL_PORT || 3001);
const MAX_BODY_BYTES = 600 * 1024;
const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
let activeWorkspace = process.env.SYNTRA_WORKSPACE ? path.resolve(process.env.SYNTRA_WORKSPACE) : REPO_ROOT;
const chatJobs = new Map();
let activeChatJobId = null;
const API_TOKEN = String(process.env.SYNTRA_API_TOKEN || crypto.randomBytes(32).toString("hex"));
const approvalCapabilities = new Map();


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
  const signal = options?.signal;
  if (method === "GET" && route === "/model/status") return getBrainStatus();
  if (method === "GET" && route === "/model/train/status") {
    return { ...getBrainTrainingStatus() };
  }
  if (method === "POST" && route === "/model/train") {
    const root = await resolveWorkspace(body?.root || activeWorkspace);
    return startBrainTraining({
      datasetPaths: [root],
      epochs: Number(body?.epochs || 1),
      batchSize: Number(body?.batch_size || 4),
      blockSize: Number(body?.block_size || 256),
      layers: Number(body?.layers || 4),
      heads: Number(body?.heads || 4),
      width: Number(body?.width || 128),
      vocabSize: Number(body?.vocab_size || 512),
      learningRate: Number(body?.learning_rate || 3e-4),
      device: String(body?.device || "auto"),
      resume: body?.resume === true
    });
  }
  if (method === "POST" && route === "/model/generate") {
    const generation = await generateBrainText({
      prompt: body?.prompt,
      maxNewTokens: Math.min(2048, Number(body?.max_new_tokens || 256)),
      temperature: Number(body?.temperature ?? 0.7),
      topK: Math.max(1, Number(body?.top_k || 40)),
      topP: Number(body?.top_p ?? 1),
      repetitionPenalty: Number(body?.repetition_penalty ?? 1.05),
      device: String(body?.device || "auto"),
      signal,
      onToken: options?.onToken,
      onMetrics: options?.onMetrics
    });
    return {
      ...generation,
      trained_steps: generation.step,
      is_model_output: true,
      generation_mode: "standalone-local-pytorch-brain"
    };
  }
  if (method === "POST" && route === "/model/tokenize") {
    throw new Error("SYNTRA no longer provides the legacy model-tokenize route. The standalone brain manages tokenizer context during inference.");
  }
  throw new Error("Unsupported local brain operation: " + method + " " + route);
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
  const partialReply = typeof job.reply === "string" ? job.reply : "";
  Object.assign(job, {
    status: "cancelled",
    phase: "cancelled",
    message: "Task cancelled.",
    reply,
    partialReply,
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


function formatSecretScan(result) {
  const lines = [
    "**Workspace secret scan**",
    "",
    "Scanned " + result.scannedFiles.toLocaleString() + " readable text file(s) (" +
      (result.scannedBytes / 1024).toFixed(1) + " KB).",
    "Potential findings: " + result.findingCount + "."
  ];
  if (result.findings.length) {
    lines.push("", "**Findings** (matching values are intentionally hidden):");
    for (const finding of result.findings) {
      const safePath = String(finding.path || "unknown").replace(/[`\r\n]/g, "");
      lines.push("- `" + safePath + ":" + finding.line + "` — " +
        finding.type.replaceAll("_", " ") + " (" + finding.severity + ")");
    }
  } else {
    lines.push("", "No supported secret patterns were detected in the scanned files.");
  }
  if (result.skippedFiles) lines.push("", "Skipped files: " + result.skippedFiles + " (" + result.unreadableFiles + " unreadable or no longer accessible).");
  if (result.truncated) lines.push("Scan reached a file, byte, or finding limit; not every readable file may have been checked.");
  lines.push("", "This is best-effort pattern detection, not proof that the workspace is secret-free. Rotate any exposed credential and remove it from source history where appropriate.");
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
    if (/^\/secrets\s*$/i.test(message.trim())) {
      updateChat(jobId, { phase: "scanning", message: "Scanning readable workspace files locally" });
      const result = await executeWorkspaceTool({
        root: activeWorkspace, auditRoot: activeWorkspace,
        name: "workspace.secrets.scan", input: {}
      });
      if (!result.ok) throw new Error(result.error || result.message || "Workspace secret scan failed.");
      finishChat(jobId, formatSecretScan(result.output), {
        toolCalls: [{
          runId: result.runId, name: result.name, status: result.status,
          permission: result.permission, durationMs: result.durationMs,
          auditRecorded: result.auditRecorded
        }]
      });
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
        "- /secrets scans supported text files for common exposed credential patterns; secret values are never shown.",
        "- /apply applies pending proposals only after you review and explicitly request it.",
        "- /discard removes pending/stale proposal records without changing project files.",
        "- /status checks whether the local model backend is reachable.",
        "- /clear clears this conversation and its saved browser history.",
        "",
        "Normal chat and coding assistance run only on the locally trained SYNTRA PyTorch brain. Train a checkpoint with the local training command before using model-backed chat."
      ].join("\n"));
      return;
    }

    if (/^\/status\s*$/i.test(trimmedMessage)) {
      updateChat(jobId, { phase: "status", message: "Checking the local model backend" });
      try {
        const modelStatus = await aiRequest("GET", "/model/status", undefined, 5000);
        const provider = modelStatus.provider || "SYNTRA local PyTorch brain";
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
        if (!modelStatus.checkpoint_exists) lines.push("Train SYNTRA locally with: python -m brain.train --dataset \"" + activeWorkspace + "\" --epochs 1");
        finishChat(jobId, lines.join("\n"), { modelStatus: "reachable" });
      } catch (error) {
        finishChat(jobId, [
          "**SYNTRA status**",
          "",
          "Workspace service: running",
          "Model backend: unavailable",
          "Network diagnostics: available without a model.",
          "",
          "The local PyTorch checkpoint is not available. Train it from the active workspace, then retry.",
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
        "SYNTRA is running without a trained brain checkpoint. Train the local model from your selected workspace:\n\n"
        + "python -m brain.train --dataset \"" + activeWorkspace + "\" --epochs 1 --device auto\n\n"
        + "Then return here and retry. The model starts from random weights; this training step does not guarantee professional coding-assistant capability.");
      return;
    }

    const resumeMatch = message.trim().match(/^\/resume\s+([0-9a-f-]{36})$/i);
    let resumeRecord = null;
    let resumeCandidate = null;
    let taskGoal = message;
    if (resumeMatch) {
      resumeRecord = await getAgentRun(activeWorkspace, resumeMatch[1]);
      if (!resumeRecord) throw new Error("Agent run was not found in the active workspace.");
      if (resumeRecord.status !== "interrupted") {
        throw new Error("Only interrupted agent runs can be resumed. Current status: " + resumeRecord.status + ".");
      }
      resumeCandidate = await getAgentCandidateCheckpoint(activeWorkspace, resumeRecord.id);
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
          event: { phase: "resuming", message: resumeCandidate
            ? "Restoring the last verified candidate snapshot and continuing at the next implementation slice."
            : resumeRecord.recoveryPatch
              ? "Restoring and revalidating the saved patch checkpoint."
              : "Restarting task discovery because no verified candidate or patch checkpoint was saved." }
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
        modelSelfReview: true,
        runId: agentRunId,
        resumeCandidate,
        onCandidateCheckpoint: checkpoint => saveAgentCandidateCheckpoint(activeWorkspace, agentRunId, checkpoint),
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
        message: applied.length
          ? "Verified changes were applied."
          : result.verification?.status === "slice_incomplete"
            ? "Agent stopped with an incomplete implementation slice; review the unverified cumulative proposals."
            : result.verification?.passed
              ? "Agent task finished; review any staged proposals."
              : "Agent task finished with unverified proposals; inspect the verification details before applying.",
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
          .slice(-8).map(item => ({ role: item.role, content: item.content.slice(-1800) }))
      : [];
    let userMessage = message.slice(-6000);
    const contextTokens = Number(modelStatus.context_tokens || modelStatus.config?.block_size || 256);
    const promptByteBudget = Math.max(512, Math.min(200_000, contextTokens * 3));
    let prompt = "";
    while (true) {
      prompt = [
        "SYNTRA ASSISTANT",
        "You are a locally trained experimental language model. Respond to the latest user message. Do not claim to have executed tools or verified facts that are not present in the conversation.",
        ...recentHistory.map(item => (item.role === "user" ? "USER: " : "ASSISTANT: ") + item.content),
        "USER: " + userMessage,
        "ASSISTANT:"
      ].join("\n\n");
      if (Buffer.byteLength(prompt, "utf8") <= promptByteBudget) break;
      if (recentHistory.length) recentHistory.shift();
      else if (userMessage.length > 256) userMessage = userMessage.slice(-Math.max(256, Math.floor(userMessage.length * 0.7)));
      else break;
    }
    updateChat(jobId, {
      phase: "generating",
      message: "Generating with the local PyTorch brain",
      metrics: {
        epoch: Number(modelStatus.epoch || 0),
        step: Number(modelStatus.trained_steps || 0),
        train_loss: modelStatus.train_loss,
        validation_loss: modelStatus.validation_loss,
        tokens_per_second: 0,
        device: "starting"
      }
    });
    const generation = await aiRequest("POST", "/model/generate", {
      prompt,
      max_new_tokens: Math.min(512, Math.max(16, Math.floor(contextTokens * 0.75))),
      temperature: 0.7,
      top_k: Math.min(40, Number(modelStatus.tokenizer_vocab_size || modelStatus.config?.vocab_size || 512)),
      top_p: 0.95,
      repetition_penalty: 1.08,
      device: "auto"
    }, 300000, {
      signal,
      onToken: (_delta, fullText, metrics) => updateChat(jobId, {
        phase: "streaming",
        message: "Responding with the local brain",
        reply: fullText,
        metrics: { ...(job.metrics || {}), ...(metrics || {}) }
      }),
      onMetrics: metrics => updateChat(jobId, {
        metrics: { ...(job.metrics || {}), ...(metrics || {}) }
      })
    });
    const reply = String(generation.text || "").trim();
    finishChat(jobId, reply || "The local model returned an empty response. This may happen with an early checkpoint; more local training or instruction examples may be needed.", {
      trainedSteps: generation.trained_steps,
      metrics: {
        ...(job.metrics || {}),
        tokens_generated: generation.tokens_generated,
        tokens_per_second: generation.tokens_per_second,
        epoch: generation.epoch,
        step: generation.step,
        train_loss: generation.train_loss,
        validation_loss: generation.validation_loss,
        device: generation.device
      }
    });

  } catch (error) {
    const cancelled = Boolean(signal?.aborted || error?.name === "AbortError");
    if (agentRunId) {
      try { await saveAgentCandidateCheckpoint(activeWorkspace, agentRunId, null); } catch {}
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

function writeSse(response, event, value) {
  if (response.destroyed || response.writableEnded) return false;
  response.write("event: " + event + "\n");
  response.write("data: " + JSON.stringify(value) + "\n\n");
  return true;
}

async function streamChat(request, response) {
  let started;
  try {
    const body = await parseBody(request);
    started = startChat(body);
  } catch (error) {
    sendJson(response, 400, { error: error instanceof Error ? error.message : "Could not start the local chat turn." });
    return;
  }

  response.writeHead(200, {
    "Content-Type": "text/event-stream; charset=utf-8",
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
    "X-Content-Type-Options": "nosniff"
  });
  response.flushHeaders?.();
  writeSse(response, "start", started);

  let sentText = "";
  let lastPhase = "";
  let lastMetrics = "";
  let completed = false;
  const job = chatJobs.get(started.jobId);
  if (!job) {
    writeSse(response, "error", { error: "Chat job disappeared before streaming started." });
    response.end();
    return;
  }

  const finish = () => {
    if (completed) return;
    completed = true;
    clearInterval(pump);
    clearInterval(heartbeat);
    response.end();
  };

  const pump = setInterval(() => {
    if (response.destroyed || response.writableEnded) {
      finish();
      return;
    }
    const current = chatJobs.get(started.jobId);
    if (!current) {
      writeSse(response, "error", { error: "Chat job expired before completion." });
      finish();
      return;
    }

    if (current.phase && current.phase !== lastPhase) {
      lastPhase = current.phase;
      writeSse(response, "phase", { phase: current.phase, message: current.message || "" });
    }

    const output = current.status === "failed" || current.status === "cancelled"
      ? String(current.partialReply || "")
      : String(current.reply || "");
    if (output !== sentText) {
      if (output.startsWith(sentText)) {
        const delta = output.slice(sentText.length);
        sentText = output;
        if (delta) writeSse(response, "delta", { text: delta });
      } else {
        sentText = output;
        writeSse(response, "replace", { text: output });
      }
    }

    if (current.metrics) {
      const signature = JSON.stringify(current.metrics);
      if (signature !== lastMetrics) {
        lastMetrics = signature;
        writeSse(response, "metrics", current.metrics);
      }
    }

    if (current.status === "running") return;

    if (current.status === "completed") {
      const finalText = String(current.reply || "");
      if (finalText && finalText !== sentText) {
        if (finalText.startsWith(sentText)) {
          const delta = finalText.slice(sentText.length);
          if (delta) writeSse(response, "delta", { text: delta });
        } else {
          writeSse(response, "replace", { text: finalText });
        }
      }
      writeSse(response, "done", {
        status: "completed",
        jobId: current.jobId,
        reply: finalText,
        metrics: current.metrics || null,
        trainedSteps: current.trainedSteps || null
      });
      finish();
      return;
    }

    if (current.status === "cancelled") {
      writeSse(response, "done", {
        status: "cancelled",
        jobId: current.jobId,
        reply: current.reply || "Generation cancelled.",
        partialReply: current.partialReply || sentText,
        metrics: current.metrics || null
      });
      finish();
      return;
    }

    if (current.status === "failed") {
      writeSse(response, "error", {
        error: current.error || current.reply || "Local chat generation failed.",
        partialReply: current.partialReply || sentText
      });
      writeSse(response, "done", {
        status: "failed",
        jobId: current.jobId,
        reply: current.partialReply || sentText,
        error: current.error || current.reply || "Local chat generation failed.",
        metrics: current.metrics || null
      });
      finish();
    }
  }, 35);

  const heartbeat = setInterval(() => {
    if (!completed && !response.destroyed && !response.writableEnded) {
      response.write(": keep-alive\n\n");
    }
  }, 15_000);

  response.on("close", () => {
    if (!completed) {
      const current = chatJobs.get(started.jobId);
      if (current?.status === "running") current.controller?.abort();
      finish();
    }
  });
}

async function trainLocalBrain(body) {
  const root = await resolveWorkspace(body.root || activeWorkspace);
  return startBrainTraining({
    datasetPaths: [root],
    epochs: Number(body.epochs || 1),
    batchSize: Number(body.batchSize || body.batch_size || 4),
    blockSize: Number(body.blockSize || body.block_size || 256),
    layers: Number(body.layers || 4),
    heads: Number(body.heads || 4),
    width: Number(body.width || 128),
    vocabSize: Number(body.vocabSize || body.vocab_size || 512),
    learningRate: Number(body.learningRate || body.learning_rate || 3e-4),
    device: String(body.device || "auto"),
    resume: body.resume === true
  });
}

function hasValidApiToken(request) {
  const value = String(request.headers?.authorization || "");
  const supplied = Buffer.from(value.startsWith("Bearer ") ? value.slice(7) : "", "utf8");
  const expected = Buffer.from(API_TOKEN, "utf8");
  return supplied.length === expected.length && crypto.timingSafeEqual(supplied, expected);
}


function canonicalApprovalValue(value) {
  if (Array.isArray(value)) return value.map(canonicalApprovalValue);
  if (value && typeof value === "object") return Object.fromEntries(Object.keys(value).sort().map(key => [key, canonicalApprovalValue(value[key])]));
  return value;
}
function approvalFingerprint(root, name, input) {
  return crypto.createHash("sha256").update(JSON.stringify(canonicalApprovalValue({ root, name, input }))).digest("hex");
}
async function issueApprovalCapability(body) {
  const name = typeof body.name === "string" ? body.name : "";
  const input = body.input && typeof body.input === "object" && !Array.isArray(body.input) ? body.input : {};
  const root = await resolveWorkspace(body.root || activeWorkspace);
  const definition = listWorkspaceTools().find(item => item.name === name);
  if (!definition || definition.approvalRequired !== true) throw new Error("An approval capability can only be issued for a tool that requires approval.");
  const now = Date.now();
  for (const [key, value] of approvalCapabilities) if (value.expiresAt <= now) approvalCapabilities.delete(key);
  while (approvalCapabilities.size >= 128) approvalCapabilities.delete(approvalCapabilities.keys().next().value);
  const token = crypto.randomBytes(32).toString("hex");
  approvalCapabilities.set(token, { root, name, fingerprint: approvalFingerprint(root, name, input), expiresAt: now + 60000 });
  return { approvalToken: token, expiresInSeconds: 60, name };
}
function consumeApprovalCapability(token, root, name, input) {
  if (typeof token !== "string" || !/^[a-f0-9]{64}$/.test(token)) return false;
  const capability = approvalCapabilities.get(token);
  approvalCapabilities.delete(token);
  return Boolean(capability && capability.expiresAt > Date.now() && capability.root === root && capability.name === name
    && capability.fingerprint === approvalFingerprint(root, name, input));
}

const routeHandlers = {
  "GET /local-api/session": async () => ({ token: API_TOKEN }),
  "GET /local-api/brain/status": getBrainStatus,
  "GET /local-api/brain/train/status": async () => getBrainTrainingStatus(),
  "POST /local-api/brain/train": trainLocalBrain,
  "POST /local-api/brain/train/cancel": async () => cancelBrainTraining(),
  "GET /local-api/ai/status": getBrainStatus,
  "GET /local-api/ai/train/status": async () => getBrainTrainingStatus(),
  "POST /local-api/ai/train": trainLocalBrain,
  "POST /local-api/approvals": issueApprovalCapability,
  "GET /": async () => ({
    status: "ok",
    service: "SYNTRA local API",
    message: "The workspace API is running. Open the SYNTRA frontend at http://127.0.0.1:3000/.",
    frontend: "http://127.0.0.1:3000/",
    health: "http://127.0.0.1:" + PORT + "/local-api/health"
  }),
  "GET /local-api/health": async () => ({
    status: "ok",
    service: "SYNTRA local chat and coding agent",
    host: HOST,
    port: PORT,
    capabilities: ["local-chat", "sse-chat-streaming", "standalone-pytorch-brain", "local-model-training", "local-network-diagnostics", "repository-import-graph", "structured-patches", "bounded-repair-loop", "sandbox-build-tests", "persistent-proposal-review", "permissioned-tool-runtime", "audited-tool-execution", "durable-agent-runs", "checkpoint-resume", "chat-cancellation", "evidence-driven-check-discovery", "candidate-integrity-observation", "model-selected-readonly-retrieval", "verified-implementation-slices", "workspace-secret-scanning"]
  }),
  "POST /local-api/chat": startChat,
  "POST /local-api/chat/status": readChatStatus,
  "POST /local-api/chat/cancel": cancelChat,
  "GET /local-api/agent/runs": listCurrentAgentRuns,
  "POST /local-api/agent/runs/get": readCurrentAgentRun,
  "POST /local-api/agent/runs/resume": resumeAgentRun,
  "GET /local-api/tools": async () => ({ tools: listWorkspaceTools() }),
  "POST /local-api/tools/execute": async body => {
    const root = await resolveWorkspace(body.root || activeWorkspace);
    const input = body.input || {};
    const approved = consumeApprovalCapability(body.approvalToken, root, body.name, input);
    return executeWorkspaceTool({ root, auditRoot: root, name: body.name, input, approval: approved });
  },
  "GET /local-api/workspace/status": async () => readWorkspaceStatus(),
  "GET /local-api/workspace/exclusions": async () => getWorkspaceExclusions(activeWorkspace),
  "POST /local-api/workspace/exclusions": async body => updateWorkspaceExclusions(activeWorkspace, body),
  "POST /local-api/workspace/switch": switchWorkspace,
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
  "POST /local-api/workspace/apply": async body => {
    const root = await resolveWorkspace(body.root || activeWorkspace);
    const input = { proposalId: body.proposalId };
    const approval = consumeApprovalCapability(body.approvalToken, root, "workspace.proposals.apply", input);
    return executeToolOutput({ root, name: "workspace.proposals.apply", input, approval });
  },
  "POST /local-api/workspace/check": async body => {
    const root = await resolveWorkspace(body.root || activeWorkspace);
    const input = { task: body.task, ...(body.timeoutMs === undefined ? {} : { timeoutMs: body.timeoutMs }) };
    const approval = consumeApprovalCapability(body.approvalToken, root, "workspace.check", input);
    return executeToolOutput({ root, auditRoot: root, name: "workspace.check", input, approval });
  }
};

const server = http.createServer(async (request, response) => {
  const parsed = new URL(request.url || "/", "http://" + HOST + ":" + PORT);
  const key = (request.method || "GET") + " " + parsed.pathname;
  const hostRejection = getHostRejection(request, request.socket?.localPort || PORT);
  if (hostRejection) { sendJson(response, 403, { error: hostRejection }); return; }
  const originRejection = getOriginRejection(request);
  if (originRejection) { sendJson(response, 403, { error: originRejection }); return; }
  const publicRoute = key === "GET /" || key === "GET /local-api/health" || key === "GET /local-api/session";
  if (!publicRoute && !hasValidApiToken(request)) {
    sendJson(response, 401, { error: "A valid SYNTRA local session token is required." }); return;
  }
  if (key === "POST /local-api/chat/stream" || key === "POST /api/chat/stream") {
    await streamChat(request, response);
    return;
  }
  const handler = routeHandlers[key];
  if (!handler) { sendJson(response, 404, { error: "Route not found." }); return; }
  try {
    const body = request.method === "GET" ? {} : await parseBody(request);
    const result = await handler(body);
    sendJson(response, 200, result);
  } catch (error) {
    sendJson(response, 400, { error: error instanceof Error ? error.message : "Unknown local workspace error." });
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
  process.stdout.write("This service is loopback-only. Chat and code tasks run through the standalone local PyTorch brain.\n");
});
process.on("SIGINT", () => server.close(() => process.exit(0)));
process.on("SIGTERM", () => server.close(() => process.exit(0)));

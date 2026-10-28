import http from "node:http";
import { URL } from "node:url";
import {
  applyFileProposal,
  createProjectPlan,
  inspectWorkspace,
  listProposals,
  proposeFileChange,
  readWorkspaceFile,
  runProjectCheck,
  searchWorkspace
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

async function aiRequest(method, route, body) {
  let response;
  try {
    response = await fetch(AI_BASE + route, {
      method,
      headers: method === "GET" ? {} : { "Content-Type": "application/json" },
      body: method === "GET" ? undefined : JSON.stringify(body || {}),
      signal: AbortSignal.timeout(method === "POST" && route === "/model/generate" ? 300000 : 15000)
    });
  } catch {
    throw new Error("SYNTRA AI service is offline. Install requirements.txt and run npm run ai:server.");
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

const routeHandlers = {
  "GET /local-api/health": async () => ({
    status: "ok",
    service: "SYNTRA Local Project Workspace",
    host: HOST,
    port: PORT,
    capabilities: ["inspect", "search", "read", "plan", "propose", "apply-confirmed", "build-checks", "local-model-proxy"]
  }),
  "GET /local-api/ai/status": async () => aiRequest("GET", "/model/status"),
  "GET /local-api/ai/train/status": async () => aiRequest("GET", "/model/train/status"),
  "POST /local-api/ai/train": async body => aiRequest("POST", "/model/train", body),
  "POST /local-api/ai/finetune": async body => aiRequest("POST", "/model/finetune", body),
  "POST /local-api/agent/generate": generateAgentCompletion,
  "POST /local-api/workspace/inspect": async body => inspectWorkspace(body.root),
  "POST /local-api/workspace/search": async body => searchWorkspace(body.root, body.query, body.limit),
  "POST /local-api/workspace/read": async body => readWorkspaceFile(body.root, body.path),
  "POST /local-api/workspace/plan": async body => createProjectPlan(body.root, body.goal),
  "POST /local-api/workspace/proposals": async body => listProposals(body.root),
  "POST /local-api/workspace/propose": async body => proposeFileChange(body.root, body.path, body.content, body.expectedSha256),
  "POST /local-api/workspace/apply": async body => applyFileProposal(body.root, body.proposalId, body.confirm === true),
  "POST /local-api/workspace/check": async body => runProjectCheck(body.root, body.task, { timeoutMs: body.timeoutMs })
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

server.listen(PORT, HOST, () => {
  process.stdout.write("SYNTRA local workspace service listening on http://" + HOST + ":" + PORT + "\n");
  process.stdout.write("This service is loopback-only. Review proposals before applying changes.\n");
});
process.on("SIGINT", () => server.close(() => process.exit(0)));
process.on("SIGTERM", () => server.close(() => process.exit(0)));

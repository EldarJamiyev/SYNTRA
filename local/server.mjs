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

const routeHandlers = {
  "GET /local-api/health": async () => ({
    status: "ok",
    service: "SYNTRA Local Project Workspace",
    host: HOST,
    port: PORT,
    capabilities: ["inspect", "search", "read", "plan", "propose", "apply-confirmed", "build-checks"]
  }),
  "POST /local-api/workspace/inspect": async body => inspectWorkspace(body.root),
  "POST /local-api/workspace/search": async body => searchWorkspace(body.root, body.query, body.limit),
  "POST /local-api/workspace/read": async body => readWorkspaceFile(body.root, body.path),
  "POST /local-api/workspace/plan": async body => createProjectPlan(body.root, body.goal),
  "POST /local-api/workspace/proposals": async body => listProposals(body.root),
  "POST /local-api/workspace/propose": async body => proposeFileChange(body.root, body.path, body.content),
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

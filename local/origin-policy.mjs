const DEFAULT_ALLOWED_ORIGINS = new Set(["http://127.0.0.1:3000","http://localhost:3000","http://127.0.0.1:5173","http://localhost:5173"]);
const LOCAL_HOST_HEADER = /^(?:localhost|127\.0\.0\.1|\[::1\])(?::[0-9]{1,5})?$/i;
const LOCAL_ORIGIN_HOSTS = new Set(["localhost","127.0.0.1","[::1]"]);
function firstHeader(value) {
  if (Array.isArray(value)) return String(value[0] || "").split(",")[0].trim();
  return typeof value === "string" ? value.split(",")[0].trim() : "";
}
function validLocalOrigin(value) {
  let parsed;
  try { parsed = new URL(value); } catch { return false; }
  if (!["http:","https:"].includes(parsed.protocol) || parsed.username || parsed.password
    || parsed.pathname !== "/" || parsed.search || parsed.hash) return false;
  return LOCAL_ORIGIN_HOSTS.has(parsed.hostname.toLowerCase());
}
/** Direct loopback service: forwarded host headers are never trusted. */
export function getHostRejection(request, expectedPort = 3001) {
  const host = request.headers?.host;
  if (typeof host !== "string" || !host || host.length > 255 || host.includes(",")) {
    return "The local API only accepts one valid loopback Host header.";
  }
  if (!Number.isInteger(expectedPort) || expectedPort < 1 || expectedPort > 65535) {
    return "The local API listener port is invalid.";
  }
  let parsed;
  try { parsed = new URL("http://" + host); }
  catch { return "The local API only accepts loopback Host headers."; }
  if (parsed.username || parsed.password || parsed.pathname !== "/" || parsed.search || parsed.hash
    || !["localhost", "127.0.0.1"].includes(parsed.hostname.toLowerCase())) {
    return "The local API only accepts loopback Host headers.";
  }
  if (!parsed.port || Number(parsed.port) !== expectedPort) {
    return "The Host header must match the local API listener port.";
  }
  return null;
}
/** Never reflect an arbitrary Origin from Host; this prevents DNS-rebinding trust. */
export function getOriginRejection(request) {
  const site = firstHeader(request.headers?.["sec-fetch-site"]).toLowerCase();
  if (site === "cross-site") return "Cross-site requests are not accepted.";
  const origin = firstHeader(request.headers?.origin);
  if (!origin) return null;
  const configured = String(process.env.SYNTRA_ALLOWED_ORIGINS || "").split(",").map(v => v.trim()).filter(Boolean);
  const allowlist = new Set([...DEFAULT_ALLOWED_ORIGINS, ...configured.filter(validLocalOrigin)]);
  return allowlist.has(origin) && validLocalOrigin(origin) ? null : "Origin is not allowed by the local workspace service.";
}

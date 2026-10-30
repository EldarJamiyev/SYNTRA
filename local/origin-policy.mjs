const DEFAULT_ALLOWED_ORIGINS = new Set([
  "http://127.0.0.1:3000",
  "http://localhost:3000",
  "http://127.0.0.1:5173",
  "http://localhost:5173"
]);

function firstHeader(value) {
  if (Array.isArray(value)) return String(value[0] || "").split(",")[0].trim();
  return typeof value === "string" ? value.split(",")[0].trim() : "";
}

/**
 * Return a rejection message for an unsafe browser origin, or null when allowed.
 *
 * The default local origins remain supported. A reverse-proxied preview is also
 * allowed when its Origin host matches the original Host / X-Forwarded-Host
 * preserved by the same-origin Vite proxy. This is an exact host comparison,
 * never a wildcard. Deployments with a different proxy arrangement can add
 * exact origins through SYNTRA_ALLOWED_ORIGINS.
 */
export function getOriginRejection(request) {
  const site = firstHeader(request.headers?.["sec-fetch-site"]).toLowerCase();
  if (site === "cross-site") return "Cross-site requests are not accepted.";

  const origin = firstHeader(request.headers?.origin);
  if (!origin) return null;

  const configured = String(process.env.SYNTRA_ALLOWED_ORIGINS || "")
    .split(",")
    .map(value => value.trim())
    .filter(Boolean);
  const allowlist = new Set([...DEFAULT_ALLOWED_ORIGINS, ...configured]);
  if (allowlist.has(origin)) return null;

  let parsedOrigin;
  try {
    parsedOrigin = new URL(origin);
  } catch {
    return "Origin is not allowed by the local workspace service.";
  }

  if (parsedOrigin.protocol !== "http:" && parsedOrigin.protocol !== "https:") {
    return "Origin is not allowed by the local workspace service.";
  }

  const forwardedHost = firstHeader(request.headers?.["x-forwarded-host"]);
  const requestHost = forwardedHost || firstHeader(request.headers?.host);
  if (!requestHost || parsedOrigin.host.toLowerCase() !== requestHost.toLowerCase()) {
    return "Origin is not allowed by the local workspace service.";
  }

  const forwardedProto = firstHeader(request.headers?.["x-forwarded-proto"]).toLowerCase();
  if (forwardedProto && parsedOrigin.protocol !== forwardedProto.replace(/:$/, "") + ":") {
    return "Origin is not allowed by the local workspace service.";
  }

  return null;
}

import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

export const SUPPORTED_NODE_RANGE = ">=22.12.0";

const NODE_VERSION_PATTERN = /^v?(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-([0-9A-Za-z.-]+))?(?:\+[0-9A-Za-z.-]+)?$/u;

export function parseNodeVersion(value) {
  if (typeof value !== "string") return null;
  const match = NODE_VERSION_PATTERN.exec(value.trim());
  if (!match) return null;

  const major = Number(match[1]);
  const minor = Number(match[2]);
  const patch = Number(match[3]);
  if (![major, minor, patch].every(Number.isSafeInteger)) return null;

  return {
    major,
    minor,
    patch,
    prerelease: match[4] ?? null
  };
}

export function isSupportedNodeVersion(value) {
  const version = parseNodeVersion(value);
  if (!version || version.prerelease !== null) return false;

  return version.major > 22 || (version.major === 22 && version.minor >= 12);
}

export function getNodeVersionError(value) {
  const actual = typeof value === "string" && value.trim() ? value.trim() : "unknown";
  return "SYNTRA requires Node.js " + SUPPORTED_NODE_RANGE
    + " to satisfy Vite 8 and SYNTRA's supported-runtime policy. Detected: " + actual + ".";
}

function runVersionCheck() {
  const current = process.versions.node;
  if (!isSupportedNodeVersion(current)) {
    console.error(getNodeVersionError(current));
    process.exitCode = 1;
    return;
  }
  console.log("Node.js " + current + " satisfies SYNTRA's " + SUPPORTED_NODE_RANGE + " runtime requirement.");
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  runVersionCheck();
}

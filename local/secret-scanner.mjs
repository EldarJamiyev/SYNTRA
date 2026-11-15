import fs from "node:fs/promises";
import { constants as fsConstants } from "node:fs";
import path from "node:path";

import {
  listWorkspaceTextFiles,
  readWorkspaceFile,
  resolveWorkspace
} from "./workspace.mjs";

const MAX_SCAN_FILES = 2000;
const MAX_SCAN_BYTES = 32 * 1024 * 1024;
const MAX_RETURNED_FINDINGS = 100;
const MAX_ALLOWLIST_BYTES = 512 * 1024;
const MAX_ALLOWLIST_ENTRIES = 500;
const MAX_ALLOWLIST_DAYS = 366;
const ALLOWLIST_SCHEMA = "syntra-secret-allowlist-v1";

const SECRET_RULES = [
  { type: "private_key_header", severity: "critical", expression: /-----BEGIN (?:RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY-----|-----BEGIN PGP PRIVATE KEY BLOCK-----/i },
  { type: "github_token", severity: "high", expression: /\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b/ },
  { type: "gitlab_token", severity: "high", expression: /\bglpat-[A-Za-z0-9_-]{20,}\b/ },
  { type: "slack_token", severity: "high", expression: /\bxox[baprs]-[A-Za-z0-9-]{20,}\b/ },
  { type: "google_api_key", severity: "high", expression: /\bAIza[0-9A-Za-z_-]{35}\b/ },
  { type: "google_oauth_client_secret", severity: "high", expression: /\bGOCSPX-[A-Za-z0-9_-]{20,}\b/ },
  { type: "stripe_live_secret", severity: "critical", expression: /\b(?:sk|rk)_live_[A-Za-z0-9]{16,}\b/ },
  { type: "openai_api_key", severity: "high", expression: /\bsk-(?:proj-)?[A-Za-z0-9_-]{24,}\b/ },
  { type: "anthropic_api_key", severity: "high", expression: /\bsk-ant-(?:api\d{2}-)?[A-Za-z0-9_-]{24,}\b/ },
  { type: "huggingface_token", severity: "high", expression: /\bhf_[A-Za-z0-9]{30,}\b/ },
  { type: "aws_access_key_id", severity: "medium", expression: /\b(?:AKIA|ASIA)[A-Z0-9]{16}\b/ },
  { type: "npm_token", severity: "high", expression: /\bnpm_[A-Za-z0-9]{36}\b/ },
  { type: "pypi_token", severity: "high", expression: /\bpypi-[A-Za-z0-9_-]{50,}\b/ },
  { type: "telegram_bot_token", severity: "high", expression: /\b\d{8,12}:[A-Za-z0-9_-]{30,}\b/ },
  { type: "sendgrid_api_key", severity: "high", expression: /\bSG\.[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}\b/ },
  { type: "twilio_api_key", severity: "high", expression: /\bSK[0-9a-fA-F]{32}\b/ },
  { type: "hashicorp_vault_token", severity: "high", expression: /\bhvs\.[A-Za-z0-9_-]{20,}\b/ },
  { type: "mailgun_api_key", severity: "high", expression: /\bkey-[0-9a-fA-F]{32}\b/ },
  { type: "discord_bot_token", severity: "high", expression: /\b[A-Za-z0-9_-]{24}\.[A-Za-z0-9_-]{6}\.[A-Za-z0-9_-]{25,}\b/ },
  { type: "bearer_token", severity: "high", expression: /\bBearer\s+[A-Za-z0-9._~-]{20,}/i },
  { type: "basic_auth_header", severity: "high", expression: /\bAuthorization\s*:\s*Basic\s+[A-Za-z0-9+/]{24,}={0,2}/i },
  { type: "jwt", severity: "high", expression: /\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b/ },
  { type: "database_url_credentials", severity: "high", expression: /\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?):\/\/[^\s/:@]+:[^\s/@]{8,}@/i },
  {
    type: "generic_secret_assignment",
    severity: "high",
    expression: /\b(?:password|passwd|secret|api[_-]?key|access[_-]?token|refresh[_-]?token|session[_-]?token|auth[_-]?token|client[_-]?secret|private[_-]?key|signing[_-]?key|encryption[_-]?key|webhook[_-]?secret|service[_-]?token|credential|aws_secret_access_key)\b\s*[:=]\s*(?:"(?!your(?:[_ -]|\b)|change[ _-]?me\b|changeme\b|example\b|placeholder\b|process\.env\b|env\(|\$\{|<)[^"\r\n]{12,}"|'(?!your(?:[_ -]|\b)|change[ _-]?me\b|changeme\b|example\b|placeholder\b|process\.env\b|env\(|\$\{|<)[^'\r\n]{12,}'|(?!your(?:[_ -]|\b)|change[ _-]?me\b|changeme\b|example\b|placeholder\b|process\.env\b|env\(|\$\{|<|%[A-Z_]+%)[A-Za-z0-9_./+=-]{16,})/i
  }
];

const SECRET_RULE_TYPES = new Set(SECRET_RULES.map(rule => rule.type));

function sameFileIdentity(left, right) {
  return left.isFile()
    && right.isFile()
    && left.dev === right.dev
    && left.ino === right.ino
    && left.size === right.size
    && left.mtimeNs === right.mtimeNs
    && left.ctimeNs === right.ctimeNs;
}

function invalidAllowlist(message) {
  return {
    status: "invalid",
    entries: [],
    expiredEntries: 0,
    warning: message
  };
}

function validRelativeAllowlistPath(value) {
  if (typeof value !== "string" || !value || value.length > 4096) return false;
  if (value.startsWith("/") || value.includes("\\") || value.includes("\0")) return false;
  const parts = value.split("/");
  return parts.every(part => part && part !== "." && part !== "..")
    && !/^[A-Za-z]:/.test(value);
}

function validateAllowlistEntry(entry, now) {
  if (!entry || typeof entry !== "object" || Array.isArray(entry)) return null;
  const expectedKeys = ["approved", "expiresAt", "path", "reason", "ruleTypes", "sha256"];
  const keys = Object.keys(entry).sort();
  if (keys.length !== expectedKeys.length || keys.some((key, index) => key !== expectedKeys[index])) return null;
  if (entry.approved !== true || !validRelativeAllowlistPath(entry.path)) return null;
  if (typeof entry.sha256 !== "string" || !/^[a-f0-9]{64}$/.test(entry.sha256)) return null;
  if (!Array.isArray(entry.ruleTypes) || !entry.ruleTypes.length || entry.ruleTypes.length > SECRET_RULES.length) return null;
  if (new Set(entry.ruleTypes).size !== entry.ruleTypes.length
    || entry.ruleTypes.some(type => typeof type !== "string" || !SECRET_RULE_TYPES.has(type))) return null;
  if (typeof entry.reason !== "string" || entry.reason.trim().length < 20
    || entry.reason.trim().length > 500 || /[\0\r\n]/.test(entry.reason)) return null;
  if (typeof entry.expiresAt !== "string" || !/^\d{4}-\d{2}-\d{2}$/.test(entry.expiresAt)) return null;

  const expiration = Date.parse(entry.expiresAt + "T00:00:00.000Z");
  const normalizedExpiration = Number.isFinite(expiration)
    ? new Date(expiration).toISOString().slice(0, 10)
    : null;
  if (normalizedExpiration !== entry.expiresAt) return null;
  const nowDate = new Date(now);
  const today = Date.UTC(nowDate.getUTCFullYear(), nowDate.getUTCMonth(), nowDate.getUTCDate());
  const maxExpiry = today + MAX_ALLOWLIST_DAYS * 24 * 60 * 60 * 1000;
  if (expiration > maxExpiry) return null;

  return {
    path: entry.path,
    sha256: entry.sha256,
    ruleTypes: [...entry.ruleTypes],
    expiresAt: entry.expiresAt,
    expired: expiration < today
  };
}

async function readSecretAllowlist(root) {
  const stateDirectory = path.join(root, ".syntra");
  const allowlistPath = path.join(stateDirectory, "secret-allowlist.json");
  let directoryBefore;
  try {
    directoryBefore = await fs.lstat(stateDirectory, { bigint: true });
  } catch (error) {
    if (error?.code === "ENOENT") {
      return { status: "not_configured", entries: [], expiredEntries: 0, warning: null };
    }
    return invalidAllowlist("The local secret allowlist could not be inspected safely.");
  }
  if (!directoryBefore.isDirectory() || directoryBefore.isSymbolicLink()) {
    return invalidAllowlist("The .syntra state path must be a real directory.");
  }

  let fileBefore;
  try {
    fileBefore = await fs.lstat(allowlistPath, { bigint: true });
  } catch (error) {
    if (error?.code === "ENOENT") {
      return { status: "not_configured", entries: [], expiredEntries: 0, warning: null };
    }
    return invalidAllowlist("The local secret allowlist could not be inspected safely.");
  }
  if (!fileBefore.isFile() || fileBefore.isSymbolicLink() || fileBefore.size < 1n
    || fileBefore.size > BigInt(MAX_ALLOWLIST_BYTES)) {
    return invalidAllowlist("The local secret allowlist must be a bounded regular file, not a link.");
  }

  let handle;
  try {
    const flags = fsConstants.O_RDONLY | (fsConstants.O_NOFOLLOW || 0);
    handle = await fs.open(allowlistPath, flags);
    const opened = await handle.stat({ bigint: true });
    if (!sameFileIdentity(fileBefore, opened)) {
      return invalidAllowlist("The local secret allowlist changed while it was opened.");
    }
    const raw = await handle.readFile();
    const after = await handle.stat({ bigint: true });
    const fileAfter = await fs.lstat(allowlistPath, { bigint: true });
    const directoryAfter = await fs.lstat(stateDirectory, { bigint: true });
    if (raw.length > MAX_ALLOWLIST_BYTES
      || !sameFileIdentity(opened, after)
      || !sameFileIdentity(opened, fileAfter)
      || !directoryBefore.isDirectory()
      || directoryBefore.dev !== directoryAfter.dev
      || directoryBefore.ino !== directoryAfter.ino
      || directoryAfter.isSymbolicLink()) {
      return invalidAllowlist("The local secret allowlist changed while it was read.");
    }

    const document = JSON.parse(raw.toString("utf8"));
    if (!document || typeof document !== "object" || Array.isArray(document)
      || Object.keys(document).sort().join(",") !== "entries,schema"
      || document.schema !== ALLOWLIST_SCHEMA
      || !Array.isArray(document.entries)
      || document.entries.length > MAX_ALLOWLIST_ENTRIES) {
      return invalidAllowlist("The local secret allowlist schema is invalid.");
    }

    const now = Date.now();
    const entries = [];
    let expiredEntries = 0;
    const seen = new Set();
    for (const rawEntry of document.entries) {
      const entry = validateAllowlistEntry(rawEntry, now);
      if (!entry) return invalidAllowlist("A secret allowlist entry is invalid or lacks explicit approval.");
      const key = entry.path + "\0" + entry.sha256;
      if (seen.has(key)) return invalidAllowlist("The secret allowlist contains duplicate file fingerprints.");
      seen.add(key);
      if (entry.expired) expiredEntries += 1;
      else entries.push(entry);
    }
    return { status: "loaded", entries, expiredEntries, warning: null };
  } catch {
    return invalidAllowlist("The local secret allowlist is unreadable or contains invalid JSON.");
  } finally {
    await handle?.close().catch(() => {});
  }
}

export async function scanWorkspaceSecrets(inputRoot) {
  const root = await resolveWorkspace(inputRoot);
  const [inventory, allowlist] = await Promise.all([
    listWorkspaceTextFiles(root),
    readSecretAllowlist(root)
  ]);
  const findings = [];
  const typeCounts = {};
  const severityCounts = {};
  const allowlistedTypeCounts = {};
  const entriesByFile = new Map();
  const usedAllowlistEntries = new Set();
  for (const entry of allowlist.entries) {
    const key = entry.path + "\0" + entry.sha256;
    entriesByFile.set(key, entry);
  }

  let scannedFiles = 0;
  let skippedFiles = 0;
  let unreadableFiles = 0;
  let scannedBytes = 0;
  let findingCount = 0;
  let allowlistedFindings = 0;
  let allowlistedFiles = 0;
  let truncated = inventory.truncated || inventory.files.length > MAX_SCAN_FILES;

  for (let index = 0; index < inventory.files.length; index += 1) {
    const file = inventory.files[index];
    if (index >= MAX_SCAN_FILES || scannedBytes + file.size > MAX_SCAN_BYTES) {
      skippedFiles += inventory.files.length - index;
      truncated = true;
      break;
    }

    let source;
    try {
      source = await readWorkspaceFile(root, file.path);
    } catch {
      skippedFiles += 1;
      unreadableFiles += 1;
      continue;
    }
    if (scannedBytes + source.bytes > MAX_SCAN_BYTES) {
      skippedFiles += inventory.files.length - index;
      truncated = true;
      break;
    }

    scannedFiles += 1;
    scannedBytes += source.bytes;
    const allowlistEntry = entriesByFile.get(source.path + "\0" + source.sha256);
    const allowedRuleTypes = new Set(allowlistEntry?.ruleTypes || []);
    let fileAllowlistedFindings = 0;
    const lines = source.content.split(/\r?\n/);
    for (let lineIndex = 0; lineIndex < lines.length; lineIndex += 1) {
      const line = lines[lineIndex];
      for (const rule of SECRET_RULES) {
        if (!rule.expression.test(line)) continue;
        if (allowedRuleTypes.has(rule.type)) {
          allowlistedFindings += 1;
          fileAllowlistedFindings += 1;
          allowlistedTypeCounts[rule.type] = (allowlistedTypeCounts[rule.type] || 0) + 1;
          usedAllowlistEntries.add(source.path + "\0" + source.sha256);
          continue;
        }
        findingCount += 1;
        typeCounts[rule.type] = (typeCounts[rule.type] || 0) + 1;
        severityCounts[rule.severity] = (severityCounts[rule.severity] || 0) + 1;
        if (findings.length < MAX_RETURNED_FINDINGS) {
          findings.push({
            path: source.path,
            line: lineIndex + 1,
            type: rule.type,
            severity: rule.severity,
            fileSha256: source.sha256
          });
        } else {
          truncated = true;
        }
      }
    }
    if (fileAllowlistedFindings > 0) allowlistedFiles += 1;
  }

  const staleAllowlistEntries = allowlist.entries.length - usedAllowlistEntries.size;
  return {
    root,
    scannedFiles,
    skippedFiles,
    unreadableFiles,
    scannedBytes,
    findingCount,
    returnedFindings: findings.length,
    findings,
    typeCounts,
    severityCounts,
    allowlistStatus: allowlist.status,
    allowlistWarning: allowlist.warning,
    allowlistEntries: allowlist.entries.length + allowlist.expiredEntries,
    allowlistExpiredEntries: allowlist.expiredEntries,
    staleAllowlistEntries,
    allowlistedFiles,
    allowlistedFindings,
    allowlistedTypeCounts,
    truncated,
    limits: {
      maxFiles: MAX_SCAN_FILES,
      maxBytes: MAX_SCAN_BYTES,
      maxReturnedFindings: MAX_RETURNED_FINDINGS,
      maxAllowlistBytes: MAX_ALLOWLIST_BYTES,
      maxAllowlistEntries: MAX_ALLOWLIST_ENTRIES,
      maxAllowlistDays: MAX_ALLOWLIST_DAYS
    },
    notice: "Best-effort pattern detection only. Secret values and matching source lines are never returned or logged. Allowlist exceptions require explicit approval, a reason, an expiry, and an exact file SHA-256; an edited file is scanned again. Review findings locally; this scanner does not prove that a workspace is secret-free."
  };
}

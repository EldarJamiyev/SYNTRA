import {
  listWorkspaceTextFiles,
  readWorkspaceFile,
  resolveWorkspace
} from "./workspace.mjs";

const MAX_SCAN_FILES = 2000;
const MAX_SCAN_BYTES = 32 * 1024 * 1024;
const MAX_RETURNED_FINDINGS = 100;

const SECRET_RULES = [
  { type: "private_key_header", severity: "critical", expression: /-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----/i },
  { type: "github_token", severity: "high", expression: /\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b/ },
  { type: "gitlab_token", severity: "high", expression: /\bglpat-[A-Za-z0-9_-]{20,}\b/ },
  { type: "slack_token", severity: "high", expression: /\bxox[baprs]-[A-Za-z0-9-]{20,}\b/ },
  { type: "google_api_key", severity: "high", expression: /\bAIza[0-9A-Za-z_-]{35}\b/ },
  { type: "stripe_live_secret", severity: "critical", expression: /\b(?:sk|rk)_live_[A-Za-z0-9]{16,}\b/ },
  { type: "openai_api_key", severity: "high", expression: /\bsk-(?:proj-)?[A-Za-z0-9_-]{24,}\b/ },
  { type: "aws_access_key_id", severity: "medium", expression: /\b(?:AKIA|ASIA)[A-Z0-9]{16}\b/ },
  { type: "bearer_token", severity: "high", expression: /\bBearer\s+[A-Za-z0-9._~-]{20,}/i },
  { type: "jwt", severity: "high", expression: /\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b/ },
  { type: "database_url_credentials", severity: "high", expression: /\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?):\/\/[^\s/:@]+:[^\s/@]{8,}@/i },
  {
    type: "generic_secret_assignment",
    severity: "high",
    expression: /\b(?:password|passwd|secret|api[_-]?key|access[_-]?token|refresh[_-]?token|auth[_-]?token|client[_-]?secret|private[_-]?key|aws_secret_access_key)\b\s*[:=]\s*(?:"(?!your\b|change[ _-]?me\b|changeme\b|example\b|placeholder\b|process\.env\b|env\(|\$\{|<)[^"\r\n]{12,}"|'(?!your\b|change[ _-]?me\b|changeme\b|example\b|placeholder\b|process\.env\b|env\(|\$\{|<)[^'\r\n]{12,}'|(?!your\b|change[ _-]?me\b|changeme\b|example\b|placeholder\b|process\.env\b|env\(|\$\{|<|%[A-Z_]+%)[A-Za-z0-9_./+=-]{16,})/i
  }
];

export async function scanWorkspaceSecrets(inputRoot) {
  const root = await resolveWorkspace(inputRoot);
  const inventory = await listWorkspaceTextFiles(root);
  const findings = [];
  const typeCounts = {};
  const severityCounts = {};
  let scannedFiles = 0;
  let skippedFiles = 0;
  let unreadableFiles = 0;
  let scannedBytes = 0;
  let findingCount = 0;
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
    const lines = source.content.split(/\r?\n/);
    for (let lineIndex = 0; lineIndex < lines.length; lineIndex += 1) {
      const line = lines[lineIndex];
      for (const rule of SECRET_RULES) {
        if (!rule.expression.test(line)) continue;
        findingCount += 1;
        typeCounts[rule.type] = (typeCounts[rule.type] || 0) + 1;
        severityCounts[rule.severity] = (severityCounts[rule.severity] || 0) + 1;
        if (findings.length < MAX_RETURNED_FINDINGS) {
          findings.push({ path: source.path, line: lineIndex + 1, type: rule.type, severity: rule.severity });
        } else {
          truncated = true;
        }
      }
    }
  }

  return {
    root, scannedFiles, skippedFiles, unreadableFiles, scannedBytes,
    findingCount, returnedFindings: findings.length, findings, typeCounts,
    severityCounts, truncated,
    limits: { maxFiles: MAX_SCAN_FILES, maxBytes: MAX_SCAN_BYTES, maxReturnedFindings: MAX_RETURNED_FINDINGS },
    notice: "Potential matches only. Secret values and matching source lines are never returned or logged. Review findings locally; this scanner does not prove that a workspace is secret-free."
  };
}

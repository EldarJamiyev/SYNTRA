import dns from "node:dns/promises";
import net from "node:net";
import { spawn } from "node:child_process";

const DNS_TIMEOUT_MS = 3000;
const TCP_TIMEOUT_MS = 3000;
const PING_TIMEOUT_MS = 6500;
const MAX_PING_OUTPUT = 8000;
const MAX_HOST_LENGTH = 253;

function invalid(message) {
  return { kind: "invalid", message };
}

function cleanHost(raw, allowSingleLabel = false) {
  if (typeof raw !== "string") return null;
  let value = raw.trim();
  if (!value || value.length > 2048) return null;

  if (/^https?:\/\//i.test(value)) {
    try {
      const parsed = new URL(value);
      if (parsed.username || parsed.password) return null;
      value = parsed.hostname;
    } catch {
      return null;
    }
  } else {
    value = value.split(/[/?#]/, 1)[0];
  }

  if (value.startsWith("[") && value.endsWith("]")) value = value.slice(1, -1);
  value = value.replace(/[),;!?]+$/g, "").replace(/\.$/, "").toLowerCase();
  if (!value) return null;
  if (net.isIP(value)) return value;
  if (value === "localhost") return allowSingleLabel ? value : null;
  if (value.length > MAX_HOST_LENGTH || value.includes(":")) return null;
  if (!allowSingleLabel && !value.includes(".")) return null;

  const labels = value.split(".");
  if (labels.some(label => !label || label.length > 63 || !/^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$/i.test(label))) return null;
  if (labels.length === 4 && labels.every(label => /^\d+$/.test(label))) return null;
  return value;
}

function parsePort(raw) {
  if (typeof raw !== "string" || !/^\d{1,5}$/.test(raw)) return null;
  const value = Number(raw);
  return Number.isInteger(value) && value >= 1 && value <= 65535 ? value : null;
}

function parseHostPort(raw, explicit = false) {
  if (typeof raw !== "string" || !raw.trim()) return null;
  let hostToken = raw.trim();
  let portToken = "";
  const ipv6 = hostToken.match(/^\[([^\]]+)\]:(\d+)$/);
  if (ipv6) {
    hostToken = ipv6[1];
    portToken = ipv6[2];
  } else {
    const pair = hostToken.match(/^(.+):(\d+)$/);
    if (pair) {
      hostToken = pair[1];
      portToken = pair[2];
    }
  }
  const host = cleanHost(hostToken, explicit);
  const port = parsePort(portToken);
  return host && port ? { kind: "port", host, port } : null;
}

function parseExplicit(message) {
  const match = message.match(/^\/(ping|dns|port)(?:\s+(.*))?$/i);
  if (!match) return null;
  const command = match[1].toLowerCase();
  const argument = (match[2] || "").trim();
  if (!argument) {
    return invalid({
      ping: "Usage: /ping example.com",
      dns: "Usage: /dns example.com",
      port: "Usage: /port example.com 443"
    }[command]);
  }

  if (command === "ping" || command === "dns") {
    if (/\s/.test(argument)) return invalid("Use one host only. Example: /" + command + " example.com");
    const host = cleanHost(argument, true);
    return host ? { kind: command, host } : invalid("That host name or IP address is not valid.");
  }

  const parts = argument.split(/\s+/);
  if (parts.length === 2) {
    const host = cleanHost(parts[0], true);
    const port = parsePort(parts[1]);
    return host && port ? { kind: "port", host, port } : invalid("Use a valid host and TCP port from 1 to 65535. Example: /port example.com 443");
  }
  return parseHostPort(argument, true)
    || invalid("Use a valid host and TCP port from 1 to 65535. Example: /port example.com 443");
}

/** Route only unambiguous, bounded local network-check requests. */
export function classifyNetworkDiagnostic(message) {
  if (typeof message !== "string") return null;
  const text = message.trim();
  if (!text || text.length > 2000) return null;

  const explicit = parseExplicit(text);
  if (explicit) return explicit;

  const natural = text
    .replace(/^(?:(?:hey|please|can you|could you|would you|will you)\s+)+/i, "")
    .replace(/[.!?]+$/g, "")
    .trim();

  let match = natural.match(/^(?:(?:run)\s+)*ping(?:\s+(?:test|check))?(?:\s+(?:for|to))?\s+(\S+)$/i);
  if (match) {
    const host = cleanHost(match[1]);
    return host ? { kind: "ping", host } : null;
  }

  const dnsPatterns = [
    /^(?:please\s+)?(?:dns(?:\s+(?:lookup|check|resolve))?|nslookup|resolve|lookup)(?:\s+(?:for|on))?\s+(\S+)$/i,
    /^(?:please\s+)?(?:check|verify)\s+dns(?:\s+(?:for|of))?\s+(\S+)$/i,
    /^(?:what\s+is|what's)\s+the\s+ip(?:\s+address)?\s+(?:of|for)\s+(\S+)$/i
  ];
  for (const pattern of dnsPatterns) {
    match = natural.match(pattern);
    if (!match) continue;
    const host = cleanHost(match[1]);
    return host ? { kind: "dns", host } : null;
  }

  match = natural.match(/^(?:(?:check|test|probe)\s+)(?:the\s+)?(?:tcp\s+)?port\s+(\d+)\s+(?:on|at|for)\s+(\S+)$/i);
  if (match) {
    const host = cleanHost(match[2]);
    const port = parsePort(match[1]);
    return host && port ? { kind: "port", host, port } : invalid("Use a valid host and TCP port from 1 to 65535.");
  }

  match = natural.match(/^(?:is|check if|test if)\s+(\S+)\s+(?:tcp\s+)?port\s+(\d+)\s+(?:open|reachable)\??$/i);
  if (match) {
    const host = cleanHost(match[1]);
    const port = parsePort(match[2]);
    return host && port ? { kind: "port", host, port } : invalid("Use a valid host and TCP port from 1 to 65535.");
  }

  match = natural.match(/^port ping\s+(\S+)$/i);
  if (match) return parseHostPort(match[1]) || invalid("Use a host and port, for example: port ping example.com:443");

  match = natural.match(/^test-netconnection\s+(?:-computername\s+)?(\S+)\s+-port\s+(\d+)$/i);
  if (match) {
    const host = cleanHost(match[1]);
    const port = parsePort(match[2]);
    return host && port ? { kind: "port", host, port } : invalid("Use a valid host and TCP port from 1 to 65535.");
  }

  return null;
}

function withTimeout(promise, milliseconds, message) {
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => {
      const error = new Error(message);
      error.code = "ETIMEDOUT";
      reject(error);
    }, milliseconds);
  });
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
}

async function runDns(task) {
  try {
    const addresses = await withTimeout(
      dns.lookup(task.host, { all: true, verbatim: true }),
      DNS_TIMEOUT_MS,
      "DNS resolution timed out."
    );
    const unique = [...new Map(addresses.map(item => [item.family + ":" + item.address, item])).values()];
    return {
      kind: "dns", host: task.host, status: unique.length ? "resolved" : "not-found",
      addresses: unique, message: unique.length ? "" : "The resolver returned no addresses."
    };
  } catch (error) {
    const code = error && typeof error === "object" && "code" in error ? String(error.code) : "";
    return {
      kind: "dns", host: task.host,
      status: code === "ETIMEDOUT" || code === "EAI_AGAIN" ? "timeout" : "failed",
      addresses: [],
      message: (code ? code + ": " : "") + (error instanceof Error ? error.message : String(error))
    };
  }
}

async function runTcp(task, { tcpConnectImpl = net.createConnection, tcpTimeoutMs = TCP_TIMEOUT_MS } = {}) {
  return new Promise(resolve => {
    const socket = tcpConnectImpl({ host: task.host, port: task.port });
    let settled = false;
    let deadlineTimer;
    const finish = (status, message = "") => {
      if (settled) return;
      settled = true;
      clearTimeout(deadlineTimer);
      const remoteAddress = socket.remoteAddress || "";
      socket.destroy();
      resolve({ kind: "port", host: task.host, port: task.port, status, message, remoteAddress });
    };
    deadlineTimer = setTimeout(
      () => finish("timeout", "TCP connection timed out after " + Math.max(1, Math.round(tcpTimeoutMs / 1000)) + " seconds."),
      tcpTimeoutMs
    );
    socket.setTimeout(tcpTimeoutMs, () => finish("timeout", "TCP connection timed out after " + Math.max(1, Math.round(tcpTimeoutMs / 1000)) + " seconds."));
    socket.once("connect", () => finish("open", "TCP connection established."));
    socket.once("error", error => {
      const code = error && typeof error === "object" && "code" in error ? String(error.code) : "";
      const status = code === "ECONNREFUSED" ? "closed"
        : ["ETIMEDOUT", "EHOSTUNREACH", "ENETUNREACH", "EAI_AGAIN"].includes(code) ? "timeout" : "failed";
      finish(status, (code ? code + ": " : "") + error.message);
    });
  });
}

async function runPing(task) {
  const executable = process.platform === "win32" ? "ping.exe" : "ping";
  const args = process.platform === "win32"
    ? ["-n", "4", "-w", "1000", task.host]
    : process.platform === "darwin"
      ? ["-c", "4", "-W", "1000", task.host]
      : ["-c", "4", "-W", "1", task.host];

  return new Promise(resolve => {
    let child;
    let settled = false;
    let timedOut = false;
    let output = "";
    let killTimer;
    let timeoutTimer;
    const finish = (status, exitCode = null, message = "") => {
      if (settled) return;
      settled = true;
      clearTimeout(timeoutTimer);
      clearTimeout(killTimer);
      resolve({ kind: "ping", host: task.host, status, exitCode, output: output.trim(), message });
    };
    try {
      child = spawn(executable, args, { shell: false, windowsHide: true, stdio: ["ignore", "pipe", "pipe"] });
    } catch (error) {
      finish("unavailable", null, error instanceof Error ? error.message : String(error));
      return;
    }
    const collect = chunk => {
      if (output.length < MAX_PING_OUTPUT) output += chunk.toString("utf8").slice(0, MAX_PING_OUTPUT - output.length);
    };
    child.stdout.on("data", collect);
    child.stderr.on("data", collect);
    child.once("error", error => finish(error && error.code === "ENOENT" ? "unavailable" : "failed", null, error instanceof Error ? error.message : String(error)));
    timeoutTimer = setTimeout(() => {
      timedOut = true;
      child.kill();
      killTimer = setTimeout(() => finish("timeout", null, "Ping timed out after 6.5 seconds."), 500);
    }, PING_TIMEOUT_MS);
    child.once("close", code => finish(timedOut ? "timeout" : code === 0 ? "reachable" : "unreachable", code, timedOut ? "Ping timed out after 6.5 seconds." : ""));
  });
}

export async function runNetworkDiagnostic(task, options = {}) {
  if (!task || task.kind === "invalid") return { kind: "invalid", status: "invalid", message: task?.message || "Invalid network diagnostic request." };
  const host = cleanHost(task.host, true);
  if (!host) return { kind: "invalid", status: "invalid", message: "That host name or IP address is not valid." };
  if (task.kind === "dns") return runDns({ kind: "dns", host });
  if (task.kind === "ping") return runPing({ kind: "ping", host });
  if (task.kind === "port") {
    const port = parsePort(String(task.port));
    if (!port) return { kind: "invalid", status: "invalid", message: "TCP port must be an integer from 1 to 65535." };
    return runTcp({ kind: "port", host, port }, options);
  }
  return { kind: "invalid", status: "invalid", message: "Unknown diagnostic type." };
}

const STATUS_LABELS = {
  resolved: "Resolved", "not-found": "No address found", reachable: "Replies received",
  unreachable: "No ping replies", open: "Open", closed: "Closed / connection refused",
  timeout: "Timed out", failed: "Failed", unavailable: "Unavailable", invalid: "Invalid request"
};

export function formatNetworkDiagnostic(result) {
  if (!result || result.kind === "invalid") return "**Network diagnostic**\n" + String(result?.message || "That request could not be parsed.");
  const lines = [];
  if (result.kind === "ping") {
    lines.push("**Ping: " + result.host + "**");
    lines.push("**Result:** " + (STATUS_LABELS[result.status] || result.status));
    if (result.exitCode !== null) lines.push("Exit code: " + result.exitCode);
    if (result.message) lines.push(result.message);
    if (result.output) {
      const fence = String.fromCharCode(96).repeat(3);
      lines.push("", "Ping output:", fence + "text", result.output, fence);
    }
  } else if (result.kind === "dns") {
    lines.push("**DNS lookup: " + result.host + "**");
    lines.push("**Result:** " + (STATUS_LABELS[result.status] || result.status));
    if (result.addresses?.length) {
      lines.push("", "Addresses:");
      for (const item of result.addresses) lines.push("- " + item.address + " (IPv" + item.family + ")");
    }
    if (result.message) lines.push("", result.message);
  } else if (result.kind === "port") {
    lines.push("**TCP port check: " + result.host + ":" + result.port + "**");
    lines.push("**Result:** " + (STATUS_LABELS[result.status] || result.status));
    if (result.remoteAddress) lines.push("Remote address: " + result.remoteAddress);
    if (result.message) lines.push(result.message);
  } else {
    return "**Network diagnostic**\n" + String(result.message || "Unknown diagnostic result.");
  }
  lines.push("", "_Executed locally by SYNTRA; a failed connection does not by itself prove the service is down._");
  return lines.join("\n");
}

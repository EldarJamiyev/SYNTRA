import {
  CHAT_HISTORY_KEY,
  loadMessages,
  normalizeMessages
} from "./chat-history.js";

export const CHAT_SESSIONS_KEY = "syntra-chat-sessions-v1";
export const MAX_CHAT_SESSIONS = 24;
export const MAX_SESSION_MESSAGES = 60;
export const MAX_SESSION_TEXT_CHARS = 280_000;

function resolveStorage(storage) {
  if (storage) return storage;
  try {
    return globalThis.localStorage || null;
  } catch {
    return null;
  }
}

function createId() {
  try {
    if (typeof globalThis.crypto?.randomUUID === "function") return globalThis.crypto.randomUUID();
  } catch {}
  return "session-" + Date.now().toString(36) + "-" + Math.random().toString(36).slice(2, 12);
}

function validId(value) {
  return typeof value === "string" && /^[a-zA-Z0-9_-]{1,120}$/.test(value);
}

function validTimestamp(value, fallback) {
  if (typeof value !== "string" || !Number.isFinite(Date.parse(value))) return fallback;
  return value;
}

export function titleFromMessages(messages) {
  const firstPrompt = normalizeMessages(messages).find(message => message.role === "user")?.content;
  if (!firstPrompt) return "New chat";
  const compact = firstPrompt.replace(/\s+/g, " ").trim();
  return compact.length > 52 ? compact.slice(0, 51).trimEnd() + "…" : compact;
}

export function createChatSession(messages = [], options = {}) {
  const now = typeof options.now === "string" ? options.now : new Date().toISOString();
  const normalizedMessages = normalizeMessages(messages).slice(-MAX_SESSION_MESSAGES);
  const requestedTitle = typeof options.title === "string" ? options.title.trim() : "";
  return {
    id: validId(options.id) ? options.id : createId(),
    title: requestedTitle.slice(0, 80) || titleFromMessages(normalizedMessages),
    createdAt: now,
    updatedAt: now,
    messages: normalizedMessages
  };
}

export function normalizeSessionState(value) {
  const rawSessions = value && typeof value === "object" && !Array.isArray(value)
    && value.version === 1 && Array.isArray(value.sessions)
    ? value.sessions
    : [];
  const candidates = rawSessions
    .filter(item => item && typeof item === "object" && !Array.isArray(item) && validId(item.id))
    .map(item => ({
      id: item.id,
      createdAt: validTimestamp(item.createdAt, new Date(0).toISOString()),
      updatedAt: validTimestamp(item.updatedAt, validTimestamp(item.createdAt, new Date(0).toISOString())),
      title: typeof item.title === "string" && item.title.trim()
        ? item.title.trim().slice(0, 80)
        : titleFromMessages(item.messages),
      messages: normalizeMessages(item.messages)
    }))
    .sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));

  const sessions = [];
  const seen = new Set();
  let totalChars = 0;
  for (const candidate of candidates) {
    if (seen.has(candidate.id) || sessions.length >= MAX_CHAT_SESSIONS) continue;
    seen.add(candidate.id);
    const messages = [];
    for (let index = candidate.messages.length - 1; index >= 0; index -= 1) {
      if (messages.length >= MAX_SESSION_MESSAGES || totalChars >= MAX_SESSION_TEXT_CHARS) break;
      const message = candidate.messages[index];
      const available = MAX_SESSION_TEXT_CHARS - totalChars;
      const content = message.content.length > available
        ? message.content.slice(-available)
        : message.content;
      if (!content.trim()) continue;
      messages.unshift({ role: message.role, content });
      totalChars += content.length;
    }
    sessions.push({
      ...candidate,
      title: candidate.title === "New chat" ? titleFromMessages(messages) : candidate.title,
      messages
    });
    if (totalChars >= MAX_SESSION_TEXT_CHARS) break;
  }

  if (!sessions.length) {
    const initial = createChatSession();
    sessions.push(initial);
  }

  const requestedActive = value && typeof value.activeSessionId === "string"
    ? value.activeSessionId
    : "";
  const activeSessionId = sessions.some(session => session.id === requestedActive)
    ? requestedActive
    : sessions[0].id;

  return { version: 1, activeSessionId, sessions };
}

export function parseStoredSessionState(serialized) {
  if (typeof serialized !== "string" || !serialized) return null;
  try {
    const parsed = JSON.parse(serialized);
    if (!parsed || parsed.version !== 1 || !Array.isArray(parsed.sessions)) return null;
    return normalizeSessionState(parsed);
  } catch {
    return null;
  }
}

export function serializeSessionState(state) {
  return JSON.stringify(normalizeSessionState(state));
}

export function loadSessionState(storage) {
  const target = resolveStorage(storage);
  if (!target) return normalizeSessionState(null);

  try {
    const stored = parseStoredSessionState(target.getItem(CHAT_SESSIONS_KEY));
    if (stored) return stored;
  } catch {}

  // One-time migration from the original single-thread history key.
  const legacyMessages = loadMessages(target);
  const migrated = createChatSession(legacyMessages);
  return normalizeSessionState({
    version: 1,
    activeSessionId: migrated.id,
    sessions: [migrated]
  });
}

export function saveSessionState(state, storage) {
  const target = resolveStorage(storage);
  if (!target) return false;
  try {
    target.setItem(CHAT_SESSIONS_KEY, serializeSessionState(state));
    return true;
  } catch {
    // Storage quota and privacy failures must never block local chat.
    return false;
  }
}

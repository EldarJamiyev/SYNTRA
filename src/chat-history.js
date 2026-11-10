export const CHAT_HISTORY_KEY = "syntra-chat-history-v1";

const MAX_MESSAGES = 60;
const MAX_MESSAGE_CHARS = 24_000;
const MAX_TOTAL_CHARS = 120_000;

function resolveStorage(storage) {
  if (storage) return storage;
  try {
    return globalThis.localStorage || null;
  } catch {
    return null;
  }
}

export function normalizeMessages(value) {
  if (!Array.isArray(value)) return [];
  const recent = [];
  let totalChars = 0;

  for (let index = value.length - 1; index >= 0 && recent.length < MAX_MESSAGES; index -= 1) {
    const item = value[index];
    if (!item || (item.role !== "user" && item.role !== "assistant") || typeof item.content !== "string") continue;
    if (!item.content.trim()) continue;

    let content = item.content.slice(-MAX_MESSAGE_CHARS);
    const remaining = MAX_TOTAL_CHARS - totalChars;
    if (remaining <= 0) break;
    if (content.length > remaining) content = content.slice(-remaining);
    if (!content) break;

    recent.push({ role: item.role, content });
    totalChars += content.length;
  }

  return recent.reverse();
}

export function parseStoredMessages(serialized) {
  if (typeof serialized !== "string" || !serialized) return [];
  try {
    return normalizeMessages(JSON.parse(serialized));
  } catch {
    return [];
  }
}

export function serializeMessages(messages) {
  return JSON.stringify(normalizeMessages(messages));
}

export function loadMessages(storage) {
  const target = resolveStorage(storage);
  if (!target) return [];
  try {
    return parseStoredMessages(target.getItem(CHAT_HISTORY_KEY));
  } catch {
    return [];
  }
}

export function saveMessages(messages, storage) {
  const target = resolveStorage(storage);
  if (!target) return false;
  try {
    target.setItem(CHAT_HISTORY_KEY, serializeMessages(messages));
    return true;
  } catch {
    // Quota/privacy errors are non-fatal: chat must remain usable without persistence.
    return false;
  }
}

import assert from "node:assert/strict";
import test from "node:test";
import {
  CHAT_SESSIONS_KEY,
  MAX_CHAT_SESSIONS,
  MAX_SESSION_TEXT_CHARS,
  createChatSession,
  loadSessionState,
  normalizeSessionState,
  parseStoredSessionState,
  saveSessionState,
  serializeSessionState,
  titleFromMessages
} from "../src/chat-sessions.js";

function memoryStorage(seed = {}) {
  const values = new Map(Object.entries(seed));
  return {
    getItem(key) { return values.get(key) ?? null; },
    setItem(key, value) { values.set(key, value); },
    values
  };
}

test("sessions persist independently and restore the selected conversation", () => {
  const storage = memoryStorage();
  const first = createChatSession([
    { role: "user", content: "Inspect the API" },
    { role: "assistant", content: "I found the route." }
  ], { id: "first-session", now: "2026-10-30T10:00:00.000Z" });
  const second = createChatSession([
    { role: "user", content: "Explain the runtime" }
  ], { id: "second-session", now: "2026-10-30T11:00:00.000Z" });
  const state = {
    version: 1,
    activeSessionId: first.id,
    sessions: [second, first]
  };

  assert.equal(saveSessionState(state, storage), true);
  const loaded = loadSessionState(storage);
  assert.equal(loaded.activeSessionId, first.id);
  assert.equal(loaded.sessions.length, 2);
  assert.deepEqual(loaded.sessions.find(item => item.id === first.id)?.messages, first.messages);
  assert.equal(loaded.sessions[0].id, second.id, "recently updated sessions should be listed first");
});

test("the previous single-thread history is migrated without losing messages", () => {
  const storage = memoryStorage({
    "syntra-chat-history-v1": JSON.stringify([
      { role: "user", content: "Old prompt" },
      { role: "assistant", content: "Old answer" }
    ])
  });
  const loaded = loadSessionState(storage);
  assert.equal(loaded.sessions.length, 1);
  assert.match(loaded.sessions[0].id, /^[a-zA-Z0-9_-]+$/);
  assert.deepEqual(loaded.sessions[0].messages, [
    { role: "user", content: "Old prompt" },
    { role: "assistant", content: "Old answer" }
  ]);
  assert.equal(loaded.sessions[0].title, "Old prompt");
});

test("malformed session data falls back safely and does not expose invalid roles", () => {
  assert.equal(parseStoredSessionState("{broken"), null);
  const normalized = normalizeSessionState({
    version: 1,
    activeSessionId: "missing",
    sessions: [
      { id: "../bad", title: "unsafe", messages: [{ role: "user", content: "x" }] },
      { id: "valid", title: "Valid", messages: [{ role: "tool", content: "hidden" }] }
    ]
  });
  assert.equal(normalized.activeSessionId, "valid");
  assert.deepEqual(normalized.sessions[0].messages, []);
});

test("session collections and total stored text remain bounded", () => {
  const sessions = Array.from({ length: MAX_CHAT_SESSIONS + 10 }, (_, index) => ({
    id: "session-" + index,
    title: "Session " + index,
    createdAt: "2026-10-30T10:00:00.000Z",
    updatedAt: new Date(Date.UTC(2026, 9, 30, 10, index)).toISOString(),
    messages: [{ role: "user", content: "x".repeat(5_000) }]
  }));
  const state = normalizeSessionState({ version: 1, activeSessionId: "session-0", sessions });
  const totalChars = state.sessions.reduce((sum, session) =>
    sum + session.messages.reduce((sub, message) => sub + message.content.length, 0), 0);
  assert.equal(state.sessions.length, MAX_CHAT_SESSIONS);
  assert.ok(totalChars <= MAX_SESSION_TEXT_CHARS);
  assert.ok(serializeSessionState(state).length < 600_000);

  const oversizedSessions = sessions.map(session => ({
    ...session,
    messages: [{ role: "user", content: "x".repeat(20_000) }]
  }));
  const bounded = normalizeSessionState({
    version: 1, activeSessionId: "session-0", sessions: oversizedSessions
  });
  const boundedChars = bounded.sessions.reduce((sum, session) =>
    sum + session.messages.reduce((sub, message) => sub + message.content.length, 0), 0);
  assert.ok(bounded.sessions.length < MAX_CHAT_SESSIONS, "the global text budget may prune sessions before the session-count limit");
  assert.ok(boundedChars <= MAX_SESSION_TEXT_CHARS);
});

test("automatic titles are based on the first user prompt and remain compact", () => {
  assert.equal(titleFromMessages([{ role: "user", content: "Check DNS records" }]), "Check DNS records");
  assert.equal(titleFromMessages([{ role: "assistant", content: "Hello" }]), "New chat");
  assert.ok(titleFromMessages([{ role: "user", content: "a".repeat(100) }]).length <= 52);
});

test("storage failures are non-fatal for session loading and saving", () => {
  const blocked = {
    getItem() { throw new Error("storage disabled"); },
    setItem() { throw new Error("quota exceeded"); }
  };
  assert.equal(loadSessionState(blocked).sessions.length, 1);
  assert.equal(saveSessionState(normalizeSessionState(null), blocked), false);
});

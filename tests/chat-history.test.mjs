import assert from "node:assert/strict";
import test from "node:test";
import {
  CHAT_HISTORY_KEY,
  loadMessages,
  normalizeMessages,
  parseStoredMessages,
  saveMessages,
  serializeMessages
} from "../src/chat-history.js";

function memoryStorage(seed = null) {
  let value = seed;
  return {
    getItem(key) {
      assert.equal(key, CHAT_HISTORY_KEY);
      return value;
    },
    setItem(key, nextValue) {
      assert.equal(key, CHAT_HISTORY_KEY);
      value = nextValue;
    },
    value() { return value; }
  };
}

test("conversation history round-trips and discards invalid entries", () => {
  const storage = memoryStorage();
  const original = [
    { role: "user", content: "Check DNS" },
    { role: "assistant", content: "Use nslookup." },
    { role: "system", content: "not a chat role" },
    { role: "user", content: "" },
    { role: "assistant", content: "Run Test-NetConnection next." }
  ];
  assert.equal(saveMessages(original, storage), true);
  assert.deepEqual(loadMessages(storage), [
    { role: "user", content: "Check DNS" },
    { role: "assistant", content: "Use nslookup." },
    { role: "assistant", content: "Run Test-NetConnection next." }
  ]);
});

test("malformed storage never prevents chat from starting", () => {
  assert.deepEqual(parseStoredMessages("{bad-json"), []);
  assert.deepEqual(loadMessages(memoryStorage("{bad-json")), []);
  assert.deepEqual(loadMessages({ getItem() { throw new Error("storage blocked"); }, setItem() {} }), []);
  assert.equal(saveMessages([], { getItem() { return null; }, setItem() { throw new Error("quota exceeded"); } }), false);
});

test("history storage is bounded by message count and total text", () => {
  const messages = Array.from({ length: 90 }, (_, index) => ({ role: index % 2 ? "assistant" : "user", content: "message-" + index }));
  const normalized = normalizeMessages(messages);
  assert.equal(normalized.length, 60);
  assert.equal(normalized[0].content, "message-30");
  assert.equal(normalized.at(-1).content, "message-89");

  const huge = normalizeMessages([{ role: "user", content: "x".repeat(200_000) }]);
  assert.equal(huge.length, 1);
  assert.equal(huge[0].content.length, 24_000);
  assert.ok(serializeMessages(Array.from({ length: 20 }, (_, i) => ({ role: "user", content: "z".repeat(10_000) }))).length < 250_000);
});

test("stored roles and content types are validated before they reach the UI", () => {
  assert.deepEqual(parseStoredMessages(JSON.stringify([
    null,
    { role: "assistant", content: 99 },
    { role: "tool", content: "should not show" },
    { role: "user", content: "safe text" }
  ])), [{ role: "user", content: "safe text" }]);
});

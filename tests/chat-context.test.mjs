import assert from "node:assert/strict";
import test from "node:test";
import { buildChatPromptContext } from "../local/chat-context.mjs";

const characterTokenizer = async text => ({ token_count: text.length });

test("preserves the complete current user message and complete retained history", async () => {
  const message = "Preserve the public API, do not remove the migration path, and keep this final constraint: END-OF-REQUEST.";
  const history = [
    { role: "user", content: "Earlier requirement: keep existing configuration compatible." },
    { role: "assistant", content: "I will preserve the existing configuration behavior." }
  ];

  const result = await buildChatPromptContext({
    message,
    history,
    contextTokens: 1024,
    tokenize: characterTokenizer
  });

  assert.equal(result.fits, true);
  assert.ok(result.prompt.endsWith("USER: " + message + "\n\nASSISTANT:"));
  assert.ok(result.prompt.includes("USER: " + history[0].content));
  assert.ok(result.prompt.includes("ASSISTANT: " + history[1].content));
  assert.equal(result.historyMessagesUsed, 2);
  assert.equal(result.historyMessagesDropped, 0);
});

test("drops optional workspace evidence before removing conversation history", async () => {
  const history = [
    { role: "user", content: "The original requirement is to retain compatibility." },
    { role: "assistant", content: "The compatibility requirement is still active." }
  ];
  const result = await buildChatPromptContext({
    message: "Fix the API.",
    history,
    workspaceGrounding: "WORKSPACE EVIDENCE: " + "repository context ".repeat(5),
    contextTokens: 240,
    tokenize: characterTokenizer
  });

  assert.equal(result.fits, true);
  assert.equal(result.workspaceGroundingIncluded, false);
  assert.equal(result.historyMessagesUsed, 2);
  assert.equal(result.historyMessagesDropped, 0);
  assert.ok(result.prompt.includes("The original requirement is to retain compatibility."));
});

test("removes complete oldest history messages while retaining the newest context", async () => {
  const oldest = "Older context must be removed as one complete message. " + "A".repeat(28);
  const newest = "Newest context must remain complete. " + "B".repeat(28);
  const result = await buildChatPromptContext({
    message: "Continue the existing task.",
    history: [
      { role: "user", content: oldest },
      { role: "assistant", content: newest }
    ],
    contextTokens: 200,
    tokenize: characterTokenizer
  });

  assert.equal(result.fits, true);
  assert.equal(result.historyMessagesUsed, 1);
  assert.equal(result.historyMessagesDropped, 1);
  assert.ok(!result.prompt.includes(oldest));
  assert.ok(result.prompt.includes(newest));
});

test("rejects an over-budget current message without truncating or rewriting it", async () => {
  const message = "BEGIN-USER-REQUEST " + "preserve-this-content ".repeat(40) + "END-USER-REQUEST";
  const observedPrompts = [];
  const result = await buildChatPromptContext({
    message,
    history: [{ role: "user", content: "Old history that must be dropped." }],
    workspaceGrounding: "Optional workspace evidence.",
    contextTokens: 128,
    tokenize: async prompt => {
      observedPrompts.push(prompt);
      return { token_count: prompt.length };
    }
  });

  assert.equal(result.fits, false);
  assert.equal(result.historyMessagesUsed, 0);
  assert.equal(result.historyMessagesDropped, 1);
  assert.equal(result.workspaceGroundingIncluded, false);
  assert.ok(observedPrompts.at(-1).includes(message));
  assert.ok(observedPrompts.at(-1).includes("BEGIN-USER-REQUEST"));
  assert.ok(observedPrompts.at(-1).includes("END-USER-REQUEST"));
  assert.ok(!observedPrompts.at(-1).includes("Old history"));
  assert.equal(Object.hasOwn(result, "prompt"), false);
});

test("drops complete oversized history entries instead of clipping their content", async () => {
  const oversized = "H".repeat(17_000);
  const retained = "The recent short requirement remains intact.";
  const result = await buildChatPromptContext({
    message: "Continue.",
    history: [
      { role: "user", content: oversized },
      { role: "assistant", content: retained }
    ],
    contextTokens: 1024,
    tokenize: characterTokenizer
  });

  assert.equal(result.fits, true);
  assert.ok(!result.prompt.includes(oversized));
  assert.ok(result.prompt.includes(retained));
  assert.equal(result.historyMessagesDropped, 1);
});

test("fails closed when the native tokenizer returns an invalid token count", async () => {
  await assert.rejects(
    buildChatPromptContext({
      message: "Check this request.",
      contextTokens: 1024,
      tokenize: async () => ({ token_count: "unknown" })
    }),
    /invalid token_count/
  );
});

test("validates the message, context window, and tokenizer dependency", async () => {
  await assert.rejects(
    buildChatPromptContext({ message: " ", contextTokens: 1024, tokenize: characterTokenizer }),
    /non-empty current user message/
  );
  await assert.rejects(
    buildChatPromptContext({ message: "hello", contextTokens: 16, tokenize: characterTokenizer }),
    /at least 32 tokens/
  );
  await assert.rejects(
    buildChatPromptContext({ message: "hello", contextTokens: 1024 }),
    /tokenizer callback is required/
  );
});

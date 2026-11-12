import assert from "node:assert/strict";
import test from "node:test";
import { buildFollowUpTaskGoal, classifyChatIntent, extractTargetPaths, extractTargetSymbols, isCodingTask } from "../local/chat-intent.mjs";
import { normalizeIntentText } from "../local/intent-language.mjs";

test("routes explicit software changes to the coding agent", () => {
  for (const message of [
    "Fix this bug in the login form",
    "Please add dark mode to the chat UI",
    "Build a React website with a responsive layout",
    "Can you refactor the API endpoint?",
    "Improve the app performance",
    "Write a Python script that checks DNS records",
    "Update the project files to repair the build error",
    "Please modify this repository",
    "Create a new component for the message composer",
    "Create a new page for my website",
    "Add a settings screen to the mobile app",
    "Test the endpoint and fix the failing unit test",
    "Could you please build a React app?",
    "Can you help me fix the API endpoint?",
    "In local/agent.mjs, update rankGraph to boost test files",
  ]) assert.equal(isCodingTask(message), true, message);
});

test("keeps ordinary writing, questions, and conversational requests in chat", () => {
  for (const message of [
    "Write an email to my manager",
    "Write a job application for a support role",
    "Write an email about fixing this app",
    "Create a poem about a broken website",
    "Write a page about the ocean",
    "Create a page for my notes",
    "Make a layout of my bedroom",
    "Make a screen recording checklist",
    "Write a poem about the sea",
    "Create a packing list for my trip",
    "Make me laugh",
    "Build me a weekly study schedule",
    "Explain what a DNS server does",
    "What is the difference between RAM and storage?",
    "Can you make this answer shorter?",
    "Tell me a story about a dragon",
    "Give me ideas for a project",
    "Fix this",
    "",
    "   ",
    null,
    42,
  ]) assert.equal(isCodingTask(message), false, String(message));
});

test("keeps the command classifier bounded and tolerant of casing/spacing", () => {
  assert.equal(isCodingTask("  PLEASE FIX the React component  "), true);
  assert.equal(isCodingTask("Write a poem. ".repeat(1200)), false);
});

test("classifyChatIntent distinguishes coding tasks, workspace queries, and general conversation", () => {
  const coding = classifyChatIntent("Refactor `rankGraph` in local/agent.mjs to weight symbol matches");
  assert.equal(coding.kind, "coding_task");
  assert.deepEqual(coding.targetPaths, ["local/agent.mjs"]);
  assert.ok(coding.targetSymbols.includes("rankGraph"));

  const query = classifyChatIntent("How does `ByteBPETokenizer` work in syntra_ai/tokenizer.py?");
  assert.equal(query.kind, "workspace_query");
  assert.deepEqual(query.targetPaths, ["syntra_ai/tokenizer.py"]);
  assert.ok(query.targetSymbols.includes("ByteBPETokenizer"));
  assert.equal(query.searchQuery, "ByteBPETokenizer");

  const convo = classifyChatIntent("Write a poem about the sea");
  assert.equal(convo.kind, "conversation");
  assert.deepEqual(extractTargetPaths("Check src/App.tsx and local/server.mjs"), ["src/App.tsx", "local/server.mjs"]);
  assert.ok(extractTargetSymbols("Inspect `validatePatch` and `search_workspace_memory`").includes("validatePatch"));
  assert.deepEqual(
    extractTargetSymbols("Inspect `BOS_ID`, `ByteBPETokenizer`, validatePatch, search_workspace_memory"),
    ["BOS_ID", "ByteBPETokenizer", "validatePatch", "search_workspace_memory"]
  );
  const longIdentifierText = "a" + "A".repeat(1800) + "a".repeat(1800) + "!";
  assert.deepEqual(extractTargetSymbols(longIdentifierText), [],
    "overlong user text must not become an unbounded symbol candidate");
});



test("normalizes shorthand and known typos only for routing", () => {
  for (const message of [
    "pls fix the login form",
    "could u plz fix the api endpiont?",
    "hey can u upgarde this repo chat featre",
    "please add dark mode to the chatt ui",
    "can you improove the app performnce",
    "duzelt chat ui",
    "исправь код",
    "lütfen website güncelle",
  ]) {
    assert.equal(classifyChatIntent(message).kind, "coding_task", message);
  }
  assert.equal(normalizeIntentText("that this what"), "that this what");
  assert.equal(normalizeIntentText("Please, could you?"), "please, could you?");
});

test("does not classify ordinary acronyms as source symbols or workspace queries", () => {
  assert.equal(classifyChatIntent("What is DNS?").kind, "conversation");
  assert.equal(classifyChatIntent("Explain what a DNS server does").kind, "conversation");
  assert.deepEqual(extractTargetSymbols("What is DNS?"), []);
  assert.ok(extractTargetSymbols("Inspect ByteBPETokenizer and validatePatch").includes("ByteBPETokenizer"));
  assert.ok(extractTargetSymbols("Inspect `DNS` configuration").includes("DNS"),
    "explicitly quoted code identifiers remain eligible");
});

test("supports object-first informal software commands without broad natural-language false positives", () => {
  assert.equal(classifyChatIntent("chat ui duzelt").kind, "coding_task");
  assert.equal(classifyChatIntent("website please fix").kind, "coding_task");
  assert.equal(classifyChatIntent("I want you to fix the API endpoint").kind, "coding_task");
  assert.equal(classifyChatIntent("I want to improve my sleep schedule").kind, "conversation");
  assert.equal(classifyChatIntent("Tell me a story about improving a website").kind, "conversation");
});

test("resolves follow-up chains while preserving every previous instruction", () => {
  const original = "Fix the login bug in src/auth.ts and preserve the existing API behavior.";
  const firstFollowUp = "make it more robust";
  const history = [
    { role: "user", content: original },
    { role: "assistant", content: "I traced the failure to token refresh." },
    { role: "user", content: firstFollowUp },
    { role: "assistant", content: "The implementation is improved." }
  ];
  const intent = classifyChatIntent("also add tests", history);
  assert.equal(intent.kind, "coding_task");
  assert.equal(intent.contextualFollowUp, true);
  assert.deepEqual(intent.targetPaths, ["src/auth.ts"]);
  assert.deepEqual(intent.contextMessages, [original, firstFollowUp]);
  const goal = buildFollowUpTaskGoal("also add tests", intent);
  assert.ok(goal.includes(original));
  assert.ok(goal.includes(firstFollowUp));
  assert.ok(goal.includes("also add tests"));
  assert.ok(goal.includes("preserve the existing API behavior"));
});

test("keeps unrelated questions as context boundaries", () => {
  const history = [
    { role: "user", content: "Fix the login bug in src/auth.ts." },
    { role: "assistant", content: "The change is ready." },
    { role: "user", content: "What is DNS?" }
  ];
  assert.equal(classifyChatIntent("continue", history).kind, "conversation");
});

test("normalizes and routes short multilingual technical requests without changing source paths", () => {
  const cases = [
    ["zəhmət olmasa API-ni düzəlt", "coding_task"],
    ["пожалуйста, исправь код", "coding_task"],
    ["lütfen website güncelle", "coding_task"],
    ["duzelt chat ui in local/server.mjs", "coding_task"]
  ];
  for (const [message, expected] of cases) {
    assert.equal(classifyChatIntent(message).kind, expected, message);
  }
  const pathResult = classifyChatIntent("duzelt kod in local/server.mjs");
  assert.deepEqual(pathResult.targetPaths, ["local/server.mjs"]);
});

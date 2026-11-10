import assert from "node:assert/strict";
import test from "node:test";
import { classifyChatIntent, extractTargetPaths, extractTargetSymbols, isCodingTask } from "../local/chat-intent.mjs";

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


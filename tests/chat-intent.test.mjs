import assert from "node:assert/strict";
import test from "node:test";
import { isCodingTask } from "../local/chat-intent.mjs";

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
    "Test the endpoint and fix the failing unit test",
  ]) assert.equal(isCodingTask(message), true, message);
});

test("keeps ordinary writing, questions, and conversational requests in chat", () => {
  for (const message of [
    "Write an email to my manager",
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

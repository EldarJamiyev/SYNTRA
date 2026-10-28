import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import {
  buildRepositoryGraph,
  parsePatchOutput,
  runAgentTask,
  validateStructuredPatch
} from "../local/agent.mjs";

async function createProject() {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "syntra-agent-test-"));
  await fs.mkdir(path.join(root, "src"), { recursive: true });
  await fs.mkdir(path.join(root, "tests"), { recursive: true });
  await fs.writeFile(path.join(root, "package.json"), JSON.stringify({
    name: "agent-fixture",
    version: "1.0.0",
    scripts: { test: "node --test" }
  }, null, 2));
  await fs.writeFile(path.join(root, "src", "value.js"), "module.exports = { value: 1 };\n");
  await fs.writeFile(path.join(root, "tests", "value.test.js"),
    'const test = require("node:test");\nconst assert = require("node:assert/strict");\nconst value = require("../src/value.js");\ntest("value is two", () => assert.equal(value.value, 2));\n');
  return root;
}

test("structured patch parser accepts one JSON object and rejects truncated output", () => {
  const parsed = parsePatchOutput('preface {"summary":"update","edits":[{"path":"a.js","find":"x","replace":"y"}]} trailing text');
  assert.equal(parsed.summary, "update");
  assert.equal(parsed.edits.length, 1);
  assert.throws(() => parsePatchOutput('{"summary":"truncated"'), /ended before the JSON object was complete/);
});

test("patch validation enforces a unique exact anchor and blocks secret paths", async () => {
  const root = await createProject();
  try {
    const valid = await validateStructuredPatch(root, {
      summary: "Change exported value",
      edits: [{
        path: "src/value.js", find: "value: 1", replace: "value: 2", reason: "Match the failing test."
      }]
    });
    assert.equal(valid.changes[0].content, "module.exports = { value: 2 };\n");
    await assert.rejects(
      () => validateStructuredPatch(root, {
        summary: "Ambiguous change",
        edits: [{ path: "src/value.js", find: " ", replace: "  ", reason: "ambiguous" }]
      }),
      /must occur exactly once/
    );
    await assert.rejects(
      () => validateStructuredPatch(root, {
        summary: "Unsafe change",
        edits: [{ path: ".env", find: "", replace: "SECRET=value", reason: "unsafe" }]
      }),
      /secret|credential|Environment secret/i
    );
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("repository graph links local imports and agent repairs a failed test in a sandbox", async () => {
  const root = await createProject();
  let generateCount = 0;
  const modelRequest = async (method, route, body) => {
    if (method === "GET" && route === "/model/status") {
      return {
        checkpoint_exists: true, context_tokens: 4096, tokenizer: "byte-BPE",
        tokenizer_vocab_size: 258, config: { seq_len: 4096, vocab_size: 258 }
      };
    }
    if (route === "/model/tokenize") return { token_count: Math.ceil(body.text.length / 4) };
    if (route === "/model/generate") {
      generateCount += 1;
      if (generateCount === 1) return { text: '{"summary":"unfinished"', generated_tokens: 4, trained_steps: 1 };
      return {
        text: JSON.stringify({
          summary: "Change exported value so the failing test passes.",
          edits: [{
            path: "src/value.js",
            find: "value: 1",
            replace: "value: 2",
            reason: "The current test expects value 2."
          }]
        }),
        generated_tokens: 40, trained_steps: 10
      };
    }
    throw new Error("Unexpected model route: " + route);
  };
  try {
    const graph = await buildRepositoryGraph(root);
    const valueNode = graph.nodes.find(node => node.path === "src/value.js");
    const testNode = graph.nodes.find(node => node.path === "tests/value.test.js");
    assert.ok(valueNode);
    assert.ok(testNode);
    assert.ok(testNode.imports.includes("src/value.js"), "the graph should resolve relative imports");

    const result = await runAgentTask({
      root,
      goal: "Change the exported value from 1 to 2 so the test passes.",
      maxIterations: 3,
      runChecks: true,
      modelRequest
    });
    assert.equal(generateCount, 2, "the loop should retry after malformed patch JSON");
    assert.equal(result.status, "verified_proposals_ready");
    assert.equal(result.verification.status, "passed");
    assert.equal(result.proposals.length, 1);
    assert.equal(result.proposals[0].path, "src/value.js");
    assert.equal(await fs.readFile(path.join(root, "src", "value.js"), "utf8"), "module.exports = { value: 1 };\n",
      "agent must not edit original project files before explicit apply");
    assert.equal(result.attempts[0].status, "invalid_patch");
    assert.equal(result.attempts[1].status, "verification_passed");
    const applied = await import("../local/workspace.mjs").then(module =>
      module.applyFileProposal(root, result.proposals[0].id, true)
    );
    assert.equal(applied.applied, true);
    assert.equal(await fs.readFile(path.join(root, "src", "value.js"), "utf8"), "module.exports = { value: 2 };\n");
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

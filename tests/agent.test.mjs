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
  await fs.mkdir(path.join(root, ".github", "workflows"), { recursive: true });
  await fs.mkdir(path.join(root, ".syntra"), { recursive: true });
  await fs.writeFile(path.join(root, ".github", "workflows", "checks.yml"), "name: checks\n");
  await fs.writeFile(path.join(root, ".syntra", "private.json"), "{}\n");
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

test("agent blocks verification when a patch rewrites package scripts or dependency manifests", async () => {
  const root = await createProject();
  let generated = 0;
  const modelRequest = async (_method, route) => {
    if (route === "/model/status") {
      return { checkpoint_exists: true, context_tokens: 4096, tokenizer: "byte-BPE", config: { seq_len: 4096, vocab_size: 258 } };
    }
    if (route === "/model/tokenize") return { token_count: 100 };
    if (route === "/model/generate") {
      generated += 1;
      return {
        text: JSON.stringify({
          summary: "Replace the existing test command with a no-op.",
          edits: [{
            path: "package.json",
            find: '"test": "node --test"',
            replace: '"test": "node -e \\"process.exit(0)\\""',
            reason: "Regression fixture: attempt to bypass the original test command."
          }]
        }),
        generated_tokens: 40,
        trained_steps: null
      };
    }
    throw new Error("Unexpected model route: " + route);
  };
  try {
    const originalManifest = await fs.readFile(path.join(root, "package.json"), "utf8");
    const result = await runAgentTask({
      root,
      goal: "Change the test script to bypass the failing project test.",
      maxIterations: 3,
      runChecks: true,
      modelRequest
    });
    assert.equal(generated, 1, "the agent must block before running a candidate-controlled test command");
    assert.equal(result.verification.status, "verification_blocked");
    assert.equal(result.verification.passed, false);
    assert.equal(result.verification.checks[0].skipped, true);
    assert.equal(result.status, "proposals_ready_unverified");
    assert.equal(result.proposals[0].path, "package.json");
    assert.match(result.verification.note, /dependency\/build manifests or lockfiles/);
    assert.equal(await fs.readFile(path.join(root, "package.json"), "utf8"), originalManifest,
      "the original manifest must remain untouched before explicit review/apply");
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});


test("agent blocks automatic verification for dependency changes even when test scripts stay intact", async () => {
  const root = await createProject();
  let generated = 0;
  await fs.writeFile(path.join(root, "requirements.txt"), "numpy>=1.24\n");
  const modelRequest = async (_method, route) => {
    if (route === "/model/status") {
      return { checkpoint_exists: true, context_tokens: 4096, tokenizer: "byte-BPE", config: { seq_len: 4096, vocab_size: 258 } };
    }
    if (route === "/model/tokenize") return { token_count: 100 };
    if (route === "/model/generate") {
      generated += 1;
      return {
        text: JSON.stringify({
          summary: "Upgrade the dependency constraint.",
          edits: [{
            path: "requirements.txt",
            find: "numpy>=1.24",
            replace: "numpy>=2.0",
            reason: "Dependency update."
          }]
        }),
        generated_tokens: 40,
        trained_steps: null
      };
    }
    throw new Error("Unexpected model route: " + route);
  };
  try {
    const original = await fs.readFile(path.join(root, "requirements.txt"), "utf8");
    const result = await runAgentTask({
      root,
      goal: "Upgrade the numpy dependency version.",
      maxIterations: 3,
      runChecks: true,
      modelRequest
    });
    assert.equal(generated, 1);
    assert.equal(result.verification.status, "verification_blocked");
    assert.equal(result.verification.passed, false);
    assert.match(result.verification.note, /requirements\.txt/);
    assert.equal(result.proposals[0].path, "requirements.txt");
    assert.equal(await fs.readFile(path.join(root, "requirements.txt"), "utf8"), original);
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
    const workflowNode = graph.nodes.find(node => node.path === ".github/workflows/checks.yml");
    assert.ok(valueNode);
    assert.ok(testNode);
    assert.ok(workflowNode, "the repository graph should include project CI configuration in .github");
    assert.equal(graph.nodes.some(node => node.path.startsWith(".syntra/")), false,
      "SYNTRA's private audit/proposal state must remain excluded");
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
    assert.equal(result.changedFiles[0].path, "src/value.js");
    assert.match(result.changedFiles[0].diff, /^--- a\/src\/value\.js\n\+\+\+ b\/src\/value\.js/m);
    assert.match(result.changedFiles[0].diff, /^-module\.exports = \{ value: 1 \};$/m);
    assert.match(result.changedFiles[0].diff, /^\+module\.exports = \{ value: 2 \};$/m);
    assert.equal(result.changedFiles[0].diffTruncated, false);
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

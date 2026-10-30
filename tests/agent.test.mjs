import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import {
  buildRepositoryGraph,
  parsePatchOutput,
  runAgentTask,
  validateImplementationSlicePlan,
  validateReadOnlyToolPlan,
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

test("implementation slice plan validation is bounded, unique, and requires measurable acceptance", () => {
  const slices = validateImplementationSlicePlan({
    slices: [
      { goal: "Update the service response mapping.", acceptance: "The existing service tests pass." },
      { goal: "Add a response metadata helper.", acceptance: "The new helper is exported and tested." }
    ]
  }, "Update the service response and metadata helper.", 3);
  assert.equal(slices.length, 2);
  assert.equal(slices[0].id, "slice-1");
  assert.equal(slices[1].status, "pending");
  assert.throws(() => validateImplementationSlicePlan({
    slices: [
      { goal: "Update the service response mapping.", acceptance: "The tests pass." },
      { goal: "Add the metadata helper module.", acceptance: "The helper is exported and tested." }
    ]
  }, "Complete a bigger service update.", 1), /between one and 1 slices/);
  assert.throws(() => validateImplementationSlicePlan({
    slices: [
      { goal: "Update the service response mapping.", acceptance: "The tests pass." },
      { goal: "Update the service response mapping.", acceptance: "The tests pass." }
    ]
  }, "Complete a bigger service update.", 3), /unique/);
  assert.throws(() => validateImplementationSlicePlan({
    slices: [{ goal: "tiny", acceptance: "ok" }]
  }, "Complete a bigger service update.", 3), /bounded goal and measurable acceptance/);
});

test("agent implements and verifies multiple slices against one cumulative candidate", async () => {
  const root = await createProject();
  const generatedPrompts = [];
  const recoveryCheckpoints = [];
  let patchCalls = 0;
  const modelRequest = async (method, route, body) => {
    if (method === "GET" && route === "/model/status") {
      return {
        checkpoint_exists: true, context_tokens: 4096, tokenizer: "byte-BPE",
        tokenizer_vocab_size: 258, config: { seq_len: 4096, vocab_size: 258 }
      };
    }
    if (route === "/model/tokenize") return { token_count: Math.ceil(body.text.length / 4) };
    if (route === "/model/generate") {
      generatedPrompts.push(body.prompt);
      if (body.prompt.includes("SYNTRA_IMPLEMENTATION_SLICE_PLAN")) {
        return {
          text: JSON.stringify({
            slices: [
              { goal: "Update the exported value from one to two.", acceptance: "The existing value test passes." },
              { goal: "Add a small exported description helper.", acceptance: "The helper file is present and the full test suite still passes." }
            ]
          }),
          generated_tokens: 55, trained_steps: null
        };
      }
      patchCalls += 1;
      if (patchCalls === 1) {
        assert.ok(body.prompt.includes("Implementation slice 1 of 2"));
        return {
          text: JSON.stringify({
            summary: "Update the exported value.",
            edits: [{
              path: "src/value.js",
              find: "value: 1",
              replace: "value: 2",
              reason: "Complete the first verified implementation slice."
            }]
          }),
          generated_tokens: 35, trained_steps: null
        };
      }
      console.log("DEBUG_SLICE_PATCH", patchCalls, body.prompt.slice(0, 180), body.prompt.length);
      if (!body.prompt.includes("Implementation slice 2 of 2")) {
        return { text: '{"summary":"debug stop"}', generated_tokens: 8, trained_steps: null };
      }
      assert.ok(body.prompt.includes("Implementation slice 2 of 2"));
      assert.ok(body.prompt.includes("module.exports = { value: 2 }"),
        "slice two must receive source context from the verified first slice");
      return {
        text: JSON.stringify({
          summary: "Add an exported description helper.",
          edits: [{
            path: "src/description.js",
            find: "",
            replace: "module.exports = { description: 'SYNTRA' };\n",
            reason: "Complete the second implementation slice."
          }]
        }),
        generated_tokens: 40, trained_steps: null
      };
    }
    throw new Error("Unexpected model route: " + route);
  };
  try {
    const result = await runAgentTask({
      root,
      goal: "Update the exported value and add a description helper.",
      maxIterations: 2,
      maxSlices: 2,
      runChecks: true,
      modelRequest,
      onProgress: update => {
        if (Object.prototype.hasOwnProperty.call(update, "recoveryPatch")) recoveryCheckpoints.push(update.recoveryPatch);
      }
    });
    console.log("DEBUG_MULTI_RESULT", JSON.stringify({
      patchCalls, status: result.status, attempts: result.attempts,
      verification: result.verification, observations: result.observations,
      slices: result.implementationSlices
    }));
    assert.equal(patchCalls, 2);
    assert.ok(recoveryCheckpoints.length > 0);
    assert.ok(recoveryCheckpoints.every(value => value === null),
      "multi-slice runs must restart from the overall goal instead of persisting a candidate-relative patch");
    assert.equal(generatedPrompts.filter(prompt => prompt.includes("SYNTRA_IMPLEMENTATION_SLICE_PLAN")).length, 1);
    assert.equal(result.implementationSlices.length, 2);
    assert.ok(result.implementationSlices.every(slice => slice.status === "completed"));
    assert.equal(result.verification.status, "passed");
    assert.equal(result.verification.passed, true);
    assert.equal(result.proposals.length, 2);
    assert.deepEqual(result.changedFiles.map(file => file.path).sort(), ["src/description.js", "src/value.js"]);
    assert.ok(result.observations.some(item =>
      item.kind === "candidate-reinspection" && item.stage === "slice-2" && item.status === "observed"
    ));
    assert.equal(await fs.readFile(path.join(root, "src", "value.js"), "utf8"), "module.exports = { value: 1 };\n");
    await assert.rejects(() => fs.readFile(path.join(root, "src", "description.js"), "utf8"), { code: "ENOENT" });
    const applied = await import("../local/workspace.mjs").then(module =>
      module.applyFileProposals(root, result.proposals.map(proposal => proposal.id), true)
    );
    assert.equal(applied.applied.length, 2);
    assert.equal(await fs.readFile(path.join(root, "src", "value.js"), "utf8"), "module.exports = { value: 2 };\n");
    assert.equal(await fs.readFile(path.join(root, "src", "description.js"), "utf8"), "module.exports = { description: 'SYNTRA' };\n");
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("an incomplete later slice cannot inherit an earlier passing verification status", async () => {
  const root = await createProject();
  let patchCalls = 0;
  const modelRequest = async (method, route, body) => {
    if (method === "GET" && route === "/model/status") {
      return {
        checkpoint_exists: true, context_tokens: 4096, tokenizer: "byte-BPE",
        tokenizer_vocab_size: 258, config: { seq_len: 4096, vocab_size: 258 }
      };
    }
    if (route === "/model/tokenize") return { token_count: Math.ceil(body.text.length / 4) };
    if (route === "/model/generate") {
      if (body.prompt.includes("SYNTRA_IMPLEMENTATION_SLICE_PLAN")) {
        return {
          text: JSON.stringify({
            slices: [
              { goal: "Update the exported value from one to two.", acceptance: "The existing value test passes." },
              { goal: "Add a description helper module.", acceptance: "The helper exists and the full suite passes." }
            ]
          }),
          generated_tokens: 40, trained_steps: null
        };
      }
      patchCalls += 1;
      if (patchCalls === 1) {
        return {
          text: JSON.stringify({
            summary: "Update exported value.",
            edits: [{
              path: "src/value.js", find: "value: 1", replace: "value: 2",
              reason: "First slice implementation."
            }]
          }),
          generated_tokens: 30, trained_steps: null
        };
      }
      return { text: '{"summary":"unfinished"', generated_tokens: 4, trained_steps: null };
    }
    throw new Error("Unexpected model route: " + route);
  };
  try {
    const result = await runAgentTask({
      root,
      goal: "Update the exported value and add a description helper.",
      maxIterations: 1,
      maxSlices: 2,
      runChecks: true,
      modelRequest
    });
    console.log("DEBUG_INCOMPLETE_RESULT", JSON.stringify({
      patchCalls, status: result.status, attempts: result.attempts,
      verification: result.verification, observations: result.observations,
      slices: result.implementationSlices
    }));
    assert.equal(patchCalls, 2);
    assert.equal(result.implementationSlices[0].status, "completed");
    assert.equal(result.implementationSlices[1].status, "blocked");
    assert.equal(result.verification.status, "slice_incomplete");
    assert.equal(result.verification.passed, false);
    assert.deepEqual(result.verification.incompleteSlices, ["slice-2"]);
    assert.equal(result.status, "proposals_ready_unverified");
    assert.equal(result.proposals.length, 1);
    assert.equal(await fs.readFile(path.join(root, "src", "value.js"), "utf8"), "module.exports = { value: 1 };\n");
    await assert.rejects(() => fs.readFile(path.join(root, "src", "description.js"), "utf8"), { code: "ENOENT" });
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

test("model-selected context plans allow only bounded search and safe listed-file reads", () => {
  const valid = validateReadOnlyToolPlan({
    calls: [
      { tool: "workspace.search", input: { query: "connection pool", limit: 5 } },
      { tool: "workspace.read", input: { path: "src/server.js", startLine: 10, endLine: 40 } }
    ]
  }, ["src/server.js", "src/db.js"]);
  assert.deepEqual(valid, [
    { tool: "workspace.search", input: { query: "connection pool", limit: 5 } },
    { tool: "workspace.read", input: { path: "src/server.js", startLine: 10, endLine: 40 } }
  ]);
  assert.throws(() => validateReadOnlyToolPlan({
    calls: [{ tool: "workspace.check", input: { task: "test" } }]
  }, ["src/server.js"]), /restricted to workspace.search and workspace.read/);
  assert.throws(() => validateReadOnlyToolPlan({
    calls: [{ tool: "workspace.read", input: { path: ".env" } }]
  }, ["src/server.js"]), /safe file list/);
  assert.throws(() => validateReadOnlyToolPlan({
    calls: [{ tool: "workspace.read", input: { path: "src/server.js", startLine: 1, endLine: 1000 } }]
  }, ["src/server.js"]), /limited to 200 lines/);
  assert.throws(() => validateReadOnlyToolPlan({
    calls: [{ tool: "workspace.search", input: { query: "db", limit: 50 } }]
  }, ["src/server.js"]), /limit from 1 to 10/);
  assert.throws(() => validateReadOnlyToolPlan({
    calls: Array.from({ length: 5 }, () => ({ tool: "workspace.search", input: { query: "handler" } }))
  }, ["src/server.js"]), /at most four calls/);
});

test("agent executes only model-selected read-only retrieval calls and feeds results into patch context", async () => {
  const root = await createProject();
  const generated = [];
  const modelRequest = async (method, route, body) => {
    if (method === "GET" && route === "/model/status") {
      return {
        checkpoint_exists: true, context_tokens: 4096, tokenizer: "byte-BPE",
        tokenizer_vocab_size: 258, config: { seq_len: 4096, vocab_size: 258 }
      };
    }
    if (route === "/model/tokenize") return { token_count: Math.ceil(body.text.length / 4) };
    if (route === "/model/generate") {
      generated.push(body.prompt);
      if (body.prompt.includes("SYNTRA_READ_ONLY_TOOL_PLAN")) {
        return {
          text: JSON.stringify({
            calls: [
              { tool: "workspace.search", input: { query: "value", limit: 5 } },
              { tool: "workspace.read", input: { path: "src/value.js", startLine: 1, endLine: 1 } }
            ]
          }),
          generated_tokens: 35, trained_steps: null
        };
      }
      assert.match(body.prompt, /MODEL-SELECTED READ-ONLY CONTEXT/);
      assert.ok(body.prompt.includes("module.exports = { value: 1 }"));
      return {
        text: JSON.stringify({
          summary: "Update the exported value.",
          edits: [{
            path: "src/value.js",
            find: "value: 1",
            replace: "value: 2",
            reason: "Update the value requested by the task."
          }]
        }),
        generated_tokens: 30, trained_steps: null
      };
    }
    throw new Error("Unexpected model route: " + route);
  };
  try {
    const result = await runAgentTask({
      root,
      goal: "Change the exported value from 1 to 2.",
      maxIterations: 1,
      runChecks: false,
      modelToolSelection: true,
      modelRequest
    });
    console.log("DEBUG_RETRIEVAL_RESULT", JSON.stringify({
      generated: generated.map(p => ({ marker: p.includes("SYNTRA_READ_ONLY_TOOL_PLAN"), length: p.length })),
      status: result.status, attempts: result.attempts, verification: result.verification
    }));
    assert.equal(generated.length, 2, "one safe tool-plan generation should precede patch generation");
    assert.equal(result.proposals.length, 1);
    assert.equal(result.proposals[0].path, "src/value.js");
    assert.deepEqual(result.toolCalls.map(call => call.name), [
      "workspace.inspect", "git.status", "git.diff-summary",
      "workspace.search", "workspace.read", "workspace.proposals.stage"
    ]);
    assert.ok(result.toolCalls.every(call => call.auditRecorded));
    assert.ok(result.observations.some(item => item.kind === "model-readonly-tool-plan" && item.status === "accepted"));
    assert.ok(result.observations.some(item => item.kind === "workspace.search" && item.status === "observed"));
    assert.ok(result.observations.some(item => item.kind === "workspace.read" && item.path === "src/value.js"));
    assert.equal(await fs.readFile(path.join(root, "src", "value.js"), "utf8"), "module.exports = { value: 1 };\n",
      "model-selected retrieval and patch staging must not write the original project file");
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

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
    assert.deepEqual(result.toolCalls.map(call => call.name), [
      "workspace.inspect", "git.status", "git.diff-summary", "workspace.proposals.stage"
    ]);
    assert.ok(result.toolCalls.every(call => call.auditRecorded));
    const toolAudit = (await fs.readFile(path.join(root, ".syntra", "actions.jsonl"), "utf8"))
      .split(/\r?\n/).filter(Boolean).map(line => JSON.parse(line))
      .filter(entry => entry.action === "tool_runtime_event");
    assert.equal(toolAudit.length, 8);
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

test("agent rejects candidate verification when a successful check mutates a proposed source file", async () => {
  const root = await createProject();
  await fs.writeFile(path.join(root, "tests", "mutate-candidate.mjs"), [
    'import { writeFile } from "node:fs/promises";',
    'await writeFile("src/value.js", "module.exports = { value: 99 };");'
  ].join("\n") + "\n");
  const manifest = JSON.parse(await fs.readFile(path.join(root, "package.json"), "utf8"));
  manifest.scripts.test = "node tests/mutate-candidate.mjs";
  await fs.writeFile(path.join(root, "package.json"), JSON.stringify(manifest, null, 2));
  const modelRequest = async (_method, route) => {
    if (route === "/model/status") return {
      checkpoint_exists: true, context_tokens: 4096, tokenizer: "byte-BPE",
      tokenizer_vocab_size: 258, config: { seq_len: 4096, vocab_size: 258 }
    };
    if (route === "/model/tokenize") return { token_count: 120 };
    if (route === "/model/generate") return {
      text: JSON.stringify({
        summary: "Change the value expected by the check fixture.",
        edits: [{
          path: "src/value.js",
          find: "value: 1",
          replace: "value: 2",
          reason: "The proposed value is 2 before the check runs."
        }]
      }),
      generated_tokens: 30, trained_steps: null
    };
    throw new Error("Unexpected model route: " + route);
  };
  try {
    const result = await runAgentTask({
      root,
      goal: "Update the exported value so the project check can run.",
      maxIterations: 1,
      runChecks: true,
      modelRequest
    });
    assert.equal(result.verification.status, "failed");
    assert.equal(result.verification.passed, false);
    assert.ok(result.verification.checks.some(item => item.task === "test" && item.passed === true),
      "the configured check process itself must exit successfully");
    assert.ok(result.verification.checks.some(item => item.task === "candidate-integrity" && !item.passed));
    assert.ok(result.observations.some(item => item.kind === "candidate-file-integrity"
      && item.path === "src/value.js" && item.status === "mismatch"));
    assert.equal(result.status, "proposals_ready_unverified");
    assert.equal(await fs.readFile(path.join(root, "src", "value.js"), "utf8"), "module.exports = { value: 1 };\n",
      "project source must remain unchanged outside the temporary candidate");
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
    assert.deepEqual(result.toolCalls.map(call => call.name), [
      "workspace.inspect", "git.status", "git.diff-summary", "workspace.checks.discover",
      "workspace.check", "workspace.read", "workspace.proposals.stage"
    ]);
    assert.ok(result.observations.some(item => item.kind === "workspace.checks.discover" && item.status === "observed"));
    assert.ok(result.observations.some(item => item.kind === "candidate-file-integrity"
      && item.path === "src/value.js" && item.status === "matched"));
    assert.ok(result.toolCalls.every(call => call.auditRecorded));
    const applied = await import("../local/workspace.mjs").then(module =>
      module.applyFileProposal(root, result.proposals[0].id, true)
    );
    assert.equal(applied.applied, true);
    assert.equal(await fs.readFile(path.join(root, "src", "value.js"), "utf8"), "module.exports = { value: 2 };\n");
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

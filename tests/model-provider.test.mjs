import assert from "node:assert/strict";
import test from "node:test";

import {
  chooseLocalModel,
  detectLocalModel,
  estimateModelResourceProfile,
  getProviderStatus,
  handleLocalModelRequest
} from "../local/model-provider.mjs";

const GB = 1024 ** 3;

test("model selection prefers local coding models and excludes cloud-only or oversized models", () => {
  const models = [
    { name: "llama3.2:latest", size: 4 * GB },
    { name: "qwen2.5-coder:14b", size: 9 * GB },
    { name: "qwen3-coder:30b", size: 19 * GB },
    { name: "qwen3-coder:480b-cloud", size: 290 * GB },
    { name: "qwen3-coder-next:latest", size: 44 * GB }
  ];
  assert.equal(chooseLocalModel(models, "", 32 * GB)?.name, "qwen2.5-coder:14b");
  assert.equal(chooseLocalModel(models, "qwen2.5-coder:14b", 32 * GB)?.name, "qwen2.5-coder:14b");
  assert.equal(chooseLocalModel(models, "qwen3-coder:480b-cloud", 512 * GB), null);
  assert.equal(chooseLocalModel(models, "", 8 * GB)?.name, "llama3.2:latest");
});

test("model selection leaves memory headroom and applies the same limit to explicit model requests", () => {
  const models = [
    { name: "qwen3-coder:30b", size: 19 * GB },
    { name: "giant-coder:latest", size: 23 * GB },
    { name: "llama3.2:latest", size: 4 * GB }
  ];
  assert.equal(chooseLocalModel(models, "", 32 * GB)?.name, "llama3.2:latest");
  assert.equal(chooseLocalModel(models, "giant-coder:latest", 32 * GB), null,
    "an explicit preference must never bypass the memory safety threshold");
  assert.equal(chooseLocalModel([{ name: "exact-limit:latest", size: 17.6 * GB }], "", 32 * GB)?.name, "exact-limit:latest");
  assert.equal(chooseLocalModel([{ name: "just-over-limit:latest", size: 17.61 * GB }], "", 32 * GB), null);
});

test("resource profile estimates KV-cache demand and reports loaded runtime memory", () => {
  const profile = estimateModelResourceProfile({
    modelSizeBytes: 8 * GB,
    totalMemoryBytes: 32 * GB,
    availableMemoryBytes: 12 * GB,
    contextTokens: 8192,
    modelInfo: {
      "qwen.block_count": 40,
      "qwen.embedding_length": 5120,
      "qwen.attention.head_count": 40,
      "qwen.attention.head_count_kv": 8
    },
    runningModel: { name: "qwen:latest", size: 9 * GB, size_vram: 2 * GB }
  });
  assert.equal(profile.modelAlreadyLoaded, true);
  assert.equal(profile.runtimeVramBytes, 2 * GB);
  assert.equal(profile.kvCacheBytesPerToken, 40 * 8 * 128 * 4);
  assert.equal(profile.estimatedKvCacheBytes, profile.kvCacheBytesPerToken * 8192);
  assert.equal(profile.estimatedHostWorkingSetBytes, 7 * GB + profile.estimatedKvCacheBytes);
  assert.equal(profile.conservativeBudgetBytes, 32 * GB * 0.55);
  assert.equal(profile.confidence, "estimated");
  assert.match(profile.limitation, /GPU offload/);
});

test("resource profile stays honest when Ollama metadata is incomplete", () => {
  const profile = estimateModelResourceProfile({
    modelSizeBytes: 4 * GB,
    totalMemoryBytes: 16 * GB,
    availableMemoryBytes: 6 * GB
  });
  assert.equal(profile.estimatedKvCacheBytes, null);
  assert.equal(profile.estimatedHostWorkingSetBytes, null);
  assert.equal(profile.withinConservativeBudget, null);
  assert.equal(profile.confidence, "partial");
});

test("native mode never probes Ollama; auto mode safely falls back if it is offline", async () => {
  let calls = 0;
  const failingFetch = async () => {
    calls += 1;
    throw new Error("connection refused");
  };
  assert.equal(await detectLocalModel({ mode: "native", fetchImpl: failingFetch }), null);
  assert.equal(calls, 0);
  assert.equal(await detectLocalModel({ mode: "auto", fetchImpl: failingFetch }), null);
  assert.equal(calls, 1);
  await assert.rejects(
    () => detectLocalModel({ mode: "ollama", fetchImpl: failingFetch }),
    /Ollama is not responding at http:\/\/127\.0\.0\.1:11434/
  );
});

test("auto mode selects an installed local model and reports local-only status", async () => {
  const fetchImpl = async url => {
    assert.equal(url, "http://127.0.0.1:11434/api/tags");
    return new Response(JSON.stringify({
      models: [
        { name: "llama3.2:latest", size: 4 * GB },
        { name: "qwen2.5-coder:14b", size: 9 * GB }
      ]
    }), { status: 200, headers: { "Content-Type": "application/json" } });
  };
  const provider = await detectLocalModel({ mode: "auto", fetchImpl, totalMemoryBytes: 32 * GB });
  assert.equal(provider.provider, "ollama-local");
  assert.equal(provider.model, "qwen2.5-coder:14b");
  const status = getProviderStatus(provider);
  assert.equal(status.checkpoint_exists, true);
  assert.equal(status.external_ai_api, false);
  assert.equal(status.local_only, true);
  assert.equal(status.context_tokens, 8192);
  assert.equal(status.tokenizer, "utf8-byte-ratio-estimate");
  assert.equal(status.tokenizer_is_estimate, true);
});

test("local generation uses only loopback Ollama and JSON mode for structured patches", async () => {
  let capturedUrl = "";
  let capturedBody = null;
  const provider = {
    provider: "ollama-local",
    baseUrl: "http://127.0.0.1:11434",
    model: "qwen2.5-coder:14b",
    modelSizeBytes: 9 * GB,
    contextTokens: 8192,
    tokenizer: "ollama-local-tokenizer"
  };
  const fetchImpl = async (url, options) => {
    capturedUrl = url;
    capturedBody = JSON.parse(options.body);
    return new Response(JSON.stringify({
      message: { role: "assistant", content: '{"summary":"Fixed","edits":[]}' },
      prompt_eval_count: 100,
      eval_count: 12
    }), { status: 200, headers: { "Content-Type": "application/json" } });
  };
  const result = await handleLocalModelRequest(provider, "POST", "/model/generate", {
    prompt: 'Return one JSON object now. Schema: {"summary":"...","edits":[]}',
    max_new_tokens: 512,
    temperature: 0.1,
    top_k: 30
  }, { fetchImpl, timeoutMs: 1000 });
  assert.equal(capturedUrl, "http://127.0.0.1:11434/api/chat");
  assert.equal(capturedBody.model, "qwen2.5-coder:14b");
  assert.equal(capturedBody.stream, false);
  assert.equal(capturedBody.format.type, "object");
  assert.deepEqual(capturedBody.format.required, ["summary", "edits"]);
  assert.equal(capturedBody.messages[0].role, "system");
  assert.equal(capturedBody.messages[1].role, "user");
  assert.equal(capturedBody.options.num_ctx, 8192);
  assert.equal(capturedBody.options.num_predict, 512);
  assert.equal(result.text, '{"summary":"Fixed","edits":[]}');
  assert.equal(result.provider, "ollama-local");
  assert.equal(result.generated_tokens, 12);
});

test("local model read-only tool plans expose the same safe search tools as the agent allowlist", async () => {
  let capturedBody = null;
  const provider = {
    provider: "ollama-local",
    baseUrl: "http://127.0.0.1:11434",
    model: "qwen2.5-coder:14b",
    modelSizeBytes: 9 * GB,
    contextTokens: 8192,
    tokenizer: "ollama-local-tokenizer"
  };
  const fetchImpl = async (_url, options) => {
    capturedBody = JSON.parse(options.body);
    return new Response(JSON.stringify({
      message: { role: "assistant", content: '{"calls":[{"tool":"workspace.search","input":{"query":"router"}}]}' },
      prompt_eval_count: 80, eval_count: 15
    }), { status: 200, headers: { "Content-Type": "application/json" } });
  };
  const result = await handleLocalModelRequest(provider, "POST", "/model/generate", {
    prompt: 'SYNTRA_READ_ONLY_TOOL_PLAN\nReturn exactly one JSON object with a calls array.',
    max_new_tokens: 256, temperature: 0.1, top_k: 20
  }, { fetchImpl, timeoutMs: 1000 });
  assert.equal(capturedBody.format.type, "object");
  assert.deepEqual(capturedBody.format.required, ["calls"]);
  assert.deepEqual(
    capturedBody.format.properties.calls.items.properties.tool.enum,
    ["workspace.search", "workspace.read", "workspace.symbols.search", "workspace.memory.search"]
  );
  assert.deepEqual(
    Object.keys(capturedBody.format.properties.calls.items.properties.input.properties).sort(),
    ["endLine", "limit", "path", "query", "startLine"]
  );
  assert.equal(capturedBody.messages[0].role, "system");
  assert.match(capturedBody.messages[0].content, /read-only context planner/);
  assert.equal(result.text, '{"calls":[{"tool":"workspace.search","input":{"query":"router"}}]}');
});

test("local model implementation slice plans use a bounded strict JSON schema", async () => {
  let capturedBody = null;
  const provider = {
    provider: "ollama-local",
    baseUrl: "http://127.0.0.1:11434",
    model: "qwen2.5-coder:14b",
    modelSizeBytes: 9 * GB,
    contextTokens: 8192,
    tokenizer: "ollama-local-tokenizer"
  };
  const fetchImpl = async (_url, options) => {
    capturedBody = JSON.parse(options.body);
    return new Response(JSON.stringify({
      message: {
        role: "assistant",
        content: '{"slices":[{"goal":"Update the API response mapping.","acceptance":"The API regression tests pass."}]}'
      },
      prompt_eval_count: 100, eval_count: 20
    }), { status: 200, headers: { "Content-Type": "application/json" } });
  };
  const result = await handleLocalModelRequest(provider, "POST", "/model/generate", {
    prompt: "SYNTRA_IMPLEMENTATION_SLICE_PLAN\nReturn one bounded slice plan.",
    max_new_tokens: 384, temperature: 0.1, top_k: 20
  }, { fetchImpl, timeoutMs: 1000 });
  assert.equal(capturedBody.format.type, "object");
  assert.deepEqual(capturedBody.format.required, ["slices"]);
  assert.equal(capturedBody.format.properties.slices.minItems, 1);
  assert.equal(capturedBody.format.properties.slices.maxItems, 3);
  assert.deepEqual(
    Object.keys(capturedBody.format.properties.slices.items.properties).sort(),
    ["acceptance", "goal"]
  );
  assert.equal(capturedBody.messages[0].role, "system");
  assert.match(capturedBody.messages[0].content, /implementation planner/);
  assert.equal(result.text, '{"slices":[{"goal":"Update the API response mapping.","acceptance":"The API regression tests pass."}]}');
});

test("local generation streams arbitrary NDJSON chunks and accumulates the complete answer", async () => {
  let capturedBody = null;
  const provider = {
    provider: "ollama-local",
    baseUrl: "http://127.0.0.1:11434",
    model: "qwen2.5-coder:14b",
    modelSizeBytes: 9 * GB,
    contextTokens: 8192,
    tokenizer: "ollama-local-tokenizer"
  };
  const lines = [
    JSON.stringify({ message: { role: "assistant", content: "Hello" }, done: false }),
    JSON.stringify({ message: { role: "assistant", content: " there" }, done: false }),
    JSON.stringify({ message: { role: "assistant", content: "!" }, done: false }),
    JSON.stringify({ message: { role: "assistant", content: "" }, done: true, eval_count: 3, prompt_eval_count: 55 })
  ].join("\n") + "\n";
  const fetchImpl = async (_url, options) => {
    capturedBody = JSON.parse(options.body);
    const encoder = new TextEncoder();
    const bytes = encoder.encode(lines);
    const split = Math.floor(bytes.length / 2);
    const stream = new ReadableStream({
      start(controller) {
        controller.enqueue(bytes.slice(0, split));
        controller.enqueue(bytes.slice(split));
        controller.close();
      }
    });
    return new Response(stream, { status: 200, headers: { "Content-Type": "application/x-ndjson" } });
  };
  const updates = [];
  const result = await handleLocalModelRequest(provider, "POST", "/model/generate", {
    prompt: "Say hello", max_new_tokens: 64
  }, { fetchImpl, onToken: (delta, fullText) => updates.push({ delta, fullText }), timeoutMs: 1000 });
  assert.equal(capturedBody.stream, true);
  assert.equal(result.text, "Hello there!");
  assert.equal(result.generated_tokens, 3);
  assert.equal(result.prompt_tokens, 55);
  assert.deepEqual(updates, [
    { delta: "Hello", fullText: "Hello" },
    { delta: " there", fullText: "Hello there" },
    { delta: "!", fullText: "Hello there!" }
  ]);
});

test("local token estimate is bounded and clearly marked as an estimate", async () => {
  const provider = {
    provider: "ollama-local",
    baseUrl: "http://127.0.0.1:11434",
    model: "qwen2.5-coder:14b",
    modelSizeBytes: null,
    contextTokens: 8192,
    tokenizer: "ollama-local-tokenizer"
  };
  const result = await handleLocalModelRequest(provider, "POST", "/model/tokenize", { text: "const value = 42;" });
  assert.ok(result.token_count > 0);
  assert.equal(result.is_estimate, true);
  assert.equal(result.provider, "ollama-local");
});


test("default model discovery caches short-lived misses to avoid repeated Ollama probes", async () => {
  const originalFetch = globalThis.fetch;
  const previousMode = process.env.SYNTRA_MODEL_PROVIDER;
  let calls = 0;
  globalThis.fetch = async url => {
    calls += 1;
    assert.equal(url, "http://127.0.0.1:11434/api/tags");
    return new Response(JSON.stringify({ models: [] }), {
      status: 200, headers: { "Content-Type": "application/json" }
    });
  };
  try {
    delete process.env.SYNTRA_MODEL_PROVIDER;
    const options = { requestedModel: "syntra-cache-regression-test" };
    assert.equal(await detectLocalModel(options), null);
    assert.equal(await detectLocalModel(options), null);
    assert.equal(calls, 1, "a repeated miss inside the short cache window should not probe Ollama twice");
  } finally {
    globalThis.fetch = originalFetch;
    if (previousMode === undefined) delete process.env.SYNTRA_MODEL_PROVIDER;
    else process.env.SYNTRA_MODEL_PROVIDER = previousMode;
  }
});


test("an incomplete Ollama stream is retried only before any text has been emitted", async () => {
  const provider = {
    provider: "ollama-local", baseUrl: "http://127.0.0.1:11434",
    model: "qwen2.5-coder:14b", modelSizeBytes: 9 * GB,
    contextTokens: 8192, tokenizer: "ollama-local-tokenizer"
  };
  let calls = 0;
  const lines = [
    JSON.stringify({ message: { role: "assistant", content: "Recovered answer" }, done: false }),
    JSON.stringify({ message: { role: "assistant", content: "" }, done: true, eval_count: 2 })
  ].join("\n") + "\n";
  const fetchImpl = async () => {
    calls += 1;
    if (calls === 1) {
      return new Response(JSON.stringify({ message: { role: "assistant", content: "" }, done: false }) + "\n", {
        status: 200, headers: { "Content-Type": "application/x-ndjson" }
      });
    }
    return new Response(lines, { status: 200, headers: { "Content-Type": "application/x-ndjson" } });
  };
  const updates = [];
  const result = await handleLocalModelRequest(provider, "POST", "/model/generate", { prompt: "Recover" }, {
    fetchImpl, onToken: (delta, fullText) => updates.push({ delta, fullText }),
    timeoutMs: 1000, retryDelayMs: 0
  });
  assert.equal(calls, 2);
  assert.equal(result.text, "Recovered answer");
  assert.deepEqual(updates, [{ delta: "Recovered answer", fullText: "Recovered answer" }]);
});

test("an interrupted Ollama stream with partial text is not replayed", async () => {
  const provider = {
    provider: "ollama-local", baseUrl: "http://127.0.0.1:11434",
    model: "qwen2.5-coder:14b", modelSizeBytes: 9 * GB,
    contextTokens: 8192, tokenizer: "ollama-local-tokenizer"
  };
  let calls = 0;
  const fetchImpl = async () => {
    calls += 1;
    return new Response(JSON.stringify({
      message: { role: "assistant", content: "Partial answer" }, done: false
    }) + "\n", { status: 200, headers: { "Content-Type": "application/x-ndjson" } });
  };
  const updates = [];
  await assert.rejects(
    () => handleLocalModelRequest(provider, "POST", "/model/generate", { prompt: "Recover" }, {
      fetchImpl, onToken: (delta, fullText) => updates.push({ delta, fullText }),
      timeoutMs: 1000, retryDelayMs: 0
    }),
    error => error.code === "STREAM_INTERRUPTED" && /did not replay/.test(error.message)
  );
  assert.equal(calls, 1);
  assert.deepEqual(updates, [{ delta: "Partial answer", fullText: "Partial answer" }]);
});

test("transient Ollama HTTP failures receive one bounded retry", async () => {
  const provider = {
    provider: "ollama-local", baseUrl: "http://127.0.0.1:11434",
    model: "qwen2.5-coder:14b", modelSizeBytes: 9 * GB,
    contextTokens: 8192, tokenizer: "ollama-local-tokenizer"
  };
  let calls = 0;
  const fetchImpl = async () => {
    calls += 1;
    if (calls === 1) return new Response(JSON.stringify({ error: "service unavailable" }), { status: 503 });
    return new Response(JSON.stringify({ message: { role: "assistant", content: "Recovered" }, eval_count: 1 }), { status: 200 });
  };
  const result = await handleLocalModelRequest(provider, "POST", "/model/generate", { prompt: "Recover" }, {
    fetchImpl, timeoutMs: 1000, retryDelayMs: 0
  });
  assert.equal(calls, 2);
  assert.equal(result.text, "Recovered");
});

test("context exhaustion is actionable and never retried", async () => {
  const provider = {
    provider: "ollama-local", baseUrl: "http://127.0.0.1:11434",
    model: "qwen2.5-coder:14b", modelSizeBytes: 9 * GB,
    contextTokens: 8192, tokenizer: "ollama-local-tokenizer"
  };
  let calls = 0;
  const fetchImpl = async () => {
    calls += 1;
    return new Response(JSON.stringify({ error: "prompt is too long for the context window" }), { status: 400 });
  };
  await assert.rejects(
    () => handleLocalModelRequest(provider, "POST", "/model/generate", { prompt: "Too much context" }, {
      fetchImpl, timeoutMs: 1000, retryDelayMs: 0
    }),
    error => error.code === "CONTEXT_EXHAUSTED" && /Reduce the task or retrieved workspace context/.test(error.message)
  );
  assert.equal(calls, 1);
});

test("user cancellation prevents the Ollama request and bypasses retries", async () => {
  const provider = {
    provider: "ollama-local", baseUrl: "http://127.0.0.1:11434",
    model: "qwen2.5-coder:14b", modelSizeBytes: 9 * GB,
    contextTokens: 8192, tokenizer: "ollama-local-tokenizer"
  };
  const controller = new AbortController();
  controller.abort();
  let calls = 0;
  await assert.rejects(
    () => handleLocalModelRequest(provider, "POST", "/model/generate", { prompt: "Cancel" }, {
      fetchImpl: async () => { calls += 1; throw new Error("must not be called"); },
      signal: controller.signal, timeoutMs: 1000, retryDelayMs: 0
    }),
    error => error.code === "GENERATION_CANCELLED"
  );
  assert.equal(calls, 0);
});


test("Ollama patch-review mode uses a strict reviewer-only JSON schema", async () => {
  const provider = {
    provider: "ollama-local",
    baseUrl: "http://127.0.0.1:11434",
    model: "qwen2.5-coder:14b",
    modelSizeBytes: 9 * GB,
    contextTokens: 8192,
    tokenizer: "ollama-local-tokenizer"
  };
  let capturedBody = null;
  const fetchImpl = async (_url, options) => {
    capturedBody = JSON.parse(options.body);
    return new Response(JSON.stringify({
      message: {
        role: "assistant",
        content: JSON.stringify({
          verdict: "accept",
          rationale: "The patch matches the request.",
          findings: []
        })
      },
      prompt_eval_count: 100,
      eval_count: 20
    }), { status: 200, headers: { "Content-Type": "application/json" } });
  };
  const result = await handleLocalModelRequest(provider, "POST", "/model/generate", {
    prompt: "SYNTRA_PATCH_REVIEW\nReturn exactly one JSON object with verdict, rationale, and findings.",
    max_new_tokens: 256, temperature: 0.05
  }, { fetchImpl, timeoutMs: 1000 });
  assert.equal(capturedBody.format.type, "object");
  assert.deepEqual(capturedBody.format.required, ["verdict", "rationale", "findings"]);
  assert.deepEqual(capturedBody.format.properties.verdict.enum, ["accept", "revise"]);
  assert.deepEqual(
    Object.keys(capturedBody.format.properties.findings.items.properties).sort(),
    ["issue", "path", "recommendation", "severity"]
  );
  assert.match(capturedBody.messages[0].content, /critical second-pass patch reviewer/);
  assert.match(capturedBody.messages[0].content, /same model that generated the candidate/);
  assert.equal(JSON.parse(result.text).verdict, "accept");
});

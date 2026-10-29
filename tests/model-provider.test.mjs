import assert from "node:assert/strict";
import test from "node:test";

import {
  chooseLocalModel,
  detectLocalModel,
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
  assert.equal(chooseLocalModel(models, "", 32 * GB)?.name, "qwen3-coder:30b");
  assert.equal(chooseLocalModel(models, "qwen2.5-coder:14b", 32 * GB)?.name, "qwen2.5-coder:14b");
  assert.equal(chooseLocalModel(models, "qwen3-coder:480b-cloud", 512 * GB), null);
  assert.equal(chooseLocalModel(models, "", 8 * GB)?.name, "llama3.2:latest");
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

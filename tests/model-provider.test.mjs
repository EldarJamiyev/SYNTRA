import assert from "node:assert/strict";
import http from "node:http";
import test from "node:test";
import {
  estimateHybridTokenCount,
  generateHybridCompletion,
  getHybridModelConfig,
  getHybridModelStatus,
  isHybridModelEnabled
} from "../local/model-provider.mjs";

async function withServer(handler, run) {
  const server = http.createServer(handler);
  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });
  const address = server.address();
  const prior = { ...process.env };
  process.env.SYNTRA_MODEL_PROVIDER = "openai-compatible";
  process.env.SYNTRA_HYBRID_BASE_URL = "http://127.0.0.1:" + address.port + "/v1";
  process.env.SYNTRA_HYBRID_MODEL = "test-model";
  process.env.SYNTRA_HYBRID_CONTEXT_TOKENS = "8192";
  delete process.env.SYNTRA_HYBRID_ALLOW_REMOTE;
  delete process.env.SYNTRA_HYBRID_API_KEY;
  try { await run(); }
  finally {
    for (const key of Object.keys(process.env)) if (!(key in prior)) delete process.env[key];
    Object.assign(process.env, prior);
    await new Promise(resolve => server.close(resolve));
  }
}

test("native remains the default and hybrid mode requires an explicit provider value", () => {
  assert.equal(isHybridModelEnabled({}), false);
  assert.equal(getHybridModelConfig({}), null);
  assert.equal(isHybridModelEnabled({ SYNTRA_MODEL_PROVIDER: "native" }), false);
  assert.equal(isHybridModelEnabled({ SYNTRA_MODEL_PROVIDER: "openai-compatible" }), true);
  assert.throws(() => getHybridModelConfig({ SYNTRA_MODEL_PROVIDER: "ollama" }), /must be 'native' or/);
});

test("hybrid configuration requires an actual model identifier and explicit context", () => {
  assert.throws(() => getHybridModelConfig({ SYNTRA_MODEL_PROVIDER: "openai-compatible" }), /requires SYNTRA_HYBRID_BASE_URL/);
  assert.throws(() => getHybridModelConfig({
    SYNTRA_MODEL_PROVIDER: "openai-compatible",
    SYNTRA_HYBRID_BASE_URL: "http://127.0.0.1:1234/v1",
    SYNTRA_HYBRID_MODEL: "model",
    SYNTRA_HYBRID_CONTEXT_TOKENS: "guess"
  }), /must be an integer/);
});

test("remote HTTP is rejected and remote HTTPS requires a second explicit opt-in", () => {
  const base = {
    SYNTRA_MODEL_PROVIDER: "openai-compatible",
    SYNTRA_HYBRID_MODEL: "model",
    SYNTRA_HYBRID_CONTEXT_TOKENS: "4096"
  };
  assert.throws(() => getHybridModelConfig({ ...base, SYNTRA_HYBRID_BASE_URL: "http://example.com/v1" }), /restricted to loopback/);
  assert.throws(() => getHybridModelConfig({ ...base, SYNTRA_HYBRID_BASE_URL: "https://example.com/v1" }), /ALLOW_REMOTE=true/);
  assert.equal(getHybridModelConfig({ ...base, SYNTRA_HYBRID_BASE_URL: "https://example.com/v1", SYNTRA_HYBRID_ALLOW_REMOTE: "true" }).allowRemote, true);
  assert.throws(() => getHybridModelConfig({ ...base, SYNTRA_HYBRID_BASE_URL: "https://user:pass@example.com/v1", SYNTRA_HYBRID_ALLOW_REMOTE: "true" }), /must not include credentials/);
});

test("provider status requires the configured model to be visible in /models", async () => {
  await withServer((request, response) => {
    assert.equal(request.url, "/v1/models");
    response.writeHead(200, { "Content-Type": "application/json" });
    response.end(JSON.stringify({ data: [{ id: "other-model" }, { id: "test-model" }] }));
  }, async () => {
    const status = await getHybridModelStatus();
    assert.equal(status.backend, "openai-compatible");
    assert.equal(status.model_available, true);
    assert.equal(status.checkpoint_exists, false);
    assert.equal(status.context_tokens, 8192);
    assert.equal(status.context_count_is_estimate, true);
  });
});

test("provider status fails closed when the configured model is not loaded", async () => {
  await withServer((_request, response) => {
    response.writeHead(200, { "Content-Type": "application/json" });
    response.end(JSON.stringify({ data: [{ id: "different-model" }] }));
  }, async () => {
    await assert.rejects(getHybridModelStatus(), /model ID was not returned/);
  });
});

test("chat completion uses the explicit model and only sends the optional bearer credential", async () => {
  await withServer(async (request, response) => {
    assert.equal(request.method, "POST");
    assert.equal(request.url, "/v1/chat/completions");
    assert.equal(request.headers.authorization, "Bearer unit-test-secret");
    let raw = "";
    for await (const chunk of request) raw += chunk;
    const payload = JSON.parse(raw);
    assert.equal(payload.model, "test-model");
    assert.equal(payload.stream, false);
    assert.equal(payload.messages[0].content, "Implement a real function.");
    response.writeHead(200, { "Content-Type": "application/json" });
    response.end(JSON.stringify({
      choices: [{ message: { content: "function actual() { return true; }" }, finish_reason: "stop" }],
      usage: { prompt_tokens: 7, completion_tokens: 9, total_tokens: 16 }
    }));
  }, async () => {
    process.env.SYNTRA_HYBRID_API_KEY = "unit-test-secret";
    const result = await generateHybridCompletion({ prompt: "Implement a real function.", max_new_tokens: 128 });
    assert.equal(result.text, "function actual() { return true; }");
    assert.equal(result.model_name, "test-model");
    assert.equal(result.usage.total_tokens, 16);
    assert.equal(result.context_count_is_estimate, true);
  });
});

test("prompt overflow is rejected before the provider receives a completion request", async () => {
  let completions = 0;
  await withServer((request, response) => {
    if (request.url === "/v1/chat/completions") completions += 1;
    response.writeHead(200, { "Content-Type": "application/json" });
    response.end(JSON.stringify({ choices: [{ message: { content: "unexpected" } }] }));
  }, async () => {
    process.env.SYNTRA_HYBRID_CONTEXT_TOKENS = "256";
    await assert.rejects(
      generateHybridCompletion({ prompt: "x".repeat(256), max_new_tokens: 32 }),
      /leaves no room/
    );
    assert.equal(completions, 0);
  });
});

test("UTF-8 byte budget is explicit and is not described as exact tokenization", () => {
  const estimate = estimateHybridTokenCount("A😀");
  assert.equal(estimate.token_count, Buffer.byteLength("A😀", "utf8"));
  assert.equal(estimate.estimated, true);
  assert.match(estimate.tokenizer, /budget/);
  assert.match(estimate.note, /not the provider model's exact tokenization/);
});

test("redirect responses are not followed and provider response content is not echoed", async () => {
  await withServer((_request, response) => {
    response.writeHead(302, { Location: "http://127.0.0.1:1/secret" });
    response.end("provider-secret-body");
  }, async () => {
    await assert.rejects(getHybridModelStatus(), /could not be reached/);
  });
});

test("hybrid streaming parses real server-sent events and forwards actual deltas", async () => {
  await withServer((request, response) => {
    assert.equal(request.url, "/v1/chat/completions");
    response.writeHead(200, { "Content-Type": "text/event-stream" });
    response.write('data: {"choices":[{"delta":{"content":"const value = "},"finish_reason":null}]}\n\n');
    response.write('data: {"choices":[{"delta":{"content":"42;"},"finish_reason":null}]}\n\n');
    response.end('data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":4,"completion_tokens":3,"total_tokens":7}}\n\n');
  }, async () => {
    const deltas = [];
    const result = await generateHybridCompletion(
      { prompt: "Write a constant.", max_new_tokens: 32 },
      { onToken: (delta, fullText) => deltas.push({ delta, fullText }) }
    );
    assert.equal(result.text, "const value = 42;");
    assert.deepEqual(deltas, [
      { delta: "const value = ", fullText: "const value = " },
      { delta: "42;", fullText: "const value = 42;" }
    ]);
    assert.equal(result.usage.total_tokens, 7);
    assert.equal(result.finish_reason, "stop");
  });
});

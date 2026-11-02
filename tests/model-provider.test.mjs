import assert from "node:assert/strict";
import test from "node:test";

import {
  chooseLocalModel,
  detectLocalModel,
  getProviderStatus,
  describeLocalModelCapabilities,
  handleLocalModelRequest,
  getGeminiProviderStatus,
  handleGeminiModelRequest
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
    ["workspace.search", "workspace.read", "workspace.symbols.search", "workspace.memory.search", "workspace.semantic.search"]
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
  assert.equal(capturedBody.format.properties.slices.maxItems, 8);
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
  assert.match(capturedBody.messages[0].content, /independent patch reviewer/);
  assert.equal(JSON.parse(result.text).verdict, "accept");
});

test("Gemini provider requires repository-content consent and uses server-side API keys", async () => {
  const previous = {
    key: process.env.SYNTRA_GEMINI_API_KEY,
    model: process.env.SYNTRA_GEMINI_MODEL,
    allow: process.env.SYNTRA_GEMINI_ALLOW_REPOSITORY_CONTENT
  };
  process.env.SYNTRA_GEMINI_API_KEY = "test-key";
  process.env.SYNTRA_GEMINI_MODEL = "gemini-3.8-flash";
  delete process.env.SYNTRA_GEMINI_ALLOW_REPOSITORY_CONTENT;
  try {
    const status = getGeminiProviderStatus();
    assert.equal(status.provider, "gemini-api");
    assert.equal(status.local_only, false);
    assert.equal(status.context_tokens, 1_048_576);
    assert.equal(status.capabilities.tokenCountIsEstimate, false);
    assert.equal(status.capabilities.incrementalTokenStreaming, true);
    assert.equal(status.capabilities.maxOutputTokens, 65536);
    assert.equal(status.capabilities.continuationTokens, false);
    await assert.rejects(
      () => handleGeminiModelRequest("POST", "/model/generate", {
        prompt: "You are SYNTRA's repository-aware software engineering agent. Return one JSON patch."
      }, { fetchImpl: async () => { throw new Error("must not be called"); } }),
      /repository inference is disabled/
    );
    await assert.rejects(
      () => handleGeminiModelRequest("POST", "/model/tokenize", {
        text: "SYNTRA_READ_ONLY_TOOL_PLAN: repository file contents and selected excerpts"
      }, { fetchImpl: async () => { throw new Error("must not be called"); } }),
      /repository tokenization is disabled/
    );
    process.env.SYNTRA_GEMINI_ALLOW_REPOSITORY_CONTENT = "1";
    let requestUrl = "", requestOptions = null;
    const received = [];
    const result = await handleGeminiModelRequest("POST", "/model/generate", {
      prompt: "SYNTRA_READ_ONLY_TOOL_PLAN Return JSON.", max_new_tokens: 200
    }, {
      fetchImpl: async (url, options) => {
        requestUrl = url; requestOptions = options;
        const events = [
          { candidates: [{ content: { parts: [{ text: '{"calls":' }] } }] },
          { candidates: [{ content: { parts: [{ text: '[]}' }] }, finishReason: "STOP" }],
            usageMetadata: { promptTokenCount: 50, candidatesTokenCount: 8 } }
        ];
        const stream = new ReadableStream({
          start(controller) {
            for (const event of events) controller.enqueue(new TextEncoder().encode("data: " + JSON.stringify(event) + "\n\n"));
            controller.close();
          }
        });
        return new Response(stream, { status: 200, headers: { "Content-Type": "text/event-stream" } });
      },
      onToken: (delta, fullText) => received.push({ delta, fullText }), timeoutMs: 1000
    });
    assert.match(requestUrl, /models\/gemini-3\.8-flash:streamGenerateContent\?alt=sse$/);
    assert.equal(requestOptions.headers["x-goog-api-key"], "test-key");
    assert.equal(requestOptions.headers.Authorization, undefined);
    assert.equal(JSON.parse(requestOptions.body).generationConfig.responseMimeType, "application/json");
    assert.equal(result.provider, "gemini-api");
    assert.equal(result.finish_reason, "STOP");
    assert.equal(result.prompt_tokens, 50);
    assert.equal(result.generated_tokens, 8);
    assert.deepEqual(received, [
      { delta: '{"calls":', fullText: '{"calls":' },
      { delta: '[]}', fullText: '{"calls":[]}' }
    ]);
    await assert.rejects(
      () => handleGeminiModelRequest("POST", "/model/generate", { prompt: "Return one short sentence." }, {
        fetchImpl: async () => {
          const partial = new ReadableStream({
            start(controller) {
              controller.enqueue(new TextEncoder().encode(
                'data: {"candidates":[{"content":{"parts":[{"text":"partial output"}]}}]}\n\n'
              ));
              controller.close();
            }
          });
          return new Response(partial, { status: 200, headers: { "Content-Type": "text/event-stream" } });
        },
        onToken: () => {}, timeoutMs: 1000
      }),
      error => error.code === "STREAM_INTERRUPTED" && /without a terminal finishReason/.test(error.message)
    );
    await assert.rejects(
      () => handleGeminiModelRequest("POST", "/model/generate", { prompt: "Return a long answer." }, {
        fetchImpl: async () => {
          const continued = new ReadableStream({
            start(controller) {
              controller.enqueue(new TextEncoder().encode(
                'data: {"candidates":[{"content":{"parts":[{"text":"partial"}]},"finishReason":"CONTINUATION"}]}\n\n'
              ));
              controller.close();
            }
          });
          return new Response(continued, { status: 200, headers: { "Content-Type": "text/event-stream" } });
        },
        onToken: () => {}, timeoutMs: 1000
      }),
      error => error.code === "UNSUPPORTED_CONTINUATION" && /does not support long decode/.test(error.message)
    );
    const tokenCount = await handleGeminiModelRequest("POST", "/model/tokenize", { text: "count this prompt" }, {
      fetchImpl: async (url, options) => {
        assert.match(url, /models\/gemini-3\.8-flash:countTokens$/);
        assert.equal(options.headers["x-goog-api-key"], "test-key");
        return new Response(JSON.stringify({ totalTokens: 17 }), { status: 200 });
      }
    });
    assert.equal(tokenCount.token_count, 17);
    assert.equal(tokenCount.is_estimate, false);
    assert.equal(tokenCount.tokenizer, "gemini-api-countTokens");
  } finally {
    for (const [key, value] of Object.entries({
      SYNTRA_GEMINI_API_KEY: previous.key,
      SYNTRA_GEMINI_MODEL: previous.model,
      SYNTRA_GEMINI_ALLOW_REPOSITORY_CONTENT: previous.allow
    })) {
      if (value === undefined) delete process.env[key]; else process.env[key] = value;
    }
  }
});

test("local model capability metadata distinguishes heuristic classification from benchmark evidence", () => {
  const profile = describeLocalModelCapabilities({
    model: "qwen2.5-coder:14b", contextTokens: 8192,
    selection: "automatic", selectionScore: 110
  });
  assert.equal(profile.codingSpecialized, true);
  assert.equal(profile.contextWindowTokens, 8192);
  assert.equal(profile.selectionMethod, "heuristic_local_model_ranking");
  assert.equal(profile.benchmarkStatus, "not_measured");
  assert.match(profile.benchmarkWarning, /not evidence of engineering task accuracy/);
  const explicit = describeLocalModelCapabilities({
    model: "llama3.2", contextTokens: 4096, selection: "explicit", selectionScore: 75
  });
  assert.equal(explicit.selectionMethod, "user_selected");
  assert.equal(explicit.codingSpecialized, false);
  const glmFlash = describeLocalModelCapabilities({
    model: "glm-4.7-flash", contextTokens: 8192, selection: "automatic", selectionScore: 132
  });
  assert.equal(glmFlash.codingSpecialized, true);
});

test("Gemini long decode resumes streaming and non-streaming responses with the latest continuation token", async () => {
  const previous = {
    key: process.env.SYNTRA_GEMINI_API_KEY,
    model: process.env.SYNTRA_GEMINI_MODEL,
    allow: process.env.SYNTRA_GEMINI_ALLOW_REPOSITORY_CONTENT,
    maxOutput: process.env.SYNTRA_GEMINI_MAX_OUTPUT_TOKENS
  };
  process.env.SYNTRA_GEMINI_API_KEY = "test-key";
  process.env.SYNTRA_GEMINI_MODEL = "gemini-4-argon";
  process.env.SYNTRA_GEMINI_ALLOW_REPOSITORY_CONTENT = "1";
  process.env.SYNTRA_GEMINI_MAX_OUTPUT_TOKENS = "131072";
  const makeSse = events => {
    const stream = new ReadableStream({
      start(controller) {
        for (const event of events) {
          controller.enqueue(new TextEncoder().encode("data: " + JSON.stringify(event) + "\n\n"));
        }
        controller.close();
      }
    });
    return new Response(stream, { status: 200, headers: { "Content-Type": "text/event-stream" } });
  };
  try {
    assert.equal(getGeminiProviderStatus().capabilities.maxOutputTokens, 131072);
    let calls = [];
    const deltas = [];
    const streaming = await handleGeminiModelRequest("POST", "/model/generate", {
      prompt: "Produce a very long source file.", max_new_tokens: 70000
    }, {
      fetchImpl: async (url, options) => {
        const request = JSON.parse(options.body);
        calls.push({ url, request });
        if (calls.length === 1) return makeSse([{
          candidates: [{
            content: { parts: [{ text: "first-segment-" }] },
            finishReason: "CONTINUATION",
            continuationToken: "opaque-token-first-hop"
          }],
          usageMetadata: { promptTokenCount: 40, candidatesTokenCount: 65_536 }
        }]);
        return makeSse([{
          candidates: [{
            content: { parts: [{ text: "second-segment" }] },
            finishReason: "STOP"
          }],
          usageMetadata: { promptTokenCount: 40, candidatesTokenCount: 100 }
        }]);
      },
      onToken: (delta, fullText) => deltas.push({ delta, fullText }),
      timeoutMs: 1000
    });
    assert.equal(calls.length, 2);
    assert.ok(calls.every(call => call.url.includes("/v1beta/models/gemini-4-argon:streamGenerateContent")));
    assert.equal(calls[0].request.generationConfig.maxOutputTokens, 70000);
    assert.equal(calls[1].request.generationConfig.maxOutputTokens, 70000);
    assert.equal(calls[1].request.continuationToken, "opaque-token-first-hop");
    assert.deepEqual(calls[1].request.contents, calls[0].request.contents,
      "continuation must resend the original prompt unchanged, not append partial output");
    assert.equal(streaming.text, "first-segment-second-segment");
    assert.equal(streaming.finish_reason, "STOP");
    assert.equal(streaming.continuation_hops, 1);
    assert.equal(streaming.prompt_tokens, 80);
    assert.equal(streaming.generated_tokens, 65636);
    assert.deepEqual(deltas, [
      { delta: "first-segment-", fullText: "first-segment-" },
      { delta: "second-segment", fullText: "first-segment-second-segment" }
    ]);

    calls = [];
    const nonStreaming = await handleGeminiModelRequest("POST", "/model/generate", {
      prompt: "Generate a long configuration file.", max_new_tokens: 40000
    }, {
      fetchImpl: async (url, options) => {
        const request = JSON.parse(options.body);
        calls.push({ url, request });
        const candidate = calls.length === 1 ? {
          content: { parts: [{ text: "yaml-prefix-" }] },
          finishReason: "CONTINUATION",
          continuationToken: "opaque-token-nonstream"
        } : {
          content: { parts: [{ text: "yaml-suffix" }] },
          finishReason: "STOP"
        };
        return new Response(JSON.stringify({
          candidates: [candidate],
          usageMetadata: { promptTokenCount: 25, candidatesTokenCount: calls.length === 1 ? 32768 : 25 }
        }), { status: 200, headers: { "Content-Type": "application/json" } });
      },
      timeoutMs: 1000
    });
    assert.equal(calls.length, 2);
    assert.ok(calls.every(call => call.url.includes("/v1beta/models/gemini-4-argon:generateContent")));
    assert.equal(calls[1].request.continuationToken, "opaque-token-nonstream");
    assert.deepEqual(calls[1].request.contents, calls[0].request.contents);
    assert.equal(nonStreaming.text, "yaml-prefix-yaml-suffix");
    assert.equal(nonStreaming.finish_reason, "STOP");
    assert.equal(nonStreaming.continuation_hops, 1);
  } finally {
    for (const [key, value] of Object.entries({
      SYNTRA_GEMINI_API_KEY: previous.key,
      SYNTRA_GEMINI_MODEL: previous.model,
      SYNTRA_GEMINI_ALLOW_REPOSITORY_CONTENT: previous.allow,
      SYNTRA_GEMINI_MAX_OUTPUT_TOKENS: previous.maxOutput
    })) {
      if (value === undefined) delete process.env[key]; else process.env[key] = value;
    }
  }
});

test("Gemini continuation rejects missing tokens, repeated tokens, and excessive hop chains", async () => {
  const previous = {
    key: process.env.SYNTRA_GEMINI_API_KEY,
    model: process.env.SYNTRA_GEMINI_MODEL,
    allow: process.env.SYNTRA_GEMINI_ALLOW_REPOSITORY_CONTENT,
    maxOutput: process.env.SYNTRA_GEMINI_MAX_OUTPUT_TOKENS
  };
  process.env.SYNTRA_GEMINI_API_KEY = "test-key";
  process.env.SYNTRA_GEMINI_MODEL = "gemini-4-argon";
  process.env.SYNTRA_GEMINI_ALLOW_REPOSITORY_CONTENT = "1";
  process.env.SYNTRA_GEMINI_MAX_OUTPUT_TOKENS = "131072";
  const response = (candidate) => new Response(JSON.stringify({
    candidates: [candidate], usageMetadata: { promptTokenCount: 1, candidatesTokenCount: 1 }
  }), { status: 200 });
  try {
    await assert.rejects(
      () => handleGeminiModelRequest("POST", "/model/generate", {
        prompt: "Very long output", max_new_tokens: 70000
      }, { fetchImpl: async () => response({ content: { parts: [{ text: "partial" }] }, finishReason: "CONTINUATION" }) }),
      error => error.code === "MISSING_CONTINUATION_TOKEN"
    );
    let calls = 0;
    await assert.rejects(
      () => handleGeminiModelRequest("POST", "/model/generate", {
        prompt: "Very long output", max_new_tokens: 70000
      }, {
        fetchImpl: async (_url, options) => {
          calls += 1;
          const request = JSON.parse(options.body);
          const token = request.continuationToken || "same-token-repeat";
          return response({
            content: { parts: [{ text: "segment" }] },
            finishReason: "CONTINUATION",
            continuationToken: token
          });
        }
      }),
      error => error.code === "REPEATED_CONTINUATION_TOKEN"
    );
    assert.equal(calls, 2);
  } finally {
    for (const [key, value] of Object.entries({
      SYNTRA_GEMINI_API_KEY: previous.key,
      SYNTRA_GEMINI_MODEL: previous.model,
      SYNTRA_GEMINI_ALLOW_REPOSITORY_CONTENT: previous.allow,
      SYNTRA_GEMINI_MAX_OUTPUT_TOKENS: previous.maxOutput
    })) {
      if (value === undefined) delete process.env[key]; else process.env[key] = value;
    }
  }
});

test("Gemini long streaming resumes from the latest checkpoint and drops speculative text after disconnect", async () => {
  const previous = {
    key: process.env.SYNTRA_GEMINI_API_KEY,
    model: process.env.SYNTRA_GEMINI_MODEL,
    allow: process.env.SYNTRA_GEMINI_ALLOW_REPOSITORY_CONTENT,
    maxOutput: process.env.SYNTRA_GEMINI_MAX_OUTPUT_TOKENS
  };
  process.env.SYNTRA_GEMINI_API_KEY = "test-key";
  process.env.SYNTRA_GEMINI_MODEL = "gemini-4-argon";
  process.env.SYNTRA_GEMINI_ALLOW_REPOSITORY_CONTENT = "1";
  process.env.SYNTRA_GEMINI_MAX_OUTPUT_TOKENS = "131072";
  const streamResponse = (events, interrupt = false) => {
    let index = 0;
    const stream = new ReadableStream({
      pull(controller) {
        if (index < events.length) {
          const event = events[index++];
          controller.enqueue(new TextEncoder().encode("data: " + JSON.stringify(event) + "\n\n"));
        } else if (interrupt) {
          controller.error(new Error("simulated socket interruption"));
        } else {
          controller.close();
        }
      }
    });
    return new Response(stream, { status: 200, headers: { "Content-Type": "text/event-stream" } });
  };
  try {
    const calls = [];
    const received = [];
    const result = await handleGeminiModelRequest("POST", "/model/generate", {
      prompt: "Generate a very large source artifact.", max_new_tokens: 70000
    }, {
      fetchImpl: async (url, options) => {
        const request = JSON.parse(options.body);
        calls.push({ url, request });
        if (calls.length === 1) {
          return streamResponse([
            { candidates: [{ content: { parts: [{ text: "committed-prefix-" }] }, continuationToken: "checkpoint-token-1" }] },
            { candidates: [{ content: { parts: [{ text: "must-be-discarded" }] } }] }
          ], true);
        }
        return streamResponse([
          { candidates: [{ content: { parts: [{ text: "resumed-suffix" }] }, finishReason: "STOP" }] }
        ]);
      },
      onToken: (delta, fullText) => received.push({ delta, fullText }),
      timeoutMs: 1000
    });
    assert.equal(calls.length, 2);
    assert.equal(calls[1].request.continuationToken, "checkpoint-token-1");
    assert.deepEqual(calls[1].request.contents, calls[0].request.contents);
    assert.equal(result.text, "committed-prefix-resumed-suffix");
    assert.doesNotMatch(result.text, /must-be-discarded/);
    assert.deepEqual(received, [
      { delta: "committed-prefix-", fullText: "committed-prefix-" },
      { delta: "resumed-suffix", fullText: "committed-prefix-resumed-suffix" }
    ]);
    assert.equal(result.finish_reason, "STOP");
    assert.equal(result.continuation_hops, 1);
  } finally {
    for (const [key, value] of Object.entries({
      SYNTRA_GEMINI_API_KEY: previous.key,
      SYNTRA_GEMINI_MODEL: previous.model,
      SYNTRA_GEMINI_ALLOW_REPOSITORY_CONTENT: previous.allow,
      SYNTRA_GEMINI_MAX_OUTPUT_TOKENS: previous.maxOutput
    })) {
      if (value === undefined) delete process.env[key]; else process.env[key] = value;
    }
  }
});

test("Gemini long streaming retries once from the original request when interrupted before its first checkpoint", async () => {
  const previous = {
    key: process.env.SYNTRA_GEMINI_API_KEY,
    model: process.env.SYNTRA_GEMINI_MODEL,
    allow: process.env.SYNTRA_GEMINI_ALLOW_REPOSITORY_CONTENT,
    maxOutput: process.env.SYNTRA_GEMINI_MAX_OUTPUT_TOKENS
  };
  process.env.SYNTRA_GEMINI_API_KEY = "test-key";
  process.env.SYNTRA_GEMINI_MODEL = "gemini-4-argon";
  process.env.SYNTRA_GEMINI_ALLOW_REPOSITORY_CONTENT = "1";
  process.env.SYNTRA_GEMINI_MAX_OUTPUT_TOKENS = "131072";
  try {
    let calls = 0;
    const output = await handleGeminiModelRequest("POST", "/model/generate", {
      prompt: "Generate a large artifact.", max_new_tokens: 70000
    }, {
      fetchImpl: async (_url, options) => {
        calls += 1;
        const request = JSON.parse(options.body);
        assert.equal(request.continuationToken, undefined,
          "a stream with no checkpoint must be retried from its original prompt");
        let delivered = false;
        const stream = new ReadableStream({
          pull(controller) {
            if (!delivered) {
              delivered = true;
              controller.enqueue(new TextEncoder().encode(
                "data: " + JSON.stringify({
                  candidates: [{
                    content: { parts: [{ text: calls === 1 ? "uncommitted-first-attempt" : "clean-retry-result" }] },
                    ...(calls === 1 ? {} : { finishReason: "STOP" })
                  }]
                }) + "\n\n"
              ));
            } else if (calls === 1) {
              controller.error(new Error("disconnect before checkpoint"));
            } else {
              controller.close();
            }
          }
        });
        return new Response(stream, { status: 200, headers: { "Content-Type": "text/event-stream" } });
      },
      onToken: () => {},
      timeoutMs: 1000
    });
    assert.equal(calls, 2);
    assert.equal(output.text, "clean-retry-result");
  } finally {
    for (const [key, value] of Object.entries({
      SYNTRA_GEMINI_API_KEY: previous.key,
      SYNTRA_GEMINI_MODEL: previous.model,
      SYNTRA_GEMINI_ALLOW_REPOSITORY_CONTENT: previous.allow,
      SYNTRA_GEMINI_MAX_OUTPUT_TOKENS: previous.maxOutput
    })) {
      if (value === undefined) delete process.env[key]; else process.env[key] = value;
    }
  }
});

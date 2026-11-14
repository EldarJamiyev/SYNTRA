const PROVIDER_NAME = "openai-compatible";
const MAX_CONTEXT_TOKENS = 1_000_000;

function readBoolean(value) {
  return String(value || "").trim().toLowerCase() === "true";
}

function isLoopback(hostname) {
  const host = String(hostname || "").toLowerCase().replace(/^\[|\]$/g, "");
  return host === "localhost" || host === "::1" || host === "127.0.0.1" || /^127\./.test(host);
}

export function isHybridModelEnabled(env = process.env) {
  return String(env.SYNTRA_MODEL_PROVIDER || "native").trim().toLowerCase() === PROVIDER_NAME;
}

export function getHybridModelConfig(env = process.env) {
  const provider = String(env.SYNTRA_MODEL_PROVIDER || "native").trim().toLowerCase();
  if (provider === "native") return null;
  if (provider !== PROVIDER_NAME) {
    throw new Error("SYNTRA_MODEL_PROVIDER must be 'native' or the explicitly opt-in 'openai-compatible'.");
  }

  const rawBaseUrl = String(env.SYNTRA_HYBRID_BASE_URL || "").trim();
  const model = String(env.SYNTRA_HYBRID_MODEL || "").trim();
  const contextText = String(env.SYNTRA_HYBRID_CONTEXT_TOKENS || "").trim();
  if (!rawBaseUrl || !model || !contextText) {
    throw new Error("Hybrid mode requires SYNTRA_HYBRID_BASE_URL, SYNTRA_HYBRID_MODEL, and SYNTRA_HYBRID_CONTEXT_TOKENS. No provider or model is selected automatically.");
  }
  if (model.length > 256 || /[\r\n]/.test(model)) {
    throw new Error("SYNTRA_HYBRID_MODEL must be a model identifier of at most 256 characters.");
  }
  const contextTokens = Number(contextText);
  if (!/^\d+$/.test(contextText) || !Number.isSafeInteger(contextTokens) || contextTokens < 256 || contextTokens > MAX_CONTEXT_TOKENS) {
    throw new Error("SYNTRA_HYBRID_CONTEXT_TOKENS must be an integer from 256 to 1,000,000, matching the model's configured context.");
  }

  let baseUrl;
  try { baseUrl = new URL(rawBaseUrl); }
  catch { throw new Error("SYNTRA_HYBRID_BASE_URL must be a valid absolute HTTP(S) URL."); }
  if (!["http:", "https:"].includes(baseUrl.protocol) || baseUrl.username || baseUrl.password || baseUrl.search || baseUrl.hash) {
    throw new Error("The hybrid base URL must use HTTP(S) and must not include credentials, query parameters, or a fragment.");
  }
  const allowRemote = readBoolean(env.SYNTRA_HYBRID_ALLOW_REMOTE);
  if (baseUrl.protocol === "http:" && !isLoopback(baseUrl.hostname)) {
    throw new Error("Hybrid HTTP endpoints are restricted to loopback. Remote endpoints must use HTTPS.");
  }
  if (!isLoopback(baseUrl.hostname) && (baseUrl.protocol !== "https:" || !allowRemote)) {
    throw new Error("Remote hybrid endpoints require HTTPS and SYNTRA_HYBRID_ALLOW_REMOTE=true. Local loopback endpoints need no remote opt-in.");
  }
  if (baseUrl.pathname.endsWith("/")) baseUrl.pathname = baseUrl.pathname.replace(/\/+$/, "");
  if (!baseUrl.pathname) baseUrl.pathname = "";

  const requestedOutput = Number(env.SYNTRA_HYBRID_MAX_OUTPUT_TOKENS || 2048);
  const maxOutputTokens = Number.isSafeInteger(requestedOutput) && requestedOutput > 0
    ? Math.min(requestedOutput, contextTokens - 1, 32768)
    : Math.min(2048, contextTokens - 1);
  return Object.freeze({
    provider: PROVIDER_NAME,
    baseUrl: baseUrl.toString().replace(/\/$/, ""),
    model,
    contextTokens,
    maxOutputTokens,
    apiKey: String(env.SYNTRA_HYBRID_API_KEY || "").trim(),
    allowRemote
  });
}

function endpoint(config, suffix) {
  return config.baseUrl + suffix;
}

function requestSignal(signal, timeoutMs) {
  const timeout = AbortSignal.timeout(timeoutMs);
  return signal ? AbortSignal.any([signal, timeout]) : timeout;
}

async function providerFetch(config, route, { method = "GET", body, signal, timeoutMs = 15000 } = {}) {
  const headers = { Accept: "application/json" };
  if (config.apiKey) headers.Authorization = "Bearer " + config.apiKey;
  if (body !== undefined) headers["Content-Type"] = "application/json";
  let response;
  try {
    response = await fetch(endpoint(config, route), {
      method,
      headers,
      ...(body === undefined ? {} : { body: JSON.stringify(body) }),
      redirect: "error",
      signal: requestSignal(signal, timeoutMs)
    });
  } catch (error) {
    if (signal?.aborted || error?.name === "AbortError") throw error;
    if (error?.name === "TimeoutError") throw new Error("The configured hybrid model endpoint timed out.");
    throw new Error("The configured hybrid model endpoint could not be reached. Verify its URL and service status.");
  }
  if (!response.ok) {
    // Do not echo provider response bodies: compatible servers sometimes include
    // request headers, local paths, or other sensitive diagnostic details.
    throw new Error("The configured hybrid model endpoint returned HTTP " + response.status + " for " + route + ".");
  }
  return response;
}

function extractContent(content) {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content.map(part => typeof part?.text === "string" ? part.text : "").join("");
}

/**
 * Provider tokenizers are not exposed through the OpenAI-compatible contract.
 * UTF-8 bytes are a deliberately conservative budget proxy, not a token count.
 * It must never be displayed or recorded as exact provider tokenization.
 */
export function estimateHybridTokenCount(text) {
  if (typeof text !== "string") throw new TypeError("Token estimate input must be a string.");
  return {
    token_count: Buffer.byteLength(text, "utf8"),
    tokenizer: "provider-utf8-byte-budget",
    estimated: true,
    note: "UTF-8 byte budget proxy; not the provider model's exact tokenization."
  };
}

export async function getHybridModelStatus({ env = process.env, signal } = {}) {
  const config = getHybridModelConfig(env);
  if (!config) throw new Error("The hybrid model provider is not enabled.");
  const response = await providerFetch(config, "/models", { signal, timeoutMs: 10000 });
  let payload;
  try { payload = await response.json(); }
  catch { throw new Error("The hybrid endpoint returned invalid JSON from its /models route."); }
  const models = Array.isArray(payload?.data) ? payload.data : [];
  const available = models.some(item => item && item.id === config.model);
  if (!available) {
    throw new Error("The configured model ID was not returned by the hybrid endpoint's /models route. Check the loaded model and SYNTRA_HYBRID_MODEL.");
  }
  return {
    backend: PROVIDER_NAME,
    provider: PROVIDER_NAME,
    model_name: config.model,
    model_available: true,
    checkpoint_exists: false,
    pretrained_weights: true,
    context_tokens: config.contextTokens,
    context_window_tokens: config.contextTokens,
    max_output_tokens: config.maxOutputTokens,
    tokenizer: "provider-utf8-byte-budget",
    context_count_is_estimate: true,
    native_training_available: false,
    embeddings_available: false
  };
}

export async function generateHybridCompletion(body, { env = process.env, signal, timeoutMs = 300000, onToken } = {}) {
  const config = getHybridModelConfig(env);
  if (!config) throw new Error("The hybrid model provider is not enabled.");
  if (!body || typeof body.prompt !== "string" || !body.prompt.trim()) {
    throw new Error("Hybrid generation requires a non-empty prompt.");
  }
  const promptBudget = estimateHybridTokenCount(body.prompt).token_count;
  const requestedMax = Number(body.max_new_tokens || config.maxOutputTokens);
  const maxTokens = Number.isSafeInteger(requestedMax) && requestedMax > 0
    ? Math.min(requestedMax, config.maxOutputTokens, config.contextTokens - promptBudget - 1)
    : Math.min(config.maxOutputTokens, config.contextTokens - promptBudget - 1);
  if (maxTokens < 1) {
    throw new Error("The prompt budget leaves no room for generation in the configured hybrid context. Shorten the request or configure the model's actual context size.");
  }
  const temperatureValue = Number(body.temperature ?? 0.2);
  const temperature = Number.isFinite(temperatureValue) ? Math.max(0, Math.min(2, temperatureValue)) : 0.2;
  const streaming = typeof onToken === "function";
  const response = await providerFetch(config, "/chat/completions", {
    method: "POST",
    signal,
    timeoutMs,
    body: {
      model: config.model,
      messages: [{ role: "user", content: body.prompt }],
      max_tokens: maxTokens,
      temperature,
      stream: streaming
    }
  });

  let text = "";
  let usage = null;
  let finishReason = null;
  if (streaming) {
    if (!response.body) throw new Error("The hybrid endpoint did not return a readable streaming response.");
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    try {
      while (true) {
        if (signal?.aborted) throw Object.assign(new Error("Generation was cancelled."), { name: "AbortError" });
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const blocks = buffer.split(/\r?\n\r?\n/);
        buffer = blocks.pop() || "";
        for (const block of blocks) {
          for (const line of block.split(/\r?\n/)) {
            if (!line.startsWith("data:")) continue;
            const raw = line.slice(5).trim();
            if (!raw || raw === "[DONE]") continue;
            let event;
            try { event = JSON.parse(raw); } catch { continue; }
            const choice = event.choices?.[0];
            const delta = extractContent(choice?.delta?.content);
            if (delta) {
              text += delta;
              onToken(delta, text);
            }
            if (choice?.finish_reason) finishReason = choice.finish_reason;
            if (event.usage && typeof event.usage === "object") usage = event.usage;
          }
        }
      }
      buffer += decoder.decode();
      for (const line of buffer.split(/\r?\n/)) {
        if (!line.startsWith("data:")) continue;
        const raw = line.slice(5).trim();
        if (!raw || raw === "[DONE]") continue;
        try {
          const event = JSON.parse(raw);
          const choice = event.choices?.[0];
          const delta = extractContent(choice?.delta?.content);
          if (delta) { text += delta; onToken(delta, text); }
          if (choice?.finish_reason) finishReason = choice.finish_reason;
          if (event.usage && typeof event.usage === "object") usage = event.usage;
        } catch { /* Ignore an incomplete final SSE event. */ }
      }
    } finally {
      reader.releaseLock();
    }
  } else {
    let payload;
    try { payload = await response.json(); }
    catch { throw new Error("The hybrid endpoint returned invalid JSON from its completion route."); }
    const choice = payload?.choices?.[0];
    text = extractContent(choice?.message?.content);
    finishReason = choice?.finish_reason || null;
    usage = payload?.usage && typeof payload.usage === "object" ? payload.usage : null;
  }
  if (!text.trim()) throw new Error("The hybrid endpoint completed without returning assistant text.");
  return {
    text,
    is_model_output: true,
    backend: PROVIDER_NAME,
    model_name: config.model,
    finish_reason: finishReason,
    usage: usage ? {
      prompt_tokens: Number.isSafeInteger(usage.prompt_tokens) ? usage.prompt_tokens : null,
      completion_tokens: Number.isSafeInteger(usage.completion_tokens) ? usage.completion_tokens : null,
      total_tokens: Number.isSafeInteger(usage.total_tokens) ? usage.total_tokens : null
    } : null,
    prompt_tokens_estimated: promptBudget,
    tokenizer: "provider-utf8-byte-budget",
    context_count_is_estimate: true
  };
}

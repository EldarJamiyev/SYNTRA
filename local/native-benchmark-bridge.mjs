import { generateBrainText, getBrainStatus, tokenizeBrainText } from "./brain-runtime.mjs";

/** Native-only bridge shared by the standalone benchmark CLI and epoch evaluation. */
export function createNativeBenchmarkModelRequest() {
  return async (method, route, body = {}, timeoutMs = undefined, options = {}) => {
    if (method === "GET" && route === "/model/status") {
      return getBrainStatus();
    }
    const timeoutSignal = timeoutMs > 0 ? AbortSignal.timeout(timeoutMs) : null;
    const signal = options.signal && timeoutSignal
      ? AbortSignal.any([options.signal, timeoutSignal])
      : options.signal || timeoutSignal || undefined;

    if (method === "POST" && route === "/model/tokenize") {
      if (typeof body?.text !== "string") throw new TypeError("Tokenizer request requires a text string.");
      return tokenizeBrainText(body.text);
    }
    if (method === "POST" && route === "/model/generate") {
      return generateBrainText({
        prompt: body?.prompt,
        maxNewTokens: body?.max_new_tokens ?? 256,
        temperature: body?.temperature ?? 0.2,
        topK: body?.top_k ?? 40,
        topP: body?.top_p ?? 1,
        repetitionPenalty: body?.repetition_penalty ?? 1.05,
        device: body?.device ?? "auto",
        signal,
      });
    }
    throw new Error("Unsupported native SYNTRA benchmark operation: " + method + " " + route);
  };
}

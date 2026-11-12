const MAX_HISTORY_MESSAGES = 8;
const MAX_HISTORY_CHARS = 16_000;

function selectRecentHistory(history) {
  if (!Array.isArray(history)) return { messages: [], dropped: 0 };

  const candidates = history
    .filter(item => item && (item.role === "user" || item.role === "assistant")
      && typeof item.content === "string" && item.content.length > 0)
    .slice(-MAX_HISTORY_MESSAGES)
    .map(item => ({ role: item.role, content: item.content }));

  const originallySelected = candidates.length;
  let totalCharacters = candidates.reduce((total, item) => total + item.content.length, 0);

  // Remove complete oldest turns when history is too large. Never cut a message
  // in half: the remaining context is either preserved whole or excluded whole.
  while (candidates.length && totalCharacters > MAX_HISTORY_CHARS) {
    const removed = candidates.shift();
    totalCharacters -= removed.content.length;
  }

  return {
    messages: candidates,
    dropped: originallySelected - candidates.length
  };
}

function buildPrompt(message, history, workspaceGrounding) {
  return [
    "SYNTRA ASSISTANT",
    "Answer the user's latest message directly and clearly. Do not claim facts or tools you do not have.",
    ...(workspaceGrounding ? [workspaceGrounding] : []),
    ...history.map(item => (item.role === "user" ? "USER: " : "ASSISTANT: ") + item.content),
    "USER: " + message,
    "ASSISTANT:"
  ].join("\n\n");
}

function readTokenCount(result) {
  const count = result && result.token_count;
  if (!Number.isSafeInteger(count) || count < 0) {
    throw new Error("The native tokenizer returned an invalid token_count.");
  }
  return count;
}

/**
 * Pack chat context against the native tokenizer's measured context budget.
 *
 * The current user message is never truncated. Optional workspace grounding is
 * removed first, then complete oldest history messages are removed. If the
 * complete current message still cannot fit, the caller must reject the turn
 * explicitly rather than sending a silently altered request to the model.
 */
export async function buildChatPromptContext({
  message,
  history = [],
  workspaceGrounding = "",
  contextTokens,
  tokenize
}) {
  if (typeof message !== "string" || !message.trim()) {
    throw new TypeError("A non-empty current user message is required.");
  }
  if (typeof tokenize !== "function") {
    throw new TypeError("The native tokenizer callback is required.");
  }

  const contextWindowTokens = Math.floor(Number(contextTokens));
  if (!Number.isSafeInteger(contextWindowTokens) || contextWindowTokens < 32) {
    throw new RangeError("The native model context window must be at least 32 tokens.");
  }

  const reservedTokens = Math.min(64, Math.max(16, Math.floor(contextWindowTokens / 8)));
  const inputBudgetTokens = contextWindowTokens - reservedTokens;
  const selected = selectRecentHistory(history);
  const retainedHistory = selected.messages.slice();
  let retainedGrounding = typeof workspaceGrounding === "string" ? workspaceGrounding : "";
  let droppedHistoryMessages = selected.dropped;

  const measure = async () => {
    const prompt = buildPrompt(message, retainedHistory, retainedGrounding);
    const tokenResult = await tokenize(prompt);
    return { prompt, tokenCount: readTokenCount(tokenResult) };
  };

  let measured = await measure();

  // Workspace evidence is supplemental, so discard it before conversation history.
  if (measured.tokenCount > inputBudgetTokens && retainedGrounding) {
    retainedGrounding = "";
    measured = await measure();
  }

  // Preserve recent context preferentially, dropping only complete older messages.
  while (measured.tokenCount > inputBudgetTokens && retainedHistory.length) {
    retainedHistory.shift();
    droppedHistoryMessages += 1;
    measured = await measure();
  }

  const fits = measured.tokenCount <= inputBudgetTokens;
  return {
    fits,
    ...(fits ? { prompt: measured.prompt } : {}),
    tokenCount: measured.tokenCount,
    contextWindowTokens,
    inputBudgetTokens,
    reservedTokens,
    historyMessagesUsed: retainedHistory.length,
    historyMessagesDropped: droppedHistoryMessages,
    workspaceGroundingIncluded: Boolean(retainedGrounding)
  };
}

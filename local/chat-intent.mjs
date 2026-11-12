import { normalizeIntentText } from "./intent-language.mjs";

const ACTION = /^(?:hey[, ]*)?(?:please\s+)?(?:(?:can|could|would|will) you(?:\s+please)?\s+)?(?:please\s+)?(?:implement|build|fix|debug|refactor|add|create|update|change|repair|write|remove|optimize|migrate|test|extend|integrate|wire|replace|make|improve|enhance|upgrade|harden|simplify|modernize|redesign|modify|edit|patch)\b/i;
const ACTION_ANYWHERE = /\b(?:implement|build|fix|debug|refactor|add|create|update|change|repair|write|remove|optimize|migrate|test|extend|integrate|wire|replace|make|improve|enhance|upgrade|harden|simplify|modernize|redesign|modify|edit|patch)\b/i;

const SOFTWARE_TARGET = /\b(?:app|website|web\s+app(?:lication)?|web\s+application|desktop\s+application|mobile\s+application|webpage|web\s+page|frontend|front[- ]end|backend|back[- ]end|ui|ux|interface|component|module|function|class|feature|bug|error|test(?:s| suite)?|build|compile|lint|typecheck|api|endpoint|route|server|client|script|package|dependency|dependencies|npm|vite|react|typescript|javascript|python|css|html|sql|database|query|schema|authentication|login|button|form|responsive|accessibility|performance|chat\s+(?:ui|interface|frontend|front[- ]end|app)|code|source|repository|repo|codebase|project files|project code|unit test|integration test|stack trace|exception|design system|dark mode|theme|navigation|routing)\b/i;
const EXPLICIT_REPO_CONTEXT = /\b(?:this|the|current|my)\s+(?:codebase|repository|repo|project files|source code)\b|\b(?:in|inside|within)\s+(?:this|the|current|my)\s+(?:project files|repository|repo|codebase|source code)\b/i;
const CONVERSATIONAL_OUTPUT = /^(?:hey[, ]*)?(?:please\s+)?(?:(?:can|could|would) you(?:\s+please)?\s+)?(?:please\s+)?(?:write|create|draft|compose|prepare|generate)\s+(?:(?:me|us)\s+)?(?:an?\s+)?(?:email|letter|poem|story|song|essay|article|blog post|social post|text message|message|proposal|job application|cover letter|resume|résumé|cv|joke|speech|recipe)\b/i;
const SOURCE_PATH_PATTERN = /\b(?:[A-Za-z0-9_.-]+\/)+[A-Za-z0-9_.-]+\.(?:m?js|cjs|tsx?|jsx?|py|go|rs|json|css|html|md|ya?ml|toml|sql|sh)\b|\b[A-Za-z0-9_-]+\.(?:mjs|cjs|tsx|jsx|py|go|rs)\b/g;
const WORKSPACE_QUERY_SIGNAL = /\b(?:where\s+is|how\s+does|what\s+(?:does|files?|modules?|functions?|classes?)|explain\s+(?:how|the)|trace|find\s+(?:where|all)|in\s+(?:this|the)\s+(?:repo|repository|workspace|codebase|project)|architecture\s+of)\b/i;
const TARGET_FIRST_COMMAND = /^(?:(?:the|a|an|my|this|that)\s+)?(?:app|application|website|web\s+(?:app|application|site)|site|frontend|front[- ]end|backend|back[- ]end|ui|ux|interface|component|module|function|feature|bug|api|endpoint|route|server|client|script|package|database|query|schema|login|button|form|chat|code|source|repository|repo|codebase|project|dark mode|theme)\b[\s\S]{0,80}\b(?:implement|build|fix|debug|refactor|add|create|update|change|repair|write|remove|optimize|migrate|test|extend|integrate|wire|replace|make|improve|enhance|upgrade|harden|simplify|modernize|redesign|modify|edit|patch)\s*[.!?]*$/i;
// Keep identifier grammars deterministic: repeated segments start with a
// delimiter that cannot be consumed by the preceding segment.
const BACKTICK_IDENTIFIER_PATTERN = /`([A-Za-z_$][A-Za-z0-9_$]{2,63})`/g;
const CAMEL_IDENTIFIER_PATTERN = /\b[A-Za-z_$][a-z0-9]+(?:[A-Z][a-z0-9]*)+\b/g;
const SNAKE_IDENTIFIER_PATTERN = /\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b/g;

const CODE_FOLLOW_UP = /^(?:(?:and|also|now|then)\s+)?(?:(?:please|pls|plz)\s+)?(?:continue(?:\s+from there)?|keep going|go on|carry on|proceed(?:\s+with (?:it|the task|the changes|the implementation))?|do (?:it|that)|finish(?: it| the task| the implementation| the changes)?|complete(?: it| the task| the implementation| the changes)?|resume(?: it)?|pick up where you left off|keep working|make (?:it|that) (?:more\s+)?(?:robust|secure|fast(?:er)?|performant|readable|clean(?:er)?|resilient|accessible|responsive|reliable|maintainable|efficient)|fix it|improve it|refine it|harden it|optimize it|update it|change it|add (?:the )?(?:tests?|edge cases|type checks)|include (?:tests?|edge cases)|handle (?:it|that|edge cases)|cover (?:it|those cases|edge cases)|run (?:the )?tests?|verify it|check it)(?:\s+(?:please|now|too))?[.!?]*$/i;
const WORKSPACE_FOLLOW_UP = /^(?:(?:and|also|then)\s+)?(?:(?:can|could|would) you\s+)?(?:explain (?:that|this|it)(?:\s+(?:again|further|in more detail))?|elaborate(?:\s+on)?(?:\s+(?:that|this|it))?(?:\s+further)?|tell me more(?:\s+about\s+(?:that|this|it))?|show me more(?:\s+about\s+(?:that|this|it))?|where (?:is|does) (?:that|this|it)\b.{0,80}|how (?:does|is|was) (?:that|this|it)\b.{0,80}|what about (?:that|this|it)\b.{0,80}|go deeper(?:\s+on\s+(?:that|this|it))?)(?:[.!?]+)?$/i;

function latestPriorUserEntry(history, currentMessage) {
  if (!Array.isArray(history)) return null;
  const current = currentMessage.trim();
  const start = Math.max(0, history.length - 24);
  for (let index = history.length - 1; index >= start; index -= 1) {
    const item = history[index];
    if (!item || item.role !== "user" || typeof item.content !== "string") continue;
    const candidate = item.content.trim();
    if (!candidate || candidate === current) continue;
    if (candidate.length > 12000) return null;
    return { index, content: candidate };
  }
  return null;
}


export function extractTargetPaths(message) {
  if (typeof message !== "string" || !message.trim()) return [];
  const matches = message.slice(0, 4000).match(SOURCE_PATH_PATTERN) || [];
  return [...new Set(matches.map(item => item.replace(/^\/+/, "")))].slice(0, 6);
}

export function extractTargetSymbols(message) {
  if (typeof message !== "string" || !message.trim()) return [];
  const slice = message.slice(0, 4000);
  const matches = [];
  for (const pattern of [
    BACKTICK_IDENTIFIER_PATTERN,
    CAMEL_IDENTIFIER_PATTERN,
    SNAKE_IDENTIFIER_PATTERN
  ]) {
    pattern.lastIndex = 0;
    let match;
    while ((match = pattern.exec(slice)) !== null) {
      const candidate = (match[1] || match[0]).trim();
      if (candidate.length <= 64) matches.push({ index: match.index, candidate });
    }
  }

  matches.sort((left, right) => left.index - right.index);
  const symbols = [];
  for (const { candidate } of matches) {
    if (!candidate.includes("/") && !candidate.includes(".") && !symbols.includes(candidate)) {
      symbols.push(candidate);
      if (symbols.length >= 6) break;
    }
  }
  return symbols;
}

/**
 * Route explicit software-change commands to the repository agent without hijacking
 * ordinary conversation. A broad verb such as "write", "create", or "make" is never
 * enough by itself; prose requests such as emails and poems remain normal chat.
 */
export function isCodingTask(message) {
  if (typeof message !== "string") return false;
  const text = normalizeIntentText(message);
  if (!text || text.length > 12000) return false;

  if (CONVERSATIONAL_OUTPUT.test(text)) return false;
  if (EXPLICIT_REPO_CONTEXT.test(text) && ACTION_ANYWHERE.test(text)) return true;

  const paths = extractTargetPaths(message);
  if (paths.length > 0 && ACTION_ANYWHERE.test(text) && !/^(?:what|how|why|where|explain|describe)\b/i.test(text)) {
    return true;
  }

  if (TARGET_FIRST_COMMAND.test(text)) return true;

  const requestCue = /^(?:please\b|(?:can|could|would)\s+you\b|i\s+(?:want|need)\s+(?:you\s+)?to\b)/i;
  if (requestCue.test(text) && ACTION_ANYWHERE.test(text) && SOFTWARE_TARGET.test(text)) return true;

  const command = text.match(ACTION);
  if (!command) {
    return /^(?:can|could|would) you\b/i.test(text)
      && ACTION_ANYWHERE.test(text)
      && (SOFTWARE_TARGET.test(text.slice(0, 240)) || paths.length > 0);
  }

  const targetWindow = text.slice(command[0].length, command[0].length + 240);
  const taskObject = targetWindow.split(/\b(?:about|regarding|concerning|describing|explaining|discussing)\b/i)[0];
  return SOFTWARE_TARGET.test(taskObject) || extractTargetPaths(taskObject).length > 0;
}

/**
 * Multi-signal cognitive classifier that distinguishes repository mutation tasks,
 * read-only workspace/architecture comprehension queries, and general conversation.
 */
export function classifyChatIntent(message, history = []) {
  if (typeof message !== "string") {
    return { kind: "conversation", confidence: 1, signals: ["non-string"], targetPaths: [], targetSymbols: [], searchQuery: "" };
  }
  const originalText = message.trim();
  const text = normalizeIntentText(originalText);
  if (!text || originalText.length > 12000) {
    return { kind: "conversation", confidence: 1, signals: ["empty-or-overlong"], targetPaths: [], targetSymbols: [], searchQuery: "" };
  }

  const targetPaths = extractTargetPaths(originalText);
  const targetSymbols = extractTargetSymbols(originalText);
  const signals = [];

  if (isCodingTask(originalText)) {
    signals.push("action-verb", "software-target");
    if (targetPaths.length) signals.push("explicit-path");
    if (targetSymbols.length) signals.push("symbol-reference");
    return {
      kind: "coding_task",
      confidence: Math.min(0.98, 0.78 + (targetPaths.length ? 0.1 : 0) + (targetSymbols.length ? 0.08 : 0)),
      signals,
      targetPaths,
      targetSymbols,
      searchQuery: (targetPaths[0] || targetSymbols[0] || originalText.slice(0, 120)).trim()
    };
  }

  const previousEntry = latestPriorUserEntry(history, originalText);
  if (previousEntry) {
    // Classify the preceding user turn against only the history that existed
    // before it. This lets a chain of concise follow-ups inherit the original
    // task without allowing an unrelated later question to revive stale intent.
    const previousIntent = classifyChatIntent(previousEntry.content, history.slice(0, previousEntry.index));
    const inheritedMessages = previousIntent.contextualFollowUp === true
      && Array.isArray(previousIntent.contextMessages)
      ? [...previousIntent.contextMessages, previousEntry.content]
      : [previousEntry.content];
    const rootMessage = previousIntent.contextMessage || previousEntry.content;

    if (previousIntent.kind === "coding_task" && CODE_FOLLOW_UP.test(text)) {
      return {
        kind: "coding_task",
        confidence: Math.min(0.9, Math.max(0.72, previousIntent.confidence - 0.04)),
        signals: ["contextual-follow-up", "prior-coding-task", "continuation-cue"],
        targetPaths: previousIntent.targetPaths,
        targetSymbols: previousIntent.targetSymbols,
        searchQuery: previousIntent.searchQuery,
        contextualFollowUp: true,
        contextMessage: rootMessage,
        contextMessages: inheritedMessages.slice(-8)
      };
    }
    if (previousIntent.kind === "workspace_query" && WORKSPACE_FOLLOW_UP.test(text)) {
      return {
        kind: "workspace_query",
        confidence: Math.min(0.9, previousIntent.confidence),
        signals: ["contextual-follow-up", "prior-workspace-query", "reference-resolution"],
        targetPaths: previousIntent.targetPaths,
        targetSymbols: previousIntent.targetSymbols,
        searchQuery: previousIntent.searchQuery,
        contextualFollowUp: true,
        contextMessage: rootMessage,
        contextMessages: inheritedMessages.slice(-8)
      };
    }
  }

  if (!CONVERSATIONAL_OUTPUT.test(text)) {
    if (targetPaths.length) signals.push("explicit-path");
    if (targetSymbols.length) signals.push("symbol-reference");
    if (WORKSPACE_QUERY_SIGNAL.test(text)) signals.push("workspace-question");
    if (EXPLICIT_REPO_CONTEXT.test(text)) signals.push("repo-context");

    const isWorkspaceQuestion = (targetPaths.length > 0 || targetSymbols.length > 0 || EXPLICIT_REPO_CONTEXT.test(text))
      && (WORKSPACE_QUERY_SIGNAL.test(text) || /\?$/.test(text) || /^(?:explain|describe|summarize|inspect|review|analyze)\b/i.test(text));

    if (isWorkspaceQuestion) {
      const fallbackTerms = (text.toLowerCase().match(/[a-z0-9_.-]{3,}/g) || [])
        .filter(word => !["what", "where", "how", "does", "explain", "describe", "this", "that", "from", "with", "into", "repository", "workspace", "codebase", "project"].includes(word))
        .slice(0, 5)
        .join(" ");
      return {
        kind: "workspace_query",
        confidence: Math.min(0.95, 0.65 + signals.length * 0.1),
        signals,
        targetPaths,
        targetSymbols,
        searchQuery: (targetSymbols[0] || targetPaths[0] || fallbackTerms || originalText.slice(0, 80)).slice(0, 120)
      };
    }
  }

  return {
    kind: "conversation",
    confidence: 0.85,
    signals: signals.length ? signals : ["general-dialogue"],
    targetPaths,
    targetSymbols,
    searchQuery: ""
  };
}



/**
 * Preserve the original task and every bounded follow-up instruction.
 * The caller still performs the normal agent verification and approval workflow.
 */
export function buildFollowUpTaskGoal(message, intent) {
  if (
    typeof message !== "string"
    || intent?.kind !== "coding_task"
    || intent.contextualFollowUp !== true
    || typeof intent.contextMessage !== "string"
    || !intent.contextMessage.trim()
  ) return message;

  const priorMessages = Array.isArray(intent.contextMessages)
    ? intent.contextMessages.filter(item => typeof item === "string" && item.trim()).slice(-8)
    : [intent.contextMessage];
  const original = intent.contextMessage.trim();
  const followUps = priorMessages.filter(item => item.trim() !== original).slice(-6);
  return [
    "CONTINUE THE EXISTING USER-REQUESTED CODING TASK. Inspect the current workspace state before changing files.",
    "The original request and prior follow-up instructions remain binding. Treat the latest follow-up as an addition, not a replacement. Preserve all constraints and verify the cumulative result.",
    "ORIGINAL USER REQUEST:",
    original,
    ...followUps.map((item, index) => "PRIOR FOLLOW-UP " + (index + 1) + ":\n" + item.trim().slice(0, 1200)),
    "LATEST USER FOLLOW-UP:",
    message
  ].join("\n\n");
}

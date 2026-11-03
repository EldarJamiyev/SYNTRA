const ACTION = /^(?:hey[, ]*)?(?:please\s+)?(?:(?:can|could|would|will) you(?:\s+please)?\s+)?(?:please\s+)?(?:implement|build|fix|debug|refactor|add|create|update|change|repair|write|remove|optimize|migrate|test|extend|integrate|wire|replace|make|improve|enhance|upgrade|harden|simplify|modernize|redesign|modify|edit|patch)\b/i;
const ACTION_ANYWHERE = /\b(?:implement|build|fix|debug|refactor|add|create|update|change|repair|write|remove|optimize|migrate|test|extend|integrate|wire|replace|make|improve|enhance|upgrade|harden|simplify|modernize|redesign|modify|edit|patch)\b/i;

const SOFTWARE_TARGET = /\b(?:app|website|web\s+app(?:lication)?|web\s+application|desktop\s+application|mobile\s+application|webpage|web\s+page|frontend|front[- ]end|backend|back[- ]end|ui|ux|interface|component|module|function|class|feature|bug|error|test(?:s| suite)?|build|compile|lint|typecheck|api|endpoint|route|server|client|script|package|dependency|dependencies|npm|vite|react|typescript|javascript|python|css|html|sql|database|query|schema|authentication|login|button|form|responsive|accessibility|performance|chat\s+(?:ui|interface|frontend|front[- ]end|app)|code|source|repository|repo|codebase|project files|project code|unit test|integration test|stack trace|exception|design system|dark mode|theme|navigation|routing)\b/i;
const EXPLICIT_REPO_CONTEXT = /\b(?:this|the|current|my)\s+(?:codebase|repository|repo|project files|source code)\b|\b(?:in|inside|within)\s+(?:this|the|current|my)\s+(?:project files|repository|repo|codebase|source code)\b/i;
const CONVERSATIONAL_OUTPUT = /^(?:hey[, ]*)?(?:please\s+)?(?:(?:can|could|would) you(?:\s+please)?\s+)?(?:please\s+)?(?:write|create|draft|compose|prepare|generate)\s+(?:(?:me|us)\s+)?(?:an?\s+)?(?:email|letter|poem|story|song|essay|article|blog post|social post|text message|message|proposal|job application|cover letter|resume|résumé|cv|joke|speech|recipe)\b/i;
const SOURCE_PATH_PATTERN = /\b(?:[A-Za-z0-9_.-]+\/)+[A-Za-z0-9_.-]+\.(?:m?js|cjs|tsx?|jsx?|py|go|rs|json|css|html|md|ya?ml|toml|sql|sh)\b|\b[A-Za-z0-9_-]+\.(?:mjs|cjs|tsx|jsx|py|go|rs)\b/g;
const WORKSPACE_QUERY_SIGNAL = /\b(?:where\s+is|how\s+does|what\s+(?:does|files?|modules?|functions?|classes?)|explain\s+(?:how|the)|trace|find\s+(?:where|all)|in\s+(?:this|the)\s+(?:repo|repository|workspace|codebase|project)|architecture\s+of)\b/i;
const IDENTIFIER_PATTERN = /`([A-Za-z_$][A-Za-z0-9_$]{2,63})`|\b([A-Za-z_$][a-z0-9]+(?:[A-Z][A-Za-z0-9]+)+|[a-z]+_[a-z0-9_]{2,50})\b/g;

export function extractTargetPaths(message) {
  if (typeof message !== "string" || !message.trim()) return [];
  const matches = message.slice(0, 4000).match(SOURCE_PATH_PATTERN) || [];
  return [...new Set(matches.map(item => item.replace(/^\/+/, "")))].slice(0, 6);
}

export function extractTargetSymbols(message) {
  if (typeof message !== "string" || !message.trim()) return [];
  const symbols = [];
  const slice = message.slice(0, 4000);
  let match;
  IDENTIFIER_PATTERN.lastIndex = 0;
  while ((match = IDENTIFIER_PATTERN.exec(slice)) !== null) {
    const candidate = (match[1] || match[2] || "").trim();
    if (candidate && !candidate.includes("/") && !candidate.includes(".") && !symbols.includes(candidate)) {
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
  const text = message.trim();
  if (!text || text.length > 12000) return false;

  if (CONVERSATIONAL_OUTPUT.test(text)) return false;
  if (EXPLICIT_REPO_CONTEXT.test(text) && ACTION_ANYWHERE.test(text)) return true;

  const paths = extractTargetPaths(text);
  if (paths.length > 0 && ACTION_ANYWHERE.test(text) && !/^(?:what|how|why|where|explain|describe)\b/i.test(text)) {
    return true;
  }

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
  const text = message.trim();
  if (!text || text.length > 12000) {
    return { kind: "conversation", confidence: 1, signals: ["empty-or-overlong"], targetPaths: [], targetSymbols: [], searchQuery: "" };
  }

  const targetPaths = extractTargetPaths(text);
  const targetSymbols = extractTargetSymbols(text);
  const signals = [];

  if (isCodingTask(text)) {
    signals.push("action-verb", "software-target");
    if (targetPaths.length) signals.push("explicit-path");
    if (targetSymbols.length) signals.push("symbol-reference");
    return {
      kind: "coding_task",
      confidence: Math.min(0.98, 0.78 + (targetPaths.length ? 0.1 : 0) + (targetSymbols.length ? 0.08 : 0)),
      signals,
      targetPaths,
      targetSymbols,
      searchQuery: (targetPaths[0] || targetSymbols[0] || text.slice(0, 120)).trim()
    };
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
        searchQuery: (targetSymbols[0] || targetPaths[0] || fallbackTerms || text.slice(0, 80)).slice(0, 120)
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


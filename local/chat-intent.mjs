const ACTION = /^(?:hey[, ]*)?(?:please\s+)?(?:(?:can|could|would) you\s+)?(?:implement|build|fix|debug|refactor|add|create|update|change|repair|write|remove|optimize|migrate|test|extend|integrate|wire|replace|make|improve|enhance|upgrade|harden|simplify|modernize|redesign|modify|edit)\b/i;

const SOFTWARE_TARGET = /\b(?:app(?:lication)?|website|web\s+app|webpage|web\s+page|frontend|front[- ]end|backend|back[- ]end|ui|ux|interface|layout|component|module|function|class|feature|bug|error|test(?:s| suite)?|build|compile|lint|typecheck|api|endpoint|route|server|client|script|package|dependency|dependencies|npm|vite|react|typescript|javascript|python|css|html|sql|database|query|schema|authentication|login|button|form|page|screen|responsive|accessibility|performance|chat\s+(?:ui|interface|frontend|front[- ]end|app)|code|source|repository|repo|codebase|project files|project code|unit test|integration test|stack trace|exception|design system|dark mode|theme|navigation|routing)\b/i;

const ACTION_ANYWHERE = /\b(?:implement|build|fix|debug|refactor|add|create|update|change|repair|write|remove|optimize|migrate|test|extend|integrate|wire|replace|make|improve|enhance|upgrade|harden|simplify|modernize|redesign|modify|edit)\b/i;
const PROJECT_CONTEXT = /\b(?:this|the|current|my)\s+(?:codebase|repository|repo|project files|source code|app|application|website|frontend|front[- ]end|backend|back[- ]end|chat interface|chat frontend|project)\b|\b(?:in|inside|within)\s+(?:this|the|current|my)\s+(?:project|repository|repo|codebase)\b/i;

/**
 * Distinguish conversational requests from commands that should modify a code workspace.
 * Broad verbs alone are intentionally insufficient: "write a poem" and "make me laugh"
 * must remain normal chat, not trigger the repository-editing agent.
 */
export function isCodingTask(message) {
  if (typeof message !== "string") return false;
  const text = message.trim();
  if (!text || text.length > 12000) return false;

  if (PROJECT_CONTEXT.test(text) && ACTION_ANYWHERE.test(text)) return true;

  const command = text.match(ACTION);
  if (!command) return false;
  const targetWindow = text.slice(command[0].length, command[0].length + 240);
  return SOFTWARE_TARGET.test(targetWindow);
}

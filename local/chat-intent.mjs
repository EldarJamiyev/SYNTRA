const ACTION = /^(?:hey[, ]*)?(?:please\s+)?(?:(?:can|could|would|will) you(?:\s+please)?\s+)?(?:please\s+)?(?:implement|build|fix|debug|refactor|add|create|update|change|repair|write|remove|optimize|migrate|test|extend|integrate|wire|replace|make|improve|enhance|upgrade|harden|simplify|modernize|redesign|modify|edit)\b/i;
const ACTION_ANYWHERE = /\b(?:implement|build|fix|debug|refactor|add|create|update|change|repair|write|remove|optimize|migrate|test|extend|integrate|wire|replace|make|improve|enhance|upgrade|harden|simplify|modernize|redesign|modify|edit)\b/i;

const SOFTWARE_TARGET = /\b(?:app|website|web\s+app(?:lication)?|web\s+application|desktop\s+application|mobile\s+application|webpage|web\s+page|frontend|front[- ]end|backend|back[- ]end|ui|ux|interface|component|module|function|class|feature|bug|error|test(?:s| suite)?|build|compile|lint|typecheck|api|endpoint|route|server|client|script|package|dependency|dependencies|npm|vite|react|typescript|javascript|python|css|html|sql|database|query|schema|authentication|login|button|form|responsive|accessibility|performance|chat\s+(?:ui|interface|frontend|front[- ]end|app)|code|source|repository|repo|codebase|project files|project code|unit test|integration test|stack trace|exception|design system|dark mode|theme|navigation|routing)\b/i;
const EXPLICIT_REPO_CONTEXT = /\b(?:this|the|current|my)\s+(?:codebase|repository|repo|project files|source code)\b|\b(?:in|inside|within)\s+(?:this|the|current|my)\s+(?:project files|repository|repo|codebase|source code)\b/i;
const CONVERSATIONAL_OUTPUT = /^(?:hey[, ]*)?(?:please\s+)?(?:(?:can|could|would) you(?:\s+please)?\s+)?(?:please\s+)?(?:write|create|draft|compose|prepare|generate)\s+(?:(?:me|us)\s+)?(?:an?\s+)?(?:email|letter|poem|story|song|essay|article|blog post|social post|text message|message|proposal|job application|cover letter|resume|résumé|cv|joke|speech|recipe)\b/i;

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

  const command = text.match(ACTION);
  if (!command) {
    return /^(?:can|could|would) you\b/i.test(text)
      && ACTION_ANYWHERE.test(text)
      && SOFTWARE_TARGET.test(text.slice(0, 240));
  }

  const targetWindow = text.slice(command[0].length, command[0].length + 240);
  const taskObject = targetWindow.split(/\b(?:about|regarding|concerning|describing|explaining|discussing)\b/i)[0];
  return SOFTWARE_TARGET.test(taskObject);
}

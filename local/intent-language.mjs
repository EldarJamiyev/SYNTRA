/**
 * Bounded language normalization for intent routing only.
 *
 * This deterministic layer handles known shorthand, common typos, and a
 * constrained command vocabulary in English, Azerbaijani, Turkish, and Russian.
 * It never rewrites the prompt sent to inference or alters extracted source
 * paths and code identifiers. It is not a generative language model.
 */
const ALIASES = new Map(Object.entries({
  pls: "please", plz: "please", plese: "please", emial: "email", featre: "feature",
  upgarde: "upgrade", u: "you", ur: "your", r: "are", btw: "by the way",
  dont: "do not", doesnt: "does not", didnt: "did not", cant: "cannot",
  couldnt: "could not", wont: "will not", wouldnt: "would not", isnt: "is not",
  arent: "are not", im: "i am", ive: "i have", ill: "i will", youre: "you are",
  youve: "you have", theyre: "they are", whats: "what is", wheres: "where is",
  hows: "how is", lets: "let us",
  // Azerbaijani and common unaccented keyboard variants.
  elave: "add", elaveet: "add", əlavə: "add", əlavəet: "add", duzelt: "fix",
  düzəlt: "fix", temire: "repair", yenile: "update", yenilə: "update",
  yarat: "create", qur: "build", deyis: "change", dəyiş: "change",
  inkisaf: "improve", inkişaf: "improve", yoxla: "test", izah: "explain",
  goster: "show", göstər: "show", davam: "continue", layihe: "project",
  layihə: "project", sayt: "website", sayti: "website", veb: "web",
  tetbiq: "application", tətbiq: "application", kod: "code", kodu: "code",
  lütfən: "please", zehmet: "please", zəhmət: "please",
  // Turkish.
  lutfen: "please", lütfen: "please", duzelt: "fix", düzelt: "fix",
  ekle: "add", olustur: "create", oluştur: "create", guncelle: "update",
  güncelle: "update", kaldir: "remove", kaldır: "remove", degistir: "change",
  değiştir: "change", iyilestir: "improve", iyileştir: "improve",
  acikla: "explain", açıkla: "explain", kontrol: "test", uygulama: "application",
  uygulamayi: "application", siteyi: "website", proje: "project",
  // Russian.
  пожалуйста: "please", исправь: "fix", исправить: "fix", почини: "fix",
  починить: "fix", добавь: "add", добавить: "add", создай: "create",
  создать: "create", обнови: "update", обновить: "update", удали: "remove",
  удалить: "remove", измени: "change", изменить: "change", улучши: "improve",
  улучшить: "improve", объясни: "explain", объяснить: "explain",
  продолжай: "continue", проверь: "test", приложение: "application",
  сайт: "website", код: "code", проект: "project"
}));

const INTENT_TERMS = [
  "implement", "build", "fix", "debug", "refactor", "add", "create", "update",
  "change", "repair", "write", "remove", "optimize", "migrate", "test", "extend",
  "integrate", "wire", "replace", "make", "improve", "enhance", "upgrade",
  "harden", "simplify", "modernize", "redesign", "modify", "edit", "patch",
  "application", "website", "webpage", "frontend", "backend", "interface",
  "component", "module", "function", "feature", "error", "tests", "compile",
  "endpoint", "route", "server", "client", "script", "package", "dependency",
  "dependencies", "react", "typescript", "javascript", "python", "database",
  "query", "schema", "authentication", "login", "button", "responsive",
  "accessibility", "performance", "repository", "repo", "codebase", "source",
  "workspace", "project", "coding", "chat", "email", "letter", "poem", "story",
  "article", "proposal", "resume", "explain", "describe", "summarize", "inspect",
  "review", "analyze", "trace", "where", "does", "files", "modules", "classes",
  "what", "theme", "dark", "dns", "code", "site", "web", "app", "ui", "ux"
];
const TERMS = new Set(INTENT_TERMS);
const STOPWORDS = new Set([
  "a", "an", "the", "this", "that", "these", "those", "what", "where", "when",
  "who", "which", "why", "how", "is", "are", "was", "were", "do", "does", "did",
  "i", "you", "we", "they", "it", "me", "my", "your", "our", "their", "about",
  "from", "with", "and", "or", "not", "for", "of", "to", "in", "on", "at", "as",
  "by", "be", "can", "could", "would", "will", "should", "please", "want", "need",
  "more", "again", "further", "there", "here", "now", "then"
]);
const TERMS_BY_LENGTH = new Map();
for (const term of TERMS) {
  const bucket = TERMS_BY_LENGTH.get(term.length) || [];
  bucket.push(term);
  TERMS_BY_LENGTH.set(term.length, bucket);
}

function editDistanceWithin(left, right, limit) {
  if (Math.abs(left.length - right.length) > limit) return limit + 1;
  let previous = Array.from({ length: right.length + 1 }, (_, index) => index);
  for (let row = 1; row <= left.length; row += 1) {
    const current = [row];
    let minimum = row;
    for (let column = 1; column <= right.length; column += 1) {
      const cost = left[row - 1] === right[column - 1] ? 0 : 1;
      const value = Math.min(
        previous[column] + 1,
        current[column - 1] + 1,
        previous[column - 1] + cost
      );
      current.push(value);
      minimum = Math.min(minimum, value);
    }
    if (minimum > limit) return limit + 1;
    previous = current;
  }
  return previous[right.length];
}

function correctKnownTerm(token) {
  if (TERMS.has(token) || STOPWORDS.has(token) || token.length < 4 || token.length > 18) return token;
  const limit = token.length >= 7 ? 2 : 1;
  let bestDistance = limit + 1;
  let best = null;
  let tied = false;
  const candidates = [];
  for (let length = Math.max(4, token.length - limit); length <= Math.min(18, token.length + limit); length += 1) {
    candidates.push(...(TERMS_BY_LENGTH.get(length) || []));
  }
  for (const candidate of candidates) {
    const distance = editDistanceWithin(token, candidate, limit);
    if (distance < bestDistance) {
      bestDistance = distance;
      best = candidate;
      tied = false;
    } else if (distance === bestDistance) {
      tied = true;
    }
  }
  return best && !tied && bestDistance <= limit ? best : token;
}

/**
 * Return routing-only normalized text. The original input remains authoritative.
 */
export function normalizeIntentText(value) {
  if (typeof value !== "string" || !value) return "";
  const bounded = value.normalize("NFKC").replace(/[’‘]/g, "'").slice(0, 12000);
  let inspectedTokens = 0;
  return bounded
    .replace(/[\p{L}\p{N}]+(?:'[\p{L}]+)?/gu, raw => {
      const token = raw.toLowerCase();
      const alias = ALIASES.get(token);
      if (alias) return alias;
      inspectedTokens += 1;
      if (inspectedTokens > 2048) return token;
      return correctKnownTerm(token);
    })
    .replace(/\s+/g, " ")
    .trim();
}

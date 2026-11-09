export function isRecord(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

export function normalizeEventType(value) {
  return value;
}

export function normalizeTimestamp(value) {
  return value;
}

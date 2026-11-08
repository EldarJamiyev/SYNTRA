import { isRecord, normalizeEventType, normalizeTimestamp } from "./schema.mjs";

export function normalizeEvent(input) {
  let record;
  try {
    record = typeof input === "string" ? JSON.parse(input) : input;
  } catch {
    return null;
  }
  if (!isRecord(record)) return null;
  return {
    type: normalizeEventType(record.type),
    timestamp: normalizeTimestamp(record.timestamp),
    payload: record.payload,
  };
}

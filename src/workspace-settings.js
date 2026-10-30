export const WORKSPACE_PATH_KEY = "syntra-active-workspace-v1";
const MAX_PATH_LENGTH = 4096;

function resolveStorage(storage) {
  if (storage) return storage;
  try {
    return globalThis.localStorage || null;
  } catch {
    return null;
  }
}

export function normalizeWorkspacePath(value) {
  if (typeof value !== "string") return null;
  const normalized = value.trim();
  if (!normalized || normalized.length > MAX_PATH_LENGTH || /[\0\r\n]/.test(normalized)) return null;
  return normalized;
}

export function loadWorkspacePath(storage) {
  const target = resolveStorage(storage);
  if (!target) return null;
  try {
    return normalizeWorkspacePath(target.getItem(WORKSPACE_PATH_KEY));
  } catch {
    return null;
  }
}

export function saveWorkspacePath(value, storage) {
  const target = resolveStorage(storage);
  const normalized = normalizeWorkspacePath(value);
  if (!target || !normalized) return false;
  try {
    target.setItem(WORKSPACE_PATH_KEY, normalized);
    return true;
  } catch {
    return false;
  }
}

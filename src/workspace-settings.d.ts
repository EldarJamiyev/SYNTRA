export const WORKSPACE_PATH_KEY: string;
export function normalizeWorkspacePath(value: unknown): string | null;
export function loadWorkspacePath(storage?: { getItem(key: string): string | null }): string | null;
export function saveWorkspacePath(value: string, storage?: { setItem(key: string, value: string): void }): boolean;

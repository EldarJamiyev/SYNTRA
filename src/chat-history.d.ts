export type ChatMessage = {
  role: "user" | "assistant";
  content: string;
};

export interface ChatStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

export const CHAT_HISTORY_KEY: string;
export function normalizeMessages(value: unknown): ChatMessage[];
export function parseStoredMessages(serialized: string | null): ChatMessage[];
export function serializeMessages(messages: readonly ChatMessage[]): string;
export function loadMessages(storage?: ChatStorage): ChatMessage[];
export function saveMessages(messages: readonly ChatMessage[], storage?: ChatStorage): boolean;

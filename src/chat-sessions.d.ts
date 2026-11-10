import type { ChatMessage, ChatStorage } from "./chat-history.js";

export type ChatSession = {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  messages: ChatMessage[];
};

export type ChatSessionState = {
  version: 1;
  activeSessionId: string;
  sessions: ChatSession[];
};

export const CHAT_SESSIONS_KEY: string;
export const MAX_CHAT_SESSIONS: number;
export const MAX_SESSION_MESSAGES: number;
export const MAX_SESSION_TEXT_CHARS: number;
export function titleFromMessages(messages: readonly ChatMessage[]): string;
export function createChatSession(messages?: readonly ChatMessage[], options?: { id?: string; title?: string; now?: string }): ChatSession;
export function normalizeSessionState(value: unknown): ChatSessionState;
export function parseStoredSessionState(serialized: string | null): ChatSessionState | null;
export function serializeSessionState(state: ChatSessionState): string;
export function loadSessionState(storage?: ChatStorage): ChatSessionState;
export function saveSessionState(state: ChatSessionState, storage?: ChatStorage): boolean;

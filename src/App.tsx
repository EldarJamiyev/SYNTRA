import { FormEvent, KeyboardEvent, useEffect, useRef, useState } from 'react';
import { ArrowUp, Check, FolderOpen, LoaderCircle, Menu, MessageSquare, Pencil, Plus, Search, Trash2, X } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { createChatSession, loadSessionState, MAX_CHAT_SESSIONS, saveSessionState, titleFromMessages } from './chat-sessions.js';
import type { ChatMessage } from './chat-history.js';
import type { ChatSession, ChatSessionState } from './chat-sessions.js';
import { loadWorkspacePath, saveWorkspacePath } from './workspace-settings.js';

type WorkspaceProposal = {
  id: string;
  path: string;
  createdAt?: string;
  reason?: string;
  summary?: { addedLines?: number; removedLines?: number; preview?: string };
};

type WorkspaceStatus = {
  root: string;
  pendingProposals: WorkspaceProposal[];
  staleProposals: WorkspaceProposal[];
};

type Message = ChatMessage;
const MAX_VISIBLE_MESSAGES = 60;

type ChatJob = {
  jobId: string;
  status: 'running' | 'completed' | 'failed';
  phase?: string;
  message?: string;
  reply?: string;
  error?: string | null;
  partialReply?: string;
};

async function postJson<T>(url: string, body: unknown): Promise<T> {
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(15_000),
  });

  const raw = await response.text();
  let result: (T & { error?: string }) | undefined;
  try {
    result = JSON.parse(raw) as T & { error?: string };
  } catch {
    const status = response.ok ? 'returned an invalid response' : 'failed with HTTP ' + response.status;
    throw new Error("SYNTRA's local service " + status + '. Check that Start-SYNTRA.bat has finished launching, then refresh and try again.');
  }

  if (!response.ok) throw new Error(result?.error || 'The local chat service returned an error.');
  if (!result || typeof result !== 'object') {
    throw new Error('SYNTRA returned an unexpected response. Restart the local service and try again.');
  }
  return result;
}

async function getJson<T>(url: string): Promise<T> {
  const response = await fetch(url, {
    method: 'GET',
    headers: { Accept: 'application/json' },
    signal: AbortSignal.timeout(8_000),
  });
  const raw = await response.text();
  let result: (T & { error?: string }) | undefined;
  try {
    result = JSON.parse(raw) as T & { error?: string };
  } catch {
    throw new Error(response.ok
      ? "SYNTRA's local workspace service returned an invalid response."
      : "SYNTRA's local workspace service failed with HTTP " + response.status + ".");
  }
  if (!response.ok) throw new Error(result?.error || "SYNTRA's local workspace service is unavailable.");
  if (!result || typeof result !== 'object') throw new Error('SYNTRA returned an unexpected workspace response.');
  return result;
}

function safeMarkdownHref(href?: string) {
  if (!href) return null;
  if (/^(?:https?:\/\/|mailto:|#|\/(?!\/)|\.{1,2}\/)/i.test(href)) return href;
  return null;
}

function workspaceLabel(root: string) {
  const trimmed = root.replace(/[\\/]+$/, '');
  const parts = trimmed.split(/[\\/]/);
  return parts[parts.length - 1] || root;
}

function sessionDateLabel(value: string) {
  const date = new Date(value);
  if (!Number.isFinite(date.getTime())) return '';
  const today = new Date();
  const yesterday = new Date(today);
  yesterday.setDate(today.getDate() - 1);
  if (date.toDateString() === today.toDateString()) return 'Today';
  if (date.toDateString() === yesterday.toDateString()) return 'Yesterday';
  return new Intl.DateTimeFormat(undefined, { month: 'short', day: 'numeric' }).format(date);
}

function MessageContent({ content }: { content: string }) {
  return (
    <div className="message-content">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        skipHtml
        components={{
          a({ href, children }) {
            const safeHref = safeMarkdownHref(href);
            if (!safeHref) return <span>{children}</span>;
            const external = /^https?:\/\//i.test(safeHref);
            return (
              <a
                href={safeHref}
                {...(external ? { target: '_blank', rel: 'noopener noreferrer' } : {})}
              >
                {children}
              </a>
            );
          },
          code({ className, children }) {
            const isBlock = Boolean(className) || String(children).includes('\n');
            return isBlock
              ? <code className={className}>{children}</code>
              : <code className="message-inline-code">{children}</code>;
          },
          pre({ children }) {
            return <pre className="message-code">{children}</pre>;
          },
          table({ children }) {
            return <div className="message-table-wrap"><table>{children}</table></div>;
          },
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}

export default function App() {
  const [sessionState, setSessionState] = useState<ChatSessionState>(() => loadSessionState());
  const [historyOpen, setHistoryOpen] = useState(false);
  const [sessionSearch, setSessionSearch] = useState('');
  const [workspaceStatus, setWorkspaceStatus] = useState<WorkspaceStatus | null>(null);
  const [workspaceLoading, setWorkspaceLoading] = useState(true);
  const [workspaceOpen, setWorkspaceOpen] = useState(false);
  const [workspacePath, setWorkspacePath] = useState(() => loadWorkspacePath() || '');
  const [workspaceSwitching, setWorkspaceSwitching] = useState(false);
  const [workspaceError, setWorkspaceError] = useState('');
  const [renamingSessionId, setRenamingSessionId] = useState<string | null>(null);
  const [renameDraft, setRenameDraft] = useState('');
  const [input, setInput] = useState('');
  const [pending, setPending] = useState(false);
  const [phase, setPhase] = useState('');
  const [streamingReply, setStreamingReply] = useState('');
  const [error, setError] = useState('');
  const sessions = sessionState.sessions;
  const activeSession = sessions.find(session => session.id === sessionState.activeSessionId) || sessions[0];
  const messages = activeSession?.messages || [];

  const setMessages = (update: Message[] | ((current: Message[]) => Message[])) => {
    setSessionState(current => {
      const target = current.sessions.find(session => session.id === current.activeSessionId);
      if (!target) return current;
      const nextMessages = typeof update === 'function' ? update(target.messages) : update;
      const safeMessages = nextMessages.slice(-MAX_VISIBLE_MESSAGES);
      const title = safeMessages.length === 0
        ? 'New chat'
        : target.title === 'New chat' ? titleFromMessages(safeMessages) : target.title;
      const updatedAt = new Date().toISOString();
      const nextSessions = current.sessions.map(session => session.id === target.id
        ? { ...session, title, messages: safeMessages, updatedAt }
        : session).sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));
      return { ...current, sessions: nextSessions };
    });
  };
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const threadRef = useRef<HTMLElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const shouldFollowRef = useRef(true);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    textareaRef.current?.focus();
    return () => { mountedRef.current = false; };
  }, []);

  useEffect(() => {
    let cancelled = false;
    const initializeWorkspace = async () => {
      try {
        let status = await getJson<WorkspaceStatus>('/local-api/workspace/status');
        const savedPath = loadWorkspacePath();
        if (savedPath && savedPath !== status.root) {
          try {
            status = await postJson<WorkspaceStatus>('/local-api/workspace/switch', { root: savedPath });
          } catch (cause) {
            if (!cancelled) {
              setWorkspaceError(cause instanceof Error ? cause.message : String(cause));
              setWorkspacePath(savedPath);
            }
          }
        } else if (!savedPath) {
          saveWorkspacePath(status.root);
        }
        if (!cancelled) {
          setWorkspaceStatus(status);
          if (!(savedPath && status.root !== savedPath)) setWorkspacePath(status.root);
          setWorkspaceLoading(false);
        }
      } catch (cause) {
        if (!cancelled) {
          setWorkspaceError(cause instanceof Error ? cause.message : String(cause));
          setWorkspaceLoading(false);
        }
      }
    };
    void initializeWorkspace();
    return () => { cancelled = true; };
  }, []);

  const startNewChat = () => {
    if (pending) return;
    shouldFollowRef.current = true;
    const session = createChatSession();
    setSessionState(current => ({
      version: 1,
      activeSessionId: session.id,
      sessions: [session, ...current.sessions].slice(0, MAX_CHAT_SESSIONS)
    }));
    setHistoryOpen(false);
    setSessionSearch('');
    setError('');
    setStreamingReply('');
    setInput('');
    textareaRef.current?.focus();
  };

  const selectSession = (sessionId: string) => {
    if (pending) return;
    shouldFollowRef.current = true;
    setSessionState(current => current.sessions.some(session => session.id === sessionId)
      ? { ...current, activeSessionId: sessionId }
      : current);
    setHistoryOpen(false);
    setError('');
    setStreamingReply('');
    setInput('');
    textareaRef.current?.focus();
  };

  const deleteSession = (sessionId: string) => {
    if (pending) return;
    setSessionState(current => {
      const remaining = current.sessions.filter(session => session.id !== sessionId);
      if (!remaining.length) {
        const fresh = createChatSession();
        return { version: 1, activeSessionId: fresh.id, sessions: [fresh] };
      }
      const activeSessionId = current.activeSessionId === sessionId
        ? remaining[0].id
        : current.activeSessionId;
      return { ...current, activeSessionId, sessions: remaining };
    });
    if (renamingSessionId === sessionId) setRenamingSessionId(null);
  };

  const beginRename = (session: ChatSession) => {
    setRenamingSessionId(session.id);
    setRenameDraft(session.title);
  };

  const saveRename = (event: FormEvent<HTMLFormElement>, sessionId: string) => {
    event.preventDefault();
    const title = renameDraft.trim().slice(0, 80);
    if (title) {
      setSessionState(current => ({
        ...current,
        sessions: current.sessions.map(session => session.id === sessionId ? { ...session, title } : session)
      }));
    }
    setRenamingSessionId(null);
    setRenameDraft('');
  };

  const filteredSessions = sessionSearch.trim()
    ? sessions.filter(session => {
        const query = sessionSearch.trim().toLowerCase();
        return session.title.toLowerCase().includes(query)
          || session.messages.some(message => message.content.toLowerCase().includes(query));
      })
    : sessions;

  const refreshWorkspaceStatus = async () => {
    const status = await getJson<WorkspaceStatus>('/local-api/workspace/status');
    if (mountedRef.current) {
      setWorkspaceStatus(status);
      setWorkspacePath(status.root);
      setWorkspaceError('');
      saveWorkspacePath(status.root);
    }
  };

  const submitWorkspaceSwitch = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (pending || workspaceSwitching) return;
    const requestedPath = workspacePath.trim();
    if (!requestedPath) {
      setWorkspaceError('Enter an absolute project folder path.');
      return;
    }
    setWorkspaceSwitching(true);
    setWorkspaceError('');
    try {
      const status = await postJson<WorkspaceStatus>('/local-api/workspace/switch', { root: requestedPath });
      setWorkspaceStatus(status);
      setWorkspacePath(status.root);
      saveWorkspacePath(status.root);
      setWorkspaceOpen(false);
    } catch (cause) {
      setWorkspaceError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setWorkspaceSwitching(false);
    }
  };

  const reviewSavedProposals = () => {
    setInput('/proposals');
    setWorkspaceOpen(false);
    textareaRef.current?.focus();
  };

  const handleThreadScroll = () => {
    const thread = threadRef.current;
    if (!thread) return;
    const distanceFromBottom = thread.scrollHeight - thread.scrollTop - thread.clientHeight;
    shouldFollowRef.current = distanceFromBottom < 120;
  };

  useEffect(() => {
    if (!shouldFollowRef.current) return;
    const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    bottomRef.current?.scrollIntoView({
      behavior: pending || reduceMotion ? 'auto' : 'smooth',
      block: 'end',
    });
  }, [messages, pending, phase, streamingReply]);

  useEffect(() => {
    // Save the user's prompt immediately too. Persist each conversation independently;
    // bounded local storage and non-fatal quota/privacy failures keep chat available.
    saveSessionState(sessionState);
  }, [sessionState]);

  const send = async (event?: FormEvent) => {
    event?.preventDefault();
    const content = input.trim();
    if (!content || pending) return;

    if (content.toLowerCase() === '/clear') {
      setMessages([]);
      setInput('');
      setError('');
      setStreamingReply('');
      if (textareaRef.current) textareaRef.current.style.height = 'auto';
      return;
    }

    shouldFollowRef.current = true;
    const history = messages.slice(-12);
    setMessages(current => [...current, { role: 'user' as const, content }].slice(-MAX_VISIBLE_MESSAGES));
    setInput('');
    setError('');
    setPending(true);
    setPhase('Thinking');
    setStreamingReply('');
    if (textareaRef.current) textareaRef.current.style.height = 'auto';

    let requestStarted = false;
    let partialReply = '';
    try {
      const started = await postJson<ChatJob>('/local-api/chat', { message: content, history });
      requestStarted = true;
      let job = started;
      while (job.status === 'running') {
        if (!mountedRef.current) return;
        setPhase(job.message || 'Thinking');
        if (job.reply && job.status === 'running') {
          partialReply = job.reply;
          setStreamingReply(partialReply);
        }
        await new Promise(resolve => window.setTimeout(resolve, 800));
        job = await postJson<ChatJob>('/local-api/chat/status', { jobId: started.jobId });
      }
      if (!mountedRef.current) return;
      if (job.status === 'failed') {
        setStreamingReply('');
        const interrupted = partialReply || job.partialReply || '';
        setMessages(current => {
          const last = current[current.length - 1];
          const restored = last?.role === 'user' && last.content === content ? current.slice(0, -1) : current;
          if (!interrupted) return restored;
          return [...restored, {
            role: 'assistant' as const,
            content: interrupted + '\n\n---\n\n*Response interrupted before completion; this may be incomplete.*',
          }].slice(-MAX_VISIBLE_MESSAGES);
        });
        setInput(content);
        setError(job.error || job.reply || 'The local request failed.');
        return;
      }
      const answer = job.reply || 'No response was returned.';
      setStreamingReply('');
      setMessages(current => [...current, { role: 'assistant' as const, content: answer }].slice(-MAX_VISIBLE_MESSAGES));
      try {
        await refreshWorkspaceStatus();
      } catch {
        // A transient status refresh must not turn a completed chat answer into a failure.
      }
    } catch (cause) {
      if (mountedRef.current) {
        setStreamingReply('');
        setMessages(current => {
          const last = current[current.length - 1];
          const restored = last?.role === 'user' && last.content === content ? current.slice(0, -1) : current;
          if (!partialReply) return restored;
          return [...restored, {
            role: 'assistant',
            content: partialReply + '\n\n---\n\n*Response interrupted before completion; this may be incomplete.*',
          }];
        });
        setInput(content);
        const detail = cause instanceof Error && cause.name === 'TimeoutError'
          ? "SYNTRA's local service did not respond within 15 seconds. Check the service windows and try again."
          : cause instanceof Error ? cause.message : String(cause);
        setError(detail);
      }
    } finally {
      if (mountedRef.current) {
        setPending(false);
        setPhase('');
        textareaRef.current?.focus();
      }
    }
  };

  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      void send();
    }
  };

  return (
    <main className="chat-app">
      <header className="chat-header">
        <button
          className="header-icon-button"
          type="button"
          aria-label={historyOpen ? 'Close conversation history' : 'Open conversation history'}
          aria-expanded={historyOpen}
          onClick={() => {
            setWorkspaceOpen(false);
            setHistoryOpen(open => !open);
          }}
        >
          <Menu size={19} strokeWidth={1.8} />
        </button>
        <span className="chat-brand">SYNTRA</span>
        <span className="header-session-title">{activeSession?.title || 'New chat'}</span>
        <button
          className="workspace-trigger"
          type="button"
          aria-label="Active workspace"
          aria-expanded={workspaceOpen}
          title={workspaceStatus?.root || 'Local project workspace'}
          onClick={() => {
            setHistoryOpen(false);
            setWorkspacePath(workspaceStatus?.root || loadWorkspacePath() || workspacePath);
            setWorkspaceOpen(open => !open);
          }}
        >
          <FolderOpen size={15} strokeWidth={1.8} />
          <span>{workspaceLoading ? 'Connecting…' : workspaceStatus ? workspaceLabel(workspaceStatus.root) : 'Workspace offline'}</span>
        </button>
        <div className="chat-header-spacer" />
        <button
          className="header-icon-button new-chat-button"
          type="button"
          aria-label="New chat"
          title="New chat"
          onClick={startNewChat}
          disabled={pending}
        >
          <Plus size={19} strokeWidth={1.9} />
        </button>
      </header>

      {workspaceOpen && (
        <section className="workspace-popover" role="dialog" aria-label="Active workspace">
          <div className="workspace-popover-header">
            <div className="workspace-popover-title"><FolderOpen size={16} /> <span>Active workspace</span></div>
            <button className="history-icon-button" type="button" aria-label="Close workspace panel" onClick={() => setWorkspaceOpen(false)}>
              <X size={17} />
            </button>
          </div>
          <div className="workspace-current-path" title={workspaceStatus?.root || ''}>
            {workspaceStatus?.root || (workspaceLoading ? 'Connecting to the local service…' : 'Workspace unavailable')}
          </div>
          <form className="workspace-switch-form" onSubmit={submitWorkspaceSwitch}>
            <label htmlFor="workspace-path">Project folder path</label>
            <input
              id="workspace-path"
              value={workspacePath}
              onChange={event => setWorkspacePath(event.target.value)}
              placeholder={'C:\\Projects\\MyProject'}
              maxLength={4096}
              disabled={pending || workspaceSwitching}
              autoComplete="off"
              spellCheck={false}
            />
            <button type="submit" disabled={pending || workspaceLoading || workspaceSwitching || !workspacePath.trim()}>
              {workspaceSwitching ? 'Switching…' : 'Switch folder'}
            </button>
          </form>
          {workspaceError && <div className="workspace-error" role="alert">{workspaceError}</div>}
          {workspaceStatus && (workspaceStatus.pendingProposals.length > 0 || workspaceStatus.staleProposals.length > 0) && (
            <div className="workspace-proposal-summary">
              <div className="workspace-proposal-count">
                {workspaceStatus.pendingProposals.length} pending
                {workspaceStatus.staleProposals.length > 0 ? ' · ' + workspaceStatus.staleProposals.length + ' stale' : ''}
              </div>
              <div className="workspace-proposal-files">
                {[...workspaceStatus.pendingProposals, ...workspaceStatus.staleProposals].slice(0, 5).map(proposal => (
                  <div className="workspace-proposal-file" key={proposal.id}>
                    <span>{proposal.path}</span>
                    <span>{proposal.summary ? '+' + Number(proposal.summary.addedLines || 0) + ' / −' + Number(proposal.summary.removedLines || 0) : 'Stale'}</span>
                  </div>
                ))}
                {workspaceStatus.pendingProposals.length + workspaceStatus.staleProposals.length > 5 && (
                  <div className="workspace-proposal-more">And {workspaceStatus.pendingProposals.length + workspaceStatus.staleProposals.length - 5} more…</div>
                )}
              </div>
              <button className="workspace-review-button" type="button" onClick={reviewSavedProposals}>
                Review saved proposals in chat
              </button>
            </div>
          )}
          <div className="workspace-popover-footer">Project files and proposal records remain on this device.</div>
        </section>
      )}

      {historyOpen && (
        <>
          <button
            className="history-backdrop"
            type="button"
            aria-label="Close conversation history"
            onClick={() => setHistoryOpen(false)}
          />
          <aside className="history-drawer" aria-label="Conversation history">
            <div className="history-drawer-header">
              <div className="history-drawer-title"><MessageSquare size={17} /> <span>Conversations</span></div>
              <button className="history-icon-button" type="button" aria-label="Close history" onClick={() => setHistoryOpen(false)}>
                <X size={18} />
              </button>
            </div>
            <label className="history-search">
              <Search size={16} />
              <input
                value={sessionSearch}
                onChange={event => setSessionSearch(event.target.value)}
                placeholder="Search conversations"
                aria-label="Search conversations"
              />
            </label>
            <button className="history-new-chat" type="button" onClick={startNewChat} disabled={pending}>
              <Plus size={16} /> <span>New chat</span>
            </button>
            <div className="history-list">
              {filteredSessions.length === 0 ? (
                <div className="history-empty">No matching conversations.</div>
              ) : filteredSessions.map(session => (
                <div className={'history-item ' + (session.id === sessionState.activeSessionId ? 'active' : '')} key={session.id}>
                  {renamingSessionId === session.id ? (
                    <form className="history-rename-form" onSubmit={event => saveRename(event, session.id)}>
                      <input
                        autoFocus
                        value={renameDraft}
                        onChange={event => setRenameDraft(event.target.value)}
                        maxLength={80}
                        aria-label="Conversation name"
                      />
                      <button className="history-icon-button" type="submit" aria-label="Save conversation name"><Check size={15} /></button>
                      <button className="history-icon-button" type="button" aria-label="Cancel rename" onClick={() => setRenamingSessionId(null)}><X size={15} /></button>
                    </form>
                  ) : (
                    <>
                      <button
                        className="history-item-open"
                        type="button"
                        onClick={() => selectSession(session.id)}
                        disabled={pending}
                        aria-current={session.id === sessionState.activeSessionId ? 'page' : undefined}
                      >
                        <span className="history-item-title">{session.title}</span>
                        <span className="history-item-date">{sessionDateLabel(session.updatedAt)} · {session.messages.length} messages</span>
                      </button>
                      <div className="history-item-actions">
                        <button className="history-icon-button" type="button" aria-label={'Rename ' + session.title} title="Rename" onClick={() => beginRename(session)} disabled={pending}>
                          <Pencil size={14} />
                        </button>
                        <button className="history-icon-button danger" type="button" aria-label={'Delete ' + session.title} title="Delete" onClick={() => deleteSession(session.id)} disabled={pending}>
                          <Trash2 size={14} />
                        </button>
                      </div>
                    </>
                  )}
                </div>
              ))}
            </div>
            <div className="history-drawer-footer">Saved locally on this device</div>
          </aside>
        </>
      )}

      <section
        ref={threadRef}
        onScroll={handleThreadScroll}
        className={'chat-thread ' + (messages.length === 0 ? 'chat-thread-empty' : '')}
        aria-label="Conversation"
        aria-live="off"
      >
        {messages.length === 0 ? (
          <div className="chat-welcome">
            <h1>How can I help?</h1>
          </div>
        ) : (
          <div className="chat-messages">
            {messages.map((message, index) => (
              <article
                className={'chat-message ' + message.role}
                key={index}
                aria-label={message.role === 'user' ? 'Your message' : 'SYNTRA response'}
                aria-live={message.role === 'assistant' ? 'polite' : 'off'}
              >
                <div className="chat-bubble">
                    {message.role === 'assistant'
                      ? <MessageContent content={message.content} />
                      : <div className="user-message-text">{message.content}</div>}
                  </div>
              </article>
            ))}
            {pending && streamingReply && (
              <article className="chat-message assistant streaming-message" aria-live="off">
                <div className="chat-bubble"><MessageContent content={streamingReply} /></div>
              </article>
            )}
            {pending && (
              <article className="chat-message assistant pending-message" role="status">
                <div className="pending-content">
                  <LoaderCircle size={15} className="thinking-spinner" />
                  <span>{streamingReply ? 'Responding' : phase || 'Thinking'}</span>
                </div>
              </article>
            )}
          </div>
        )}
        <div ref={bottomRef} />
      </section>

      <div className="chat-composer-wrap">
        {error && <div className="chat-error" role="alert">{error}</div>}
        <form className="chat-composer" onSubmit={send}>
          <textarea
            ref={textareaRef}
            value={input}
            onChange={event => {
              setInput(event.target.value);
              if (error) setError('');
              event.currentTarget.style.height = 'auto';
              event.currentTarget.style.height = Math.min(event.currentTarget.scrollHeight, 180) + 'px';
            }}
            onKeyDown={onKeyDown}
            placeholder="Message SYNTRA"
            aria-label="Message SYNTRA"
            maxLength={12000}
            rows={1}
            disabled={pending || workspaceLoading}
          />
          <button className="send-button" type="submit" aria-label="Send message" disabled={pending || workspaceLoading || !input.trim()}>
            {pending ? <LoaderCircle size={18} className="thinking-spinner" /> : <ArrowUp size={18} strokeWidth={2.4} />}
          </button>
        </form>
      </div>
    </main>
  );
}

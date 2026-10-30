import { FormEvent, KeyboardEvent, useEffect, useRef, useState } from 'react';
import { ArrowUp, Check, FolderOpen, LoaderCircle, Menu, MessageSquare, Pencil, Plus, Search, Square, Trash2, X } from 'lucide-react';
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

type WorkspaceMemoryNote = {
  id: string;
  title: string;
  content: string;
  category: 'architecture' | 'convention' | 'decision' | 'workflow' | 'testing' | 'other';
  tags: string[];
  createdAt: string;
  updatedAt: string;
  owner: 'user';
};

type WorkspaceToolResponse<T> = {
  ok: boolean;
  output?: T;
  error?: string;
  message?: string;
  status?: string;
};

type Message = ChatMessage;
const MAX_VISIBLE_MESSAGES = 60;

type ChatJob = {
  jobId: string;
  status: 'running' | 'completed' | 'failed' | 'cancelled';
  phase?: string;
  message?: string;
  reply?: string;
  error?: string | null;
  partialReply?: string;
};

type AgentRunSummary = {
  id: string;
  goal: string;
  status: 'running' | 'completed' | 'failed' | 'cancelled' | 'interrupted' | 'queued';
  phase: string;
  message: string;
  updatedAt: string;
  recoveryAvailable: boolean;
  checkpoint?: { stage?: string; iteration?: number };
  observations?: Array<{ kind: string; status: string; path?: string | null }>;
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
  const [agentRuns, setAgentRuns] = useState<AgentRunSummary[]>([]);
  const [memoryNotes, setMemoryNotes] = useState<WorkspaceMemoryNote[]>([]);
  const [memorySearch, setMemorySearch] = useState('');
  const [memoryTitle, setMemoryTitle] = useState('');
  const [memoryContent, setMemoryContent] = useState('');
  const [memoryCategory, setMemoryCategory] = useState<WorkspaceMemoryNote['category']>('architecture');
  const [memoryTags, setMemoryTags] = useState('');
  const [editingMemoryId, setEditingMemoryId] = useState<string | null>(null);
  const [deletingMemoryId, setDeletingMemoryId] = useState<string | null>(null);
  const [memoryLoading, setMemoryLoading] = useState(false);
  const [memorySaving, setMemorySaving] = useState(false);
  const [memoryError, setMemoryError] = useState('');
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
  const activeJobIdRef = useRef<string | null>(null);

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
          try {
            const runs = await getJson<{ runs: AgentRunSummary[] }>('/local-api/agent/runs');
            if (!cancelled) setAgentRuns(runs.runs || []);
          } catch {
            // Chat remains available when the optional run-history view is unavailable.
          }
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

  const filteredMemoryNotes = memorySearch.trim()
    ? memoryNotes.filter(note => {
        const query = memorySearch.trim().toLowerCase();
        return note.title.toLowerCase().includes(query)
          || note.content.toLowerCase().includes(query)
          || note.tags.some(tag => tag.toLowerCase().includes(query))
          || note.category.includes(query);
      })
    : memoryNotes;

  const refreshWorkspaceStatus = async () => {
    const status = await getJson<WorkspaceStatus>('/local-api/workspace/status');
    if (mountedRef.current) {
      setWorkspaceStatus(status);
      setWorkspacePath(status.root);
      setWorkspaceError('');
      saveWorkspacePath(status.root);
    }
  };

  const executeWorkspaceTool = async <T,>(
    name: string,
    toolInput: Record<string, unknown> = {},
    approval = false,
    rootOverride?: string,
  ): Promise<T> => {
    const root = rootOverride || workspaceStatus?.root || workspacePath || undefined;
    const result = await postJson<WorkspaceToolResponse<T>>('/local-api/tools/execute', {
      ...(root ? { root } : {}),
      name,
      input: toolInput,
      approval,
    });
    if (!result.ok || result.output === undefined) {
      throw new Error(result.error || result.message || 'The workspace tool did not complete.');
    }
    return result.output;
  };

  const refreshWorkspaceMemory = async (rootOverride?: string) => {
    setMemoryLoading(true);
    try {
      const result = await executeWorkspaceTool<{ notes: WorkspaceMemoryNote[] }>(
        'workspace.memory.list', {}, false, rootOverride,
      );
      if (mountedRef.current) {
        setMemoryNotes(result.notes || []);
        setMemoryError('');
      }
    } finally {
      if (mountedRef.current) setMemoryLoading(false);
    }
  };

  const resetWorkspaceMemoryDraft = () => {
    setEditingMemoryId(null);
    setDeletingMemoryId(null);
    setMemoryTitle('');
    setMemoryContent('');
    setMemoryCategory('architecture');
    setMemoryTags('');
    setMemoryError('');
  };

  const submitWorkspaceMemory = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (memorySaving || pending) return;
    setMemorySaving(true);
    setMemoryError('');
    try {
      const tags = memoryTags.split(',').map(tag => tag.trim()).filter(Boolean);
      const editingNote = memoryNotes.find(note => note.id === editingMemoryId);
      if (editingMemoryId && !editingNote) {
        throw new Error('The note being edited is no longer available. Reload project memory and try again.');
      }
      if (editingNote) {
        await executeWorkspaceTool('workspace.memory.update', {
          id: editingNote.id,
          expectedUpdatedAt: editingNote.updatedAt,
          title: memoryTitle,
          content: memoryContent,
          category: memoryCategory,
          tags,
        }, true);
      } else {
        await executeWorkspaceTool('workspace.memory.add', {
          title: memoryTitle,
          content: memoryContent,
          category: memoryCategory,
          tags,
        }, true);
      }
      resetWorkspaceMemoryDraft();
      await refreshWorkspaceMemory();
    } catch (cause) {
      setMemoryError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setMemorySaving(false);
    }
  };

  const beginEditWorkspaceMemory = (note: WorkspaceMemoryNote) => {
    setEditingMemoryId(note.id);
    setDeletingMemoryId(null);
    setMemoryTitle(note.title);
    setMemoryContent(note.content);
    setMemoryCategory(note.category);
    setMemoryTags(note.tags.join(', '));
    setMemoryError('');
  };

  const confirmDeleteWorkspaceMemory = async (note: WorkspaceMemoryNote) => {
    if (memorySaving || pending) return;
    setMemorySaving(true);
    setMemoryError('');
    try {
      await executeWorkspaceTool('workspace.memory.delete', {
        id: note.id,
        expectedUpdatedAt: note.updatedAt,
      }, true);
      if (editingMemoryId === note.id) resetWorkspaceMemoryDraft();
      setDeletingMemoryId(null);
      await refreshWorkspaceMemory();
    } catch (cause) {
      setMemoryError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setMemorySaving(false);
    }
  };

  const submitWorkspaceSwitch = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (pending || workspaceSwitching || memoryLoading || memorySaving) return;
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
      setMemoryNotes([]);
      resetWorkspaceMemoryDraft();
      try { await refreshAgentRuns(); } catch { setAgentRuns([]); }
      try { await refreshWorkspaceMemory(status.root); } catch (cause) {
        setMemoryError(cause instanceof Error ? cause.message : String(cause));
      }
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

  const refreshAgentRuns = async () => {
    const result = await getJson<{ root: string; runs: AgentRunSummary[] }>('/local-api/agent/runs');
    if (mountedRef.current) setAgentRuns(result.runs || []);
  };

  const cancelActiveChat = async () => {
    const jobId = activeJobIdRef.current;
    if (!jobId) return;
    setPhase('Stopping safely…');
    try {
      const result = await postJson<{ ok: boolean; message?: string }>('/local-api/chat/cancel', { jobId });
      if (!result.ok && result.message) setError(result.message);
    } catch (cause) {
      if (mountedRef.current) setError(cause instanceof Error ? cause.message : String(cause));
    }
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

  const send = async (event?: FormEvent, explicitContent?: string) => {
    event?.preventDefault();
    const content = (explicitContent ?? input).trim();
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
      activeJobIdRef.current = started.jobId;
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
      if (job.status === 'cancelled') {
        setStreamingReply('');
        const partial = partialReply || job.partialReply || '';
        const cancellation = job.reply || 'Task cancelled. Any staged proposals remain available in workspace review.';
        setMessages(current => [...current, {
          role: 'assistant' as const,
          content: partial ? partial + '\n\n---\n\n' + cancellation : cancellation
        }].slice(-MAX_VISIBLE_MESSAGES));
        try { await refreshAgentRuns(); } catch {}
        try { await refreshWorkspaceStatus(); } catch {}
        return;
      }
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
        await refreshAgentRuns();
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
      activeJobIdRef.current = null;
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
            const opening = !workspaceOpen;
            setWorkspaceOpen(opening);
            if (opening) {
              // A local service may have restarted while this view stayed open.
              // Fetch its recovered run state when the workspace panel is reopened.
              void refreshAgentRuns().catch(() => {});
              void refreshWorkspaceStatus().catch(() => {});
              void refreshWorkspaceMemory().catch(cause => {
                if (mountedRef.current) setMemoryError(cause instanceof Error ? cause.message : String(cause));
              });
            }
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
              disabled={pending || workspaceSwitching || memoryLoading || memorySaving}
              autoComplete="off"
              spellCheck={false}
            />
            <button type="submit" disabled={pending || workspaceLoading || workspaceSwitching || memoryLoading || memorySaving || !workspacePath.trim()}>
              {workspaceSwitching ? 'Switching…' : 'Switch folder'}
            </button>
          </form>
          {workspaceError && <div className="workspace-error" role="alert">{workspaceError}</div>}
          <section className="workspace-memory-panel" aria-label="Project memory">
            <div className="workspace-memory-heading">
              <div>
                <div className="workspace-agent-runs-heading">Project memory</div>
                <div className="workspace-memory-subtitle">Only notes you save are available to the agent.</div>
              </div>
              <span className="workspace-memory-count">{memoryNotes.length}/200</span>
            </div>
            <form className="workspace-memory-form" onSubmit={submitWorkspaceMemory}>
              <label htmlFor="memory-title">{editingMemoryId ? 'Edit note' : 'New note'}</label>
              <input
                id="memory-title"
                value={memoryTitle}
                onChange={event => setMemoryTitle(event.target.value)}
                placeholder="Note title"
                maxLength={120}
                required
                disabled={memorySaving || memoryLoading || pending}
              />
              <div className="workspace-memory-form-row">
                <select
                  aria-label="Project memory category"
                  value={memoryCategory}
                  onChange={event => setMemoryCategory(event.target.value as WorkspaceMemoryNote['category'])}
                  disabled={memorySaving || memoryLoading || pending}
                >
                  <option value="architecture">Architecture</option>
                  <option value="convention">Convention</option>
                  <option value="decision">Decision</option>
                  <option value="workflow">Workflow</option>
                  <option value="testing">Testing</option>
                  <option value="other">Other</option>
                </select>
                <input
                  value={memoryTags}
                  onChange={event => setMemoryTags(event.target.value)}
                  placeholder="Tags, comma separated"
                  maxLength={300}
                  aria-label="Project memory tags"
                  disabled={memorySaving || memoryLoading || pending}
                />
              </div>
              <textarea
                aria-label="Project memory note content"
                value={memoryContent}
                onChange={event => setMemoryContent(event.target.value)}
                placeholder="A technical decision, project convention, or detail worth keeping…"
                maxLength={2400}
                rows={3}
                required
                disabled={memorySaving || pending}
              />
              <div className="workspace-memory-form-actions">
                <button type="submit" disabled={memorySaving || memoryLoading || pending || (!editingMemoryId && memoryNotes.length >= 200) || !memoryTitle.trim() || !memoryContent.trim()}>
                  {memorySaving ? 'Saving…' : editingMemoryId ? 'Save changes' : 'Save note'}
                </button>
                {editingMemoryId && (
                  <button type="button" className="workspace-memory-cancel" onClick={resetWorkspaceMemoryDraft} disabled={memorySaving || memoryLoading}>
                    Cancel
                  </button>
                )}
              </div>
            </form>
            <div className="workspace-memory-search">
              <Search size={13} />
              <input
                value={memorySearch}
                onChange={event => setMemorySearch(event.target.value)}
                placeholder="Find a saved note"
                aria-label="Search project memory"
              />
            </div>
            {memoryError && <div className="workspace-error" role="alert">{memoryError}</div>}
            <div className="workspace-memory-list" aria-live="polite">
              {memoryLoading ? (
                <div className="workspace-agent-runs-empty">Loading project memory…</div>
              ) : filteredMemoryNotes.length === 0 ? (
                <div className="workspace-agent-runs-empty">
                  {memoryNotes.length ? 'No notes match this search.' : 'No saved project notes yet.'}
                </div>
              ) : filteredMemoryNotes.slice(0, 20).map(note => (
                <article className="workspace-memory-note" key={note.id}>
                  <div className="workspace-memory-note-top">
                    <span className="workspace-memory-note-title">{note.title}</span>
                    <span className="workspace-memory-note-category">{note.category}</span>
                  </div>
                  <p className="workspace-memory-note-content">{note.content}</p>
                  {note.tags.length > 0 && (
                    <div className="workspace-memory-tags">
                      {note.tags.map(tag => <span key={tag}>{tag}</span>)}
                    </div>
                  )}
                  <div className="workspace-memory-note-footer">
                    <span>{new Date(note.updatedAt).toLocaleDateString()}</span>
                    {deletingMemoryId === note.id ? (
                      <span className="workspace-memory-confirm">
                        Delete?
                        <button type="button" onClick={() => void confirmDeleteWorkspaceMemory(note)} disabled={memorySaving || memoryLoading || pending}>Confirm</button>
                        <button type="button" onClick={() => setDeletingMemoryId(null)} disabled={memorySaving}>Cancel</button>
                      </span>
                    ) : (
                      <span className="workspace-memory-note-actions">
                        <button type="button" aria-label={'Edit ' + note.title} title="Edit note" onClick={() => beginEditWorkspaceMemory(note)} disabled={memorySaving || memoryLoading || pending}>
                          <Pencil size={12} />
                        </button>
                        <button type="button" aria-label={'Delete ' + note.title} title="Delete note" onClick={() => setDeletingMemoryId(note.id)} disabled={memorySaving || memoryLoading || pending}>
                          <Trash2 size={12} />
                        </button>
                      </span>
                    )}
                  </div>
                </article>
              ))}
              {filteredMemoryNotes.length > 20 && (
                <div className="workspace-agent-runs-empty">Showing 20 of {filteredMemoryNotes.length} notes. Search to narrow the list.</div>
              )}
            </div>
            <div className="workspace-memory-footnote">Stored in this workspace’s .syntra folder. The agent can search notes but cannot change them.</div>
          </section>

          <div className="workspace-agent-runs">
            <div className="workspace-agent-runs-heading">Recent agent runs</div>
            {agentRuns.length === 0 ? (
              <div className="workspace-agent-runs-empty">No coding tasks recorded yet.</div>
            ) : agentRuns.slice(0, 5).map(run => (
              <div className="workspace-agent-run" key={run.id}>
                <div className="workspace-agent-run-main">
                  <span className="workspace-agent-run-goal">{run.goal}</span>
                  <span className="workspace-agent-run-meta">
                    <span className={'agent-run-status ' + run.status}>{run.status}</span>
                    <span>{run.checkpoint?.stage || run.phase}</span>
                    {(run.observations?.length || 0) > 0 && (
                      <span title="Persisted repository, verification, and candidate-integrity observations">
                        {run.observations?.length} evidence points
                      </span>
                    )}
                  </span>
                </div>
                {run.status === 'interrupted' && (
                  <button className="workspace-agent-resume" type="button" onClick={() => {
                    setWorkspaceOpen(false);
                    void send(undefined, '/resume ' + run.id);
                  }}>Resume</button>
                )}
              </div>
            ))}
          </div>
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
          <div className="workspace-popover-footer">Project files, project notes, and proposal records remain on this device.</div>
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
          {pending ? (
            <button className="send-button stop-button" type="button" aria-label="Stop task" title="Stop task" onClick={() => void cancelActiveChat()}>
              <Square size={14} fill="currentColor" strokeWidth={1.5} />
            </button>
          ) : (
            <button className="send-button" type="submit" aria-label="Send message" disabled={workspaceLoading || !input.trim()}>
              <ArrowUp size={18} strokeWidth={2.4} />
            </button>
          )}
        </form>
      </div>
    </main>
  );
}

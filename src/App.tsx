import { FormEvent, KeyboardEvent, useEffect, useRef, useState } from 'react';
import { ArrowUp, LoaderCircle } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { loadMessages, saveMessages } from './chat-history.js';

type Message = { role: 'user' | 'assistant'; content: string };
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

function safeMarkdownHref(href?: string) {
  if (!href) return null;
  if (/^(?:https?:\/\/|mailto:|#|\/(?!\/)|\.{1,2}\/)/i.test(href)) return href;
  return null;
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
  const [messages, setMessages] = useState<Message[]>(() => loadMessages());
  const [input, setInput] = useState('');
  const [pending, setPending] = useState(false);
  const [phase, setPhase] = useState('');
  const [streamingReply, setStreamingReply] = useState('');
  const [error, setError] = useState('');
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

  const handleThreadScroll = () => {
    const thread = threadRef.current;
    if (!thread) return;
    const distanceFromBottom = thread.scrollHeight - thread.scrollTop - thread.clientHeight;
    shouldFollowRef.current = distanceFromBottom < 120;
  };

  useEffect(() => {
    if (!shouldFollowRef.current) return;
    bottomRef.current?.scrollIntoView({ behavior: pending ? 'auto' : 'smooth', block: 'end' });
  }, [messages, pending, phase, streamingReply]);

  useEffect(() => {
    // Avoid persisting a turn before it finishes; successful replies and clearly marked
    // interrupted output are saved when pending becomes false.
    if (!pending) saveMessages(messages);
  }, [messages, pending]);

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
    setMessages(current => [...current, { role: 'user', content }]);
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
            role: 'assistant',
            content: interrupted + '\n\n---\n\n*Response interrupted before completion; this may be incomplete.*',
          }];
        });
        setInput(content);
        setError(job.error || job.reply || 'The local request failed.');
        return;
      }
      const answer = job.reply || 'No response was returned.';
      setStreamingReply('');
      setMessages(current => [...current, { role: 'assistant', content: answer }]);
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
        <span className="chat-brand">SYNTRA</span>
      </header>

      <section
        ref={threadRef}
        onScroll={handleThreadScroll}
        className={'chat-thread ' + (messages.length === 0 ? 'chat-thread-empty' : '')}
        aria-live="polite"
      >
        {messages.length === 0 ? (
          <div className="chat-welcome">
            <h1>How can I help?</h1>
          </div>
        ) : (
          <div className="chat-messages">
            {messages.map((message, index) => (
              <article className={'chat-message ' + message.role} key={index}>
                <div className="chat-bubble">
                    {message.role === 'assistant'
                      ? <MessageContent content={message.content} />
                      : <div className="user-message-text">{message.content}</div>}
                  </div>
              </article>
            ))}
            {pending && streamingReply && (
              <article className="chat-message assistant streaming-message">
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
            disabled={pending}
          />
          <button className="send-button" type="submit" aria-label="Send message" disabled={pending || !input.trim()}>
            {pending ? <LoaderCircle size={18} className="thinking-spinner" /> : <ArrowUp size={18} strokeWidth={2.4} />}
          </button>
        </form>
      </div>
    </main>
  );
}

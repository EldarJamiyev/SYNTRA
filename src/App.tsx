import { FormEvent, KeyboardEvent, useEffect, useRef, useState } from 'react';
import { ArrowUp, LoaderCircle } from 'lucide-react';

type Message = { role: 'user' | 'assistant'; content: string };
type ChatJob = {
  jobId: string;
  status: 'running' | 'completed' | 'failed';
  phase?: string;
  message?: string;
  reply?: string;
  error?: string | null;
};

async function postJson<T>(url: string, body: unknown): Promise<T> {
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const result = await response.json() as T & { error?: string };
  if (!response.ok) throw new Error(result.error || 'The local chat service returned an error.');
  return result;
}

function MessageContent({ content }: { content: string }) {
  const parts = content.split(String.fromCharCode(96, 96, 96));
  return (
    <div className="message-content">
      {parts.map((part, index) => {
        if (index % 2 === 1) {
          const code = part.replace(/^[a-zA-Z0-9_+.-]*\n/, '');
          return <pre className="message-code" key={index}><code>{code}</code></pre>;
        }
        return part ? <span className="message-text" key={index}>{part}</span> : null;
      })}
    </div>
  );
}

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [pending, setPending] = useState(false);
  const [phase, setPhase] = useState('');
  const [error, setError] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    textareaRef.current?.focus();
    return () => { mountedRef.current = false; };
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [messages, pending, phase]);

  const send = async (event?: FormEvent) => {
    event?.preventDefault();
    const content = input.trim();
    if (!content || pending) return;

    const history = messages.slice(-12);
    setMessages(current => [...current, { role: 'user', content }]);
    setInput('');
    setError('');
    setPending(true);
    setPhase('Thinking');
    if (textareaRef.current) textareaRef.current.style.height = 'auto';

    try {
      const started = await postJson<ChatJob>('/local-api/chat', { message: content, history });
      let job = started;
      while (job.status === 'running') {
        if (!mountedRef.current) return;
        setPhase(job.message || 'Thinking');
        await new Promise(resolve => window.setTimeout(resolve, 800));
        job = await postJson<ChatJob>('/local-api/chat/status', { jobId: started.jobId });
      }
      if (!mountedRef.current) return;
      const answer = job.reply || job.error || 'No response was returned.';
      setMessages(current => [...current, { role: 'assistant', content: answer }]);
      if (job.status === 'failed') setError(job.error || job.reply || 'The local request failed.');
    } catch (cause) {
      if (mountedRef.current) setError(cause instanceof Error ? cause.message : String(cause));
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

      <section className={'chat-thread ' + (messages.length === 0 ? 'chat-thread-empty' : '')} aria-live="polite">
        {messages.length === 0 ? (
          <div className="chat-welcome">
            <h1>How can I help?</h1>
          </div>
        ) : (
          <div className="chat-messages">
            {messages.map((message, index) => (
              <article className={'chat-message ' + message.role} key={index}>
                <div className="chat-bubble"><MessageContent content={message.content} /></div>
              </article>
            ))}
            {pending && (
              <article className="chat-message assistant pending-message" role="status">
                <div className="pending-content">
                  <LoaderCircle size={15} className="thinking-spinner" />
                  <span>{phase || 'Thinking'}</span>
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
              event.currentTarget.style.height = 'auto';
              event.currentTarget.style.height = Math.min(event.currentTarget.scrollHeight, 180) + 'px';
            }}
            onKeyDown={onKeyDown}
            placeholder="Message SYNTRA"
            aria-label="Message SYNTRA"
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

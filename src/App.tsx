import { FormEvent, KeyboardEvent, useEffect, useRef, useState } from 'react';
import { BrainCircuit, LoaderCircle, Send } from 'lucide-react';

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
  if (!response.ok) throw new Error(result.error || 'SYNTRA local service request failed.');
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
    const nextMessages: Message[] = [...messages, { role: 'user', content }];
    setMessages(nextMessages);
    setInput('');
    setError('');
    setPending(true);
    setPhase('Connecting');
    if (textareaRef.current) textareaRef.current.style.height = 'auto';

    try {
      const started = await postJson<ChatJob>('/local-api/chat', { message: content, history });
      let job = started;
      while (job.status === 'running') {
        if (!mountedRef.current) return;
        setPhase(job.message || (job.phase === 'planning' ? 'Planning changes' : 'Thinking'));
        await new Promise(resolve => window.setTimeout(resolve, 900));
        job = await postJson<ChatJob>('/local-api/chat/status', { jobId: started.jobId });
      }
      if (!mountedRef.current) return;
      if (job.status === 'failed') {
        setError(job.error || job.reply || 'The local request failed.');
      }
      setMessages(current => [...current, {
        role: 'assistant',
        content: job.reply || job.error || 'No response was returned.',
      }]);
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
        <div className="chat-brand"><BrainCircuit size={23} strokeWidth={1.7} /><span>SYNTRA</span></div>
      </header>

      <section className={'chat-thread ' + (messages.length === 0 ? 'chat-thread-empty' : '')} aria-live="polite">
        {messages.length === 0 ? (
          <div className="chat-welcome">
            <div className="welcome-mark"><BrainCircuit size={29} strokeWidth={1.5} /></div>
            <h1>What are we working on?</h1>
            <p>Ask a question or tell SYNTRA what to build.</p>
          </div>
        ) : (
          <div className="chat-messages">
            {messages.map((message, index) => (
              <article className={'chat-message ' + message.role} key={index}>
                {message.role === 'assistant' && <div className="assistant-mark"><BrainCircuit size={17} /></div>}
                <div className="chat-bubble"><MessageContent content={message.content} /></div>
              </article>
            ))}
            {pending && (
              <article className="chat-message assistant pending-message">
                <div className="assistant-mark"><BrainCircuit size={17} /></div>
                <div className="pending-content"><LoaderCircle size={15} className="thinking-spinner" /><span>{phase || 'Thinking'}</span></div>
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
            {pending ? <LoaderCircle size={18} className="thinking-spinner" /> : <Send size={17} />}
          </button>
        </form>
        <div className="composer-hint">Enter to send · Shift + Enter for a new line</div>
      </div>
    </main>
  );
}

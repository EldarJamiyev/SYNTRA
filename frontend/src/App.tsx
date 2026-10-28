import { useCallback, useEffect, useState } from 'react';
import {
  Activity,
  ArrowUpRight,
  BrainCircuit,
  CheckCircle2,
  CircleAlert,
  Code2,
  Cpu,
  Database,
  Github,
  LoaderCircle,
  Play,
  RefreshCw,
  Server,
  TerminalSquare,
  Zap,
} from 'lucide-react';

type RuntimeStatus = {
  name: string;
  mode: string;
  checkpoint: string;
  checkpoint_exists: boolean;
  model_loaded: boolean;
  step: number | null;
  device: string | null;
  pretrained_weights: boolean;
  external_ai_api: boolean;
};

type TrainingStatus = {
  status: string;
  step: number;
  max_steps: number;
  train_loss: number | null;
  validation_loss: number | null;
  message: string;
};

function App() {
  const [status, setStatus] = useState<RuntimeStatus | null>(null);
  const [training, setTraining] = useState<TrainingStatus | null>(null);
  const [prompt, setPrompt] = useState("def fibonacci(n: int) -> list[int]:\n    ");
  const [maxTokens, setMaxTokens] = useState(96);
  const [temperature, setTemperature] = useState(0.8);
  const [output, setOutput] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [dataBusy, setDataBusy] = useState(false);
  const [trainingMessage, setTrainingMessage] = useState('');
  const [lastRun, setLastRun] = useState('');

  const refreshStatus = useCallback(async () => {
    try {
      const [statusResponse, trainingResponse] = await Promise.all([
        fetch('/api/status'),
        fetch('/api/train/status'),
      ]);
      if (!statusResponse.ok || !trainingResponse.ok) throw new Error('Backend status request failed');
      setStatus(await statusResponse.json() as RuntimeStatus);
      setTraining(await trainingResponse.json() as TrainingStatus);
    } catch {
      setStatus(null);
      setTraining(null);
    }
  }, []);

  useEffect(() => {
    void refreshStatus();
    const timer = window.setInterval(() => void refreshStatus(), 3000);
    return () => window.clearInterval(timer);
  }, [refreshStatus]);

  const prepareData = async () => {
    setDataBusy(true);
    setTrainingMessage('');
    try {
      const response = await fetch('/api/data/prepare', { method: 'POST' });
      const data = await response.json() as { files?: number; tokens?: number; output?: string; detail?: string };
      if (!response.ok) throw new Error(data.detail || 'Data preparation failed');
      setTrainingMessage('Prepared ' + String(data.files) + ' files · ' + String(data.tokens) + ' tokens.');
      await refreshStatus();
    } catch (cause) {
      setTrainingMessage(cause instanceof Error ? cause.message : 'Data preparation failed.');
    } finally {
      setDataBusy(false);
    }
  };

  const startTraining = async () => {
    setDataBusy(true);
    setTrainingMessage('');
    try {
      const response = await fetch('/api/train/start', { method: 'POST' });
      const data = await response.json() as TrainingStatus & { detail?: string };
      if (!response.ok) throw new Error(data.detail || 'Could not start training');
      setTrainingMessage('Training started. Metrics update at evaluation intervals.');
      await refreshStatus();
    } catch (cause) {
      setTrainingMessage(cause instanceof Error ? cause.message : 'Could not start training.');
    } finally {
      setDataBusy(false);
    }
  };

  const generate = async () => {
    setBusy(true);
    setError('');
    setOutput('');
    setLastRun('');
    try {
      const response = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, max_new_tokens: maxTokens, temperature, top_k: 40 }),
      });
      const data = await response.json() as { output?: string; detail?: string; step?: number; device?: string; warning?: string };
      if (!response.ok) throw new Error(data.detail || 'Generation failed with HTTP ' + response.status);
      setOutput(data.output || '');
      setLastRun('Step ' + String(data.step ?? '?') + ' · ' + String(data.device ?? 'local') + ' · locally generated');
      await refreshStatus();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Unknown generation error');
    } finally {
      setBusy(false);
    }
  };

  const modelReady = Boolean(status?.checkpoint_exists);
  const backendOnline = Boolean(status);
  const isTraining = training?.status === 'running';
  const trainingProgress = training && training.max_steps > 0
    ? Math.min(100, Math.round((training.step / training.max_steps) * 100))
    : 0;

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark"><BrainCircuit size={23} strokeWidth={1.7} /></div>
          <div>
            <div className="brand-name">SYNTRA<span>®</span></div>
            <div className="brand-subtitle">CODING INTELLIGENCE</div>
          </div>
        </div>
        <div className="workspace-label">WORKSPACE</div>
        <div className="nav-item active"><Code2 size={17} /> Playground <span className="nav-dot" /></div>
        <div className="nav-item muted"><Activity size={17} /> Runtime status</div>
        <div className="nav-item muted"><TerminalSquare size={17} /> CLI commands</div>
        <div className="sidebar-bottom">
          <div className="local-card">
            <div className="local-title"><span className="live-dot" /> LOCAL-FIRST</div>
            <p>No external AI API. The model runs from your own checkpoint.</p>
          </div>
          <a className="github-link" href="https://github.com/EldarJamiyev/SYNTRA" target="_blank" rel="noreferrer">
            <Github size={16} /> Repository <ArrowUpRight size={14} />
          </a>
          <div className="sidebar-version">SYNTRA STUDIO <span>v0.1.0</span></div>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div className="breadcrumb">SYNTRA <span>/</span> <strong>Studio</strong></div>
          <div className="topbar-status">
            <span className={backendOnline ? 'status-indicator online' : 'status-indicator'} />
            {backendOnline ? 'LOCAL RUNTIME CONNECTED' : 'BACKEND OFFLINE'}
            <button className="icon-button" title="Refresh runtime status" onClick={() => void refreshStatus()}><RefreshCw size={15} /></button>
          </div>
        </header>

        <section className="page-heading">
          <div className="eyebrow"><span className="eyebrow-line" /> INDEPENDENT MODEL LAB</div>
          <h1>Build. Train. <span>Generate.</span></h1>
          <p>A coding model trained from random initialization. No pretrained weights, no hosted inference, no external AI API.</p>
        </section>

        <section className="stats-grid">
          <div className="stat-card">
            <div className="stat-icon blue"><BrainCircuit size={18} /></div>
            <div className="stat-label">MODEL CHECKPOINT</div>
            <div className="stat-value">{modelReady ? 'AVAILABLE' : 'NOT TRAINED'}</div>
            <div className="stat-foot">{modelReady ? 'Local weights detected' : 'Train a model to begin'}</div>
          </div>
          <div className="stat-card">
            <div className="stat-icon violet"><Cpu size={18} /></div>
            <div className="stat-label">INFERENCE DEVICE</div>
            <div className="stat-value">{status?.device?.toUpperCase() || '—'}</div>
            <div className="stat-foot">{status?.model_loaded ? 'Model loaded in memory' : 'Loads on first request'}</div>
          </div>
          <div className="stat-card">
            <div className="stat-icon green"><Zap size={18} /></div>
            <div className="stat-label">TRAINING STEP</div>
            <div className="stat-value">{training?.step?.toLocaleString() ?? '—'}</div>
            <div className="stat-foot">{training?.max_steps ? 'Target: ' + training.max_steps.toLocaleString() : 'No active training run'}</div>
          </div>
          <div className="stat-card">
            <div className="stat-icon amber"><Server size={18} /></div>
            <div className="stat-label">INFERENCE MODE</div>
            <div className="stat-value">LOCAL ONLY</div>
            <div className="stat-foot">Loopback service · no cloud model</div>
          </div>
        </section>

        <section className="playground-grid">
          <div className="panel prompt-panel">
            <div className="panel-heading">
              <div className="panel-title"><Code2 size={17} /> Code playground</div>
              <span className="panel-kicker">PROMPT INPUT</span>
            </div>
            <label className="field-label" htmlFor="prompt">Your instruction or code prefix</label>
            <textarea id="prompt" className="code-input" value={prompt}
              onChange={(event) => setPrompt(event.target.value)} spellCheck={false}
              placeholder="Describe a programming task or enter a code prefix..." />
            <div className="controls-row">
              <label className="control-group">
                <span>MAX NEW TOKENS <strong>{maxTokens}</strong></span>
                <input type="range" min="16" max="256" step="16" value={maxTokens}
                  onChange={(event) => setMaxTokens(Number(event.target.value))} />
              </label>
              <label className="control-group">
                <span>TEMPERATURE <strong>{temperature.toFixed(1)}</strong></span>
                <input type="range" min="0.2" max="1.4" step="0.1" value={temperature}
                  onChange={(event) => setTemperature(Number(event.target.value))} />
              </label>
            </div>
            <div className="action-row">
              <span className="hint-text">Generation uses only the trained local checkpoint.</span>
              <button className="primary-button" onClick={() => void generate()} disabled={busy || !prompt.trim() || !modelReady}>
                {busy ? <LoaderCircle size={16} className="spin" /> : <Play size={15} fill="currentColor" />}
                {busy ? 'Generating' : 'Generate code'}
              </button>
            </div>
            {!modelReady && (
              <div className="notice notice-warning">
                <CircleAlert size={16} />
                <div><strong>No trained checkpoint found.</strong><p>Prepare local source data and train SYNTRA below before generating text.</p></div>
              </div>
            )}
            {!backendOnline && (
              <div className="notice notice-error">
                <CircleAlert size={16} />
                <div><strong>Backend not reachable.</strong><p>Start it with python -m syntra.cli serve --host 127.0.0.1 --port 8000.</p></div>
              </div>
            )}
            {error && <div className="notice notice-error"><CircleAlert size={16} /><div><strong>Request failed</strong><p>{error}</p></div></div>}
          </div>

          <div className="panel output-panel">
            <div className="panel-heading">
              <div className="panel-title"><TerminalSquare size={17} /> Model output</div>
              <span className="panel-kicker">GENERATED TEXT</span>
            </div>
            <div className="output-area">
              {busy ? (
                <div className="empty-state"><LoaderCircle size={25} className="spin" /><strong>Generating locally...</strong><p>The model is sampling one token at a time.</p></div>
              ) : output ? (
                <pre className="output-code">{output}</pre>
              ) : (
                <div className="empty-state">
                  <div className="empty-icon"><Code2 size={24} /></div>
                  <strong>Your output appears here</strong>
                  <p>Train your own model, write a prompt, and run generation.</p>
                </div>
              )}
            </div>
            <div className="output-footer">
              <span>{lastRun || 'Waiting for a generation request'}</span>
              {output && <button className="text-button" onClick={() => void navigator.clipboard.writeText(output)}>Copy output</button>}
            </div>
          </div>
        </section>

        <section className="bottom-grid">
          <div className="panel info-panel">
            <div className="panel-title"><CheckCircle2 size={17} /> Model independence</div>
            <div className="check-list">
              <div><CheckCircle2 size={15} /><span>Randomly initialized model weights</span><b>YES</b></div>
              <div><CheckCircle2 size={15} /><span>Custom tokenizer and transformer code</span><b>YES</b></div>
              <div><CheckCircle2 size={15} /><span>External inference APIs</span><b className="negative">NONE</b></div>
              <div><CheckCircle2 size={15} /><span>Generated code execution by server</span><b className="neutral">DISABLED</b></div>
            </div>
          </div>
          <div className="panel training-panel">
            <div className="panel-title"><Database size={17} /> Data & training</div>
            <p>Put permitted source code in data/raw, then prepare it for training.</p>
            <div className="training-state"><span className={isTraining ? 'status-indicator online' : 'status-indicator'} /> {training?.status?.toUpperCase() || 'BACKEND OFFLINE'}</div>
            <div className="progress-track"><div className="progress-fill" style={{ width: trainingProgress + '%' }} /></div>
            <div className="training-metrics">
              <span>Train loss <strong>{training?.train_loss == null ? '—' : training.train_loss.toFixed(4)}</strong></span>
              <span>Val loss <strong>{training?.validation_loss == null ? '—' : training.validation_loss.toFixed(4)}</strong></span>
            </div>
            <div className="training-message">{trainingMessage || training?.message || 'Waiting for the local runtime.'}</div>
            <button className="secondary-button" disabled={dataBusy || isTraining || !backendOnline} onClick={() => void prepareData()}>
              {dataBusy ? <LoaderCircle size={14} className="spin" /> : <Database size={14} />} Prepare local data
            </button>
            <button className="primary-button training-button" disabled={dataBusy || isTraining || !backendOnline} onClick={() => void startTraining()}>
              {isTraining ? <LoaderCircle size={14} className="spin" /> : <Play size={14} />} {isTraining ? 'Training in progress' : 'Start training'}
            </button>
          </div>
          <div className="panel commands-panel">
            <div className="panel-title"><TerminalSquare size={17} /> CLI commands</div>
            <p>All functions are also available from PowerShell.</p>
            <div className="command-line">python -m syntra.cli prepare</div>
            <div className="command-line">python -m syntra.cli train --config config/default.json</div>
            <div className="command-line">python -m syntra.cli serve</div>
          </div>
        </section>

        <footer className="page-footer">
          <span>© 2026 SYNTRA · AN EXPERIMENT IN INDEPENDENT CODING INTELLIGENCE</span>
          <span><span className="live-dot" /> LOCAL-FIRST BY DESIGN</span>
        </footer>
      </main>
    </div>
  );
}

export default App;

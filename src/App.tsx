import { useCallback, useEffect, useState, useRef } from 'react';
import {
  Activity,
  ArrowUpRight,
  BrainCircuit,
  CheckCircle2,
  CircleAlert,
  Code2,
  Cpu,
  Database,
  Download,
  Github,
  LoaderCircle,
  Play,
  Pause,
  Plus,
  RefreshCw,
  RotateCcw,
  Server,
  TerminalSquare,
  Trash2,
  Zap,
} from 'lucide-react';

import {
  syntraEngine,
  RuntimeStatus,
  TrainingStatus,
  MetricRow,
  EvaluationResult,
} from './syntra/engine';
import { calculateParameterCount } from './syntra/config';
import { tokenizer } from './syntra/tokenizer';
import { CorpusFile } from './syntra/data';

type Tab = 'playground' | 'runtime' | 'training' | 'cli';

const CODE_TEMPLATES: Record<string, string> = {
  fibonacci: 'def fibonacci(n: int) -> list[int]:\n    ',
  binary_search: 'def binary_search(array: list[int], target: int) -> int:\n    ',
  quicksort: 'def quicksort(items: list[int]) -> list[int]:\n    ',
  queue_class: 'class Queue:\n    def __init__(self):\n        ',
  http_handler: 'class ApiHandler(BaseHTTPRequestHandler):\n    def do_GET(self):\n        ',
};

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>('playground');
  const [status, setStatus] = useState<RuntimeStatus>(() => syntraEngine.getRuntimeStatus());
  const [training, setTraining] = useState<TrainingStatus>(() => syntraEngine.getTrainingStatus());
  const [metrics, setMetrics] = useState<MetricRow[]>(() => syntraEngine.getMetricsHistory());
  const [config, setConfig] = useState(() => syntraEngine.getConfig());
  const [corpusFiles, setCorpusFiles] = useState<CorpusFile[]>(() => syntraEngine.getCorpusFiles());

  // Playground state
  const [prompt, setPrompt] = useState('def fibonacci(n: int) -> list[int]:\n    ');
  const [maxTokens, setMaxTokens] = useState(96);
  const [temperature, setTemperature] = useState(0.8);
  const [topK, setTopK] = useState(40);
  const [output, setOutput] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [dataBusy, setDataBusy] = useState(false);
  const [trainingMessage, setTrainingMessage] = useState('');
  const [lastRun, setLastRun] = useState('');
  const [copyFeedback, setCopyFeedback] = useState(false);

  // Tokenizer playground in Runtime tab
  const [tokenizerInput, setTokenizerInput] = useState('def add(a, b):\n    return a + b');
  const [encodedTokens, setEncodedTokens] = useState<number[]>([]);

  // Corpus management
  const [selectedFileId, setSelectedFileId] = useState<string>('f1');
  const [newFileName, setNewFileName] = useState('');
  const [newFileContent, setNewFileContent] = useState('');
  const [showAddFile, setShowAddFile] = useState(false);

  // Evaluation state
  const [evalResult, setEvalResult] = useState<EvaluationResult | null>(null);

  // CLI state
  const [terminalHistory, setTerminalHistory] = useState<
    Array<{ type: 'command' | 'output' | 'error' | 'success'; text: string }>
  >([
    { type: 'output', text: 'SYNTRA Local CLI Environment [Version 0.1.0]' },
    { type: 'output', text: 'Type "syntra --help" or click a suggested command below.\n' },
  ]);
  const [cliInput, setCliInput] = useState('');
  const terminalEndRef = useRef<HTMLDivElement>(null);

  // Sync engine state
  const refreshStatus = useCallback(() => {
    setStatus(syntraEngine.getRuntimeStatus());
    setTraining(syntraEngine.getTrainingStatus());
    setMetrics(syntraEngine.getMetricsHistory());
    setCorpusFiles(syntraEngine.getCorpusFiles());
    setConfig(syntraEngine.getConfig());
  }, []);

  useEffect(() => {
    const unsubscribe = syntraEngine.subscribe(() => {
      refreshStatus();
    });
    refreshStatus();
    return () => {
      unsubscribe();
    };
  }, [refreshStatus]);

  // Update tokenizer playground tokens
  useEffect(() => {
    try {
      const tokens = tokenizer.encode(tokenizerInput, { addBos: true, addEos: false });
      setEncodedTokens(tokens);
    } catch {
      setEncodedTokens([]);
    }
  }, [tokenizerInput]);

  useEffect(() => {
    if (activeTab === 'cli') {
      terminalEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [terminalHistory, activeTab]);

  const prepareData = () => {
    setDataBusy(true);
    setTrainingMessage('');
    try {
      const prep = syntraEngine.prepareData();
      setTrainingMessage(`Prepared ${prep.files} files · ${prep.tokens} tokens.`);
      refreshStatus();
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
      setTrainingMessage('Training started. Optimizing transformer weights...');
      await syntraEngine.startTraining();
    } catch (cause) {
      setTrainingMessage(cause instanceof Error ? cause.message : 'Could not start training.');
    } finally {
      setDataBusy(false);
    }
  };

  const pauseTraining = () => {
    syntraEngine.pauseTraining();
    setTrainingMessage('Training paused.');
  };

  const resetModel = () => {
    syntraEngine.resetCheckpoint();
    setOutput('');
    setLastRun('');
    setEvalResult(null);
    setTrainingMessage('Checkpoint reset to random initialization.');
  };

  const generate = () => {
    setBusy(true);
    setError('');
    setOutput('');
    setLastRun('');

    // Brief timeout to let UI update loader
    setTimeout(() => {
      try {
        const result = syntraEngine.generate(prompt, {
          maxNewTokens: maxTokens,
          temperature,
          topK,
        });
        setOutput(result.output);
        setLastRun(
          `Step ${result.step} · ${result.device} · ${result.tokensGenerated} tokens generated`
        );
      } catch (cause) {
        setError(cause instanceof Error ? cause.message : 'Generation failed');
      } finally {
        setBusy(false);
      }
    }, 40);
  };

  const copyToClipboard = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopyFeedback(true);
      setTimeout(() => setCopyFeedback(false), 2000);
    } catch {
      // Fallback
    }
  };

  const runEvaluation = () => {
    try {
      const res = syntraEngine.evaluate();
      setEvalResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Evaluation failed');
    }
  };

  const handleCliSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const cmd = cliInput.trim();
    if (!cmd) return;
    executeCliCommand(cmd);
    setCliInput('');
  };

  const executeCliCommand = (cmd: string) => {
    const history = [...terminalHistory, { type: 'command' as const, text: `$ ${cmd}` }];

    if (cmd === 'clear') {
      setTerminalHistory([]);
      return;
    }

    if (cmd === 'syntra --help' || cmd === 'python -m syntra.cli --help') {
      history.push({
        type: 'output',
        text: `SYNTRA CLI Usage:
  syntra prepare [--raw-dir data/raw] [--out data/processed/tokens.bin]
  syntra train [--config config/default.json] [--resume checkpoints/latest.pt]
  syntra generate --checkpoint checkpoints/latest.pt --prompt "..." [--max-new-tokens 80]
  syntra evaluate [--checkpoint checkpoints/latest.pt]
  syntra serve [--host 127.0.0.1] [--port 8000]
  clear`,
      });
    } else if (cmd.includes('prepare')) {
      try {
        const prep = syntraEngine.prepareData();
        history.push({
          type: 'success',
          text: `[OK] Corpus prepared successfully.\nFiles processed: ${prep.files}\nTotal tokens: ${prep.tokens}\nSaved token stream to ${prep.output}`,
        });
      } catch (err) {
        history.push({ type: 'error', text: `[Error] ${String(err)}` });
      }
    } else if (cmd.includes('train')) {
      history.push({
        type: 'output',
        text: 'Starting training loop in background from random initialization...',
      });
      startTraining();
      history.push({
        type: 'success',
        text: `SYNTRA training started with device: ${status.device}.\nMetrics logging to checkpoints/training_metrics.csv. Check the Training tab or Playground stats.`,
      });
    } else if (cmd.includes('generate')) {
      if (!status.checkpoint_exists) {
        history.push({
          type: 'error',
          text: 'Error: No trained checkpoint found. Run "syntra train" first.',
        });
      } else {
        const promptMatch = cmd.match(/--prompt\s+["']([^"']+)["']/);
        const p = promptMatch ? promptMatch[1] : 'def add(a, b):';
        try {
          const res = syntraEngine.generate(p, { maxNewTokens: 64, temperature: 0.8 });
          history.push({
            type: 'output',
            text: `--- Generated (${res.tokensGenerated} tokens at step ${res.step}) ---\n${res.output}`,
          });
        } catch (err) {
          history.push({ type: 'error', text: `[Error] ${String(err)}` });
        }
      }
    } else if (cmd.includes('evaluate')) {
      try {
        const ev = syntraEngine.evaluate();
        history.push({
          type: 'success',
          text: `Evaluation results for ${ev.checkpoint}:\nStep: ${ev.step}\nEvaluation batches: ${ev.evaluation_batches}\nCross entropy: ${ev.cross_entropy}\nPerplexity: ${ev.perplexity}`,
        });
      } catch (err) {
        history.push({ type: 'error', text: `[Error] ${String(err)}` });
      }
    } else if (cmd.includes('serve')) {
      history.push({
        type: 'success',
        text: 'SYNTRA Local Runtime loopback server is running on 127.0.0.1:3000 (status: CONNECTED).',
      });
    } else {
      history.push({
        type: 'error',
        text: `Command not recognized: "${cmd}". Type "syntra --help" for available commands.`,
      });
    }

    setTerminalHistory(history);
  };

  const handleAddFile = (e: React.FormEvent) => {
    e.preventDefault();
    if (!newFileName.trim() || !newFileContent.trim()) return;
    const ext = newFileName.includes('.') ? newFileName.split('.').pop() || 'python' : 'python';
    syntraEngine.addCorpusFile({
      name: newFileName.trim(),
      language: ext,
      content: newFileContent,
    });
    setNewFileName('');
    setNewFileContent('');
    setShowAddFile(false);
  };

  const exportMetricsCsv = () => {
    if (metrics.length === 0) return;
    const headers = 'step,train_loss,validation_loss,learning_rate,seconds\n';
    const rows = metrics
      .map(
        (m) =>
          `${m.step},${m.train_loss},${m.validation_loss},${m.learning_rate.toExponential(3)},${m.seconds}`
      )
      .join('\n');
    const blob = new Blob([headers + rows], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'training_metrics.csv';
    a.click();
    URL.revokeObjectURL(url);
  };

  const modelReady = Boolean(status.checkpoint_exists);
  const backendOnline = true; // In-browser client engine is always active
  const isTraining = training.status === 'running';
  const trainingProgress =
    training && training.max_steps > 0
      ? Math.min(100, Math.round((training.step / training.max_steps) * 100))
      : 0;

  const paramCounts = calculateParameterCount(config);
  const selectedCorpusFile = corpusFiles.find((f) => f.id === selectedFileId) || corpusFiles[0];

  return (
    <div className="app-shell">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <BrainCircuit size={23} strokeWidth={1.7} />
          </div>
          <div>
            <div className="brand-name">
              SYNTRA<span>®</span>
            </div>
            <div className="brand-subtitle">CODING INTELLIGENCE</div>
          </div>
        </div>

        <div className="workspace-label">WORKSPACE</div>
        <button
          className={`nav-item ${activeTab === 'playground' ? 'active' : ''}`}
          onClick={() => setActiveTab('playground')}
        >
          <Code2 size={17} /> Playground {activeTab === 'playground' && <span className="nav-dot" />}
        </button>
        <button
          className={`nav-item ${activeTab === 'runtime' ? 'active' : ''}`}
          onClick={() => setActiveTab('runtime')}
        >
          <Activity size={17} /> Runtime & Model{' '}
          {activeTab === 'runtime' && <span className="nav-dot" />}
        </button>
        <button
          className={`nav-item ${activeTab === 'training' ? 'active' : ''}`}
          onClick={() => setActiveTab('training')}
        >
          <Database size={17} /> Data & Training{' '}
          {activeTab === 'training' && <span className="nav-dot" />}
        </button>
        <button
          className={`nav-item ${activeTab === 'cli' ? 'active' : ''}`}
          onClick={() => setActiveTab('cli')}
        >
          <TerminalSquare size={17} /> CLI Commands{' '}
          {activeTab === 'cli' && <span className="nav-dot" />}
        </button>

        <div className="sidebar-bottom">
          <div className="local-card">
            <div className="local-title">
              <span className="live-dot" /> LOCAL-FIRST
            </div>
            <p>No external AI API. The model runs from your own checkpoint.</p>
          </div>
          <a
            className="github-link"
            href="https://github.com/EldarJamiyev/SYNTRA"
            target="_blank"
            rel="noreferrer"
          >
            <Github size={16} /> Repository <ArrowUpRight size={14} />
          </a>
          <div className="sidebar-version">
            SYNTRA STUDIO <span>v0.1.0</span>
          </div>
        </div>
      </aside>

      {/* Main Area */}
      <main className="main-content">
        <header className="topbar">
          <div className="breadcrumb">
            SYNTRA <span>/</span>{' '}
            <strong>
              {activeTab === 'playground' && 'Studio Playground'}
              {activeTab === 'runtime' && 'Runtime & Architecture'}
              {activeTab === 'training' && 'Data & Training Engine'}
              {activeTab === 'cli' && 'Interactive CLI Terminal'}
            </strong>
          </div>
          <div className="topbar-status">
            <span className={backendOnline ? 'status-indicator online' : 'status-indicator'} />
            {backendOnline ? 'LOCAL RUNTIME CONNECTED' : 'BACKEND OFFLINE'}
            <button
              className="icon-button"
              title="Refresh runtime status"
              onClick={() => refreshStatus()}
            >
              <RefreshCw size={15} />
            </button>
          </div>
        </header>

        <section className="page-heading">
          <div className="eyebrow">
            <span className="eyebrow-line" /> INDEPENDENT MODEL LAB
          </div>
          <h1>
            Build. Train. <span>Generate.</span>
          </h1>
          <p>
            A coding model trained from random initialization. No pretrained weights, no hosted
            inference, no external AI API.
          </p>
        </section>

        {/* Top Stats Grid */}
        <section className="stats-grid">
          <div className="stat-card">
            <div className="stat-icon blue">
              <BrainCircuit size={18} />
            </div>
            <div className="stat-label">MODEL CHECKPOINT</div>
            <div className="stat-value">{modelReady ? 'AVAILABLE' : 'NOT TRAINED'}</div>
            <div className="stat-foot">
              {modelReady
                ? `Step ${training.step} weights ready`
                : 'Train a model to begin'}
            </div>
          </div>
          <div className="stat-card">
            <div className="stat-icon violet">
              <Cpu size={18} />
            </div>
            <div className="stat-label">INFERENCE DEVICE</div>
            <div className="stat-value">{status.device?.toUpperCase() || '—'}</div>
            <div className="stat-foot">
              {modelReady ? 'Loaded in memory' : 'Random initialization'}
            </div>
          </div>
          <div className="stat-card">
            <div className="stat-icon green">
              <Zap size={18} />
            </div>
            <div className="stat-label">TRAINING STEP</div>
            <div className="stat-value">{training.step.toLocaleString()}</div>
            <div className="stat-foot">
              {training.max_steps
                ? `Target: ${training.max_steps.toLocaleString()} steps`
                : 'No active training run'}
            </div>
          </div>
          <div className="stat-card">
            <div className="stat-icon amber">
              <Server size={18} />
            </div>
            <div className="stat-label">INFERENCE MODE</div>
            <div className="stat-value">LOCAL ONLY</div>
            <div className="stat-foot">Client sandbox · no cloud model</div>
          </div>
        </section>

        {/* TAB 1: PLAYGROUND */}
        {activeTab === 'playground' && (
          <>
            <section className="playground-grid">
              {/* Input Panel */}
              <div className="panel prompt-panel">
                <div className="panel-heading">
                  <div className="panel-title">
                    <Code2 size={17} /> Code playground
                  </div>
                  <span className="panel-kicker">PROMPT INPUT</span>
                </div>

                <div className="field-header">
                  <label className="field-label" htmlFor="prompt">
                    Your instruction or code prefix
                  </label>
                  <select
                    className="template-select"
                    onChange={(e) => {
                      if (CODE_TEMPLATES[e.target.value]) {
                        setPrompt(CODE_TEMPLATES[e.target.value]);
                      }
                    }}
                    defaultValue="fibonacci"
                  >
                    <option value="" disabled>
                      Load code snippet...
                    </option>
                    <option value="fibonacci">Fibonacci Function</option>
                    <option value="binary_search">Binary Search</option>
                    <option value="quicksort">QuickSort</option>
                    <option value="queue_class">Queue Class</option>
                    <option value="http_handler">HTTP Server Handler</option>
                  </select>
                </div>

                <textarea
                  id="prompt"
                  className="code-input"
                  value={prompt}
                  onChange={(event) => setPrompt(event.target.value)}
                  spellCheck={false}
                  placeholder="Describe a programming task or enter a code prefix..."
                />

                <div className="controls-row">
                  <label className="control-group">
                    <span>
                      MAX NEW TOKENS <strong>{maxTokens}</strong>
                    </span>
                    <input
                      type="range"
                      min="16"
                      max="256"
                      step="16"
                      value={maxTokens}
                      onChange={(event) => setMaxTokens(Number(event.target.value))}
                    />
                  </label>
                  <label className="control-group">
                    <span>
                      TEMPERATURE <strong>{temperature.toFixed(1)}</strong>
                    </span>
                    <input
                      type="range"
                      min="0.2"
                      max="1.4"
                      step="0.1"
                      value={temperature}
                      onChange={(event) => setTemperature(Number(event.target.value))}
                    />
                  </label>
                  <label className="control-group">
                    <span>
                      TOP-K <strong>{topK}</strong>
                    </span>
                    <input
                      type="range"
                      min="10"
                      max="100"
                      step="5"
                      value={topK}
                      onChange={(event) => setTopK(Number(event.target.value))}
                    />
                  </label>
                </div>

                <div className="action-row">
                  <span className="hint-text">Generation uses only the trained local checkpoint.</span>
                  <button
                    className="primary-button"
                    onClick={() => generate()}
                    disabled={busy || !prompt.trim() || !modelReady}
                  >
                    {busy ? (
                      <LoaderCircle size={16} className="spin" />
                    ) : (
                      <Play size={15} fill="currentColor" />
                    )}
                    {busy ? 'Generating' : 'Generate code'}
                  </button>
                </div>

                {!modelReady && (
                  <div className="notice notice-warning">
                    <CircleAlert size={16} />
                    <div>
                      <strong>No trained checkpoint found.</strong>
                      <p>
                        Prepare local source data and train SYNTRA below before generating text.
                      </p>
                    </div>
                  </div>
                )}
                {error && (
                  <div className="notice notice-error">
                    <CircleAlert size={16} />
                    <div>
                      <strong>Request failed</strong>
                      <p>{error}</p>
                    </div>
                  </div>
                )}
              </div>

              {/* Output Panel */}
              <div className="panel output-panel">
                <div className="panel-heading">
                  <div className="panel-title">
                    <TerminalSquare size={17} /> Model output
                  </div>
                  <span className="panel-kicker">GENERATED TEXT</span>
                </div>
                <div className="output-area">
                  {busy ? (
                    <div className="empty-state">
                      <LoaderCircle size={25} className="spin" />
                      <strong>Generating locally...</strong>
                      <p>The model is sampling one token at a time.</p>
                    </div>
                  ) : output ? (
                    <pre className="output-code">{output}</pre>
                  ) : (
                    <div className="empty-state">
                      <div className="empty-icon">
                        <Code2 size={24} />
                      </div>
                      <strong>Your output appears here</strong>
                      <p>Train your own model, write a prompt, and run generation.</p>
                    </div>
                  )}
                </div>
                <div className="output-footer">
                  <span>{lastRun || 'Waiting for a generation request'}</span>
                  {output && (
                    <button
                      className="text-button"
                      onClick={() => void copyToClipboard(output)}
                    >
                      {copyFeedback ? 'Copied!' : 'Copy output'}
                    </button>
                  )}
                </div>
              </div>
            </section>

            {/* Bottom 3 Cards */}
            <section className="bottom-grid">
              <div className="panel info-panel">
                <div className="panel-title">
                  <CheckCircle2 size={17} /> Model independence
                </div>
                <div className="check-list">
                  <div>
                    <CheckCircle2 size={15} />
                    <span>Randomly initialized model weights</span>
                    <b>YES</b>
                  </div>
                  <div>
                    <CheckCircle2 size={15} />
                    <span>Custom tokenizer and transformer code</span>
                    <b>YES</b>
                  </div>
                  <div>
                    <CheckCircle2 size={15} />
                    <span>External inference APIs</span>
                    <b className="negative">NONE</b>
                  </div>
                  <div>
                    <CheckCircle2 size={15} />
                    <span>Generated code execution by server</span>
                    <b className="neutral">DISABLED</b>
                  </div>
                </div>
              </div>

              <div className="panel training-panel">
                <div className="panel-title">
                  <Database size={17} /> Data & training
                </div>
                <p>Curated code is ready in data/raw. Prepare it and train from scratch.</p>
                <div className="training-state">
                  <span
                    className={isTraining ? 'status-indicator online' : 'status-indicator'}
                  />{' '}
                  {training.status.toUpperCase()}
                </div>
                <div className="progress-track">
                  <div className="progress-fill" style={{ width: trainingProgress + '%' }} />
                </div>
                <div className="training-metrics">
                  <span>
                    Train loss{' '}
                    <strong>
                      {training.train_loss == null ? '—' : training.train_loss.toFixed(4)}
                    </strong>
                  </span>
                  <span>
                    Val loss{' '}
                    <strong>
                      {training.validation_loss == null
                        ? '—'
                        : training.validation_loss.toFixed(4)}
                    </strong>
                  </span>
                </div>
                <div className="training-message">
                  {trainingMessage || training.message || 'Waiting for training.'}
                </div>
                <button
                  className="secondary-button"
                  disabled={dataBusy || isTraining}
                  onClick={() => prepareData()}
                >
                  {dataBusy ? (
                    <LoaderCircle size={14} className="spin" />
                  ) : (
                    <Database size={14} />
                  )}{' '}
                  Prepare local data
                </button>
                {isTraining ? (
                  <button
                    className="secondary-button"
                    style={{ marginTop: 8 }}
                    onClick={() => pauseTraining()}
                  >
                    <Pause size={14} /> Pause training
                  </button>
                ) : (
                  <button
                    className="primary-button training-button"
                    disabled={dataBusy}
                    onClick={() => void startTraining()}
                  >
                    <Play size={14} />{' '}
                    {training.step > 0 && training.step < training.max_steps
                      ? 'Resume training'
                      : 'Start training'}
                  </button>
                )}
              </div>

              <div className="panel commands-panel">
                <div className="panel-title">
                  <TerminalSquare size={17} /> CLI commands
                </div>
                <p>Click any command to inspect or execute it in the CLI tab.</p>
                <div
                  className="command-line"
                  onClick={() => {
                    setActiveTab('cli');
                    executeCliCommand('syntra prepare');
                  }}
                >
                  python -m syntra.cli prepare
                </div>
                <div
                  className="command-line"
                  onClick={() => {
                    setActiveTab('cli');
                    executeCliCommand('syntra train --config config/default.json');
                  }}
                >
                  python -m syntra.cli train --config config/default.json
                </div>
                <div
                  className="command-line"
                  onClick={() => {
                    setActiveTab('cli');
                    executeCliCommand('syntra evaluate');
                  }}
                >
                  python -m syntra.cli evaluate
                </div>
              </div>
            </section>
          </>
        )}

        {/* TAB 2: RUNTIME & ARCHITECTURE */}
        {activeTab === 'runtime' && (
          <div className="playground-grid">
            <div className="panel">
              <div className="panel-heading">
                <div className="panel-title">
                  <Cpu size={17} /> Model Architecture Specification
                </div>
                <span className="panel-kicker">7.2M PARAMETERS</span>
              </div>

              <div className="arch-card">
                <div className="arch-header">
                  <div className="arch-title">Decoder-Only Causal Transformer</div>
                  <span className="tag-badge">AUTOREGRESSIVE</span>
                </div>
                <p className="hint-text" style={{ marginTop: 4 }}>
                  Trained strictly from random Gaussian weights (mean=0.0, std=0.02). Weight tying
                  between token embeddings and final linear language modeling head.
                </p>
                <div className="arch-specs">
                  <div className="arch-spec-item">
                    TOTAL PARAMETERS
                    <strong>{paramCounts.total.toLocaleString()}</strong>
                  </div>
                  <div className="arch-spec-item">
                    VOCABULARY SIZE
                    <strong>{config.vocab_size} tokens</strong>
                  </div>
                  <div className="arch-spec-item">
                    EMBEDDING DIM (n_embd)
                    <strong>{config.n_embd}</strong>
                  </div>
                  <div className="arch-spec-item">
                    TRANSFORMER LAYERS
                    <strong>{config.n_layer} blocks</strong>
                  </div>
                  <div className="arch-spec-item">
                    ATTENTION HEADS
                    <strong>{config.n_head} heads</strong>
                  </div>
                  <div className="arch-spec-item">
                    CONTEXT WINDOW
                    <strong>{config.block_size} tokens</strong>
                  </div>
                </div>
              </div>

              <div className="arch-card" style={{ marginTop: 14 }}>
                <div className="arch-header">
                  <div className="arch-title">Active Checkpoint Status</div>
                  <span className="tag-badge">
                    {modelReady ? 'LOADED IN MEMORY' : 'RANDOM INITIALIZATION'}
                  </span>
                </div>
                <div className="arch-specs">
                  <div className="arch-spec-item">
                    CHECKPOINT FILE
                    <strong>{status.checkpoint}</strong>
                  </div>
                  <div className="arch-spec-item">
                    CURRENT STEP
                    <strong>{training.step} / {config.max_steps}</strong>
                  </div>
                  <div className="arch-spec-item">
                    TRAIN LOSS
                    <strong>{training.train_loss?.toFixed(4) ?? 'None'}</strong>
                  </div>
                  <div className="arch-spec-item">
                    VAL LOSS
                    <strong>{training.validation_loss?.toFixed(4) ?? 'None'}</strong>
                  </div>
                </div>

                <div style={{ marginTop: 16, display: 'flex', gap: 10 }}>
                  <button
                    className="danger-button"
                    onClick={resetModel}
                    title="Reset model checkpoint to random weights"
                  >
                    <RotateCcw size={14} /> Reset to Random Initialization
                  </button>
                  <button
                    className="secondary-button"
                    style={{ width: 'auto', marginTop: 0 }}
                    onClick={runEvaluation}
                    disabled={!modelReady}
                  >
                    <CheckCircle2 size={14} /> Run Validation Evaluation
                  </button>
                </div>

                {evalResult && (
                  <div className="notice notice-success" style={{ marginTop: 14 }}>
                    <CheckCircle2 size={16} />
                    <div>
                      <strong>Evaluation Complete</strong>
                      <p>
                        Cross-Entropy Loss: {evalResult.cross_entropy} · Perplexity:{' '}
                        {evalResult.perplexity} ({evalResult.evaluation_batches} validation batches)
                      </p>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Tokenizer Sandbox */}
            <div className="panel">
              <div className="panel-heading">
                <div className="panel-title">
                  <BrainCircuit size={17} /> Byte Tokenizer Sandbox
                </div>
                <span className="panel-kicker">UTF-8 LOSSLESS</span>
              </div>
              <p className="hint-text">
                SYNTRA uses a 260-token UTF-8 byte tokenizer. Reserved tokens: 0=&lt;PAD&gt;,
                1=&lt;BOS&gt;, 2=&lt;EOS&gt;, 3=&lt;UNK&gt;. Bytes 0–255 are mapped to token IDs
                4–259 (BYTE_OFFSET = 4).
              </p>

              <label className="field-label" style={{ marginTop: 12 }}>
                Live Tokenizer Text Input
              </label>
              <textarea
                className="code-input"
                style={{ minHeight: 110 }}
                value={tokenizerInput}
                onChange={(e) => setTokenizerInput(e.target.value)}
                spellCheck={false}
              />

              <div style={{ marginTop: 14 }}>
                <span className="field-label">
                  Encoded Tokens ({encodedTokens.length} tokens):
                </span>
                <div
                  style={{
                    background: '#090d14',
                    padding: 12,
                    borderRadius: 8,
                    border: '1px solid #222d3e',
                    maxHeight: 180,
                    overflowY: 'auto',
                    display: 'flex',
                    flexWrap: 'wrap',
                    gap: 6,
                  }}
                >
                  {encodedTokens.map((t, idx) => {
                    const charStr = tokenizer.tokenToString(t);
                    return (
                      <span
                        key={idx}
                        style={{
                          background: '#152132',
                          border: '1px solid #293d5c',
                          borderRadius: 4,
                          padding: '2px 6px',
                          fontSize: 10,
                          fontFamily: 'var(--mono)',
                          color: t < 4 ? '#f6a7b2' : '#9fc7ff',
                        }}
                        title={`Token ID: ${t}`}
                      >
                        {t}: {charStr === ' ' ? '␣' : charStr === '\n' ? '↵' : charStr}
                      </span>
                    );
                  })}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 3: DATA & TRAINING */}
        {activeTab === 'training' && (
          <div className="playground-grid">
            {/* Corpus File Management */}
            <div className="panel">
              <div className="panel-heading">
                <div className="panel-title">
                  <Database size={17} /> Raw Training Corpus (data/raw)
                </div>
                <span className="panel-kicker">LOCAL SOURCE FILES</span>
              </div>

              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <p className="hint-text">
                  Corpus files used to encode tokens into <code>tokens.bin</code>.
                </p>
                <button
                  className="secondary-button"
                  style={{ width: 'auto', marginTop: 0, padding: '6px 10px' }}
                  onClick={() => setShowAddFile(!showAddFile)}
                >
                  <Plus size={14} /> Add Source File
                </button>
              </div>

              {showAddFile && (
                <form
                  onSubmit={handleAddFile}
                  style={{
                    background: '#0d131e',
                    border: '1px solid #27374d',
                    borderRadius: 8,
                    padding: 14,
                    marginTop: 12,
                  }}
                >
                  <label className="field-label">File Name (e.g. math_ops.py, utils.ts)</label>
                  <input
                    type="text"
                    className="code-input"
                    style={{ minHeight: 'auto', padding: '8px 12px', marginBottom: 10 }}
                    placeholder="example.py"
                    value={newFileName}
                    onChange={(e) => setNewFileName(e.target.value)}
                    required
                  />
                  <label className="field-label">Source Code</label>
                  <textarea
                    className="code-input"
                    style={{ minHeight: 120, marginBottom: 10 }}
                    placeholder="def custom_code():\n    pass"
                    value={newFileContent}
                    onChange={(e) => setNewFileContent(e.target.value)}
                    required
                  />
                  <div style={{ display: 'flex', gap: 8 }}>
                    <button type="submit" className="primary-button">
                      Save to Corpus
                    </button>
                    <button
                      type="button"
                      className="secondary-button"
                      style={{ width: 'auto', marginTop: 0 }}
                      onClick={() => setShowAddFile(false)}
                    >
                      Cancel
                    </button>
                  </div>
                </form>
              )}

              <table className="data-table">
                <thead>
                  <tr>
                    <th>FILENAME</th>
                    <th>LANG</th>
                    <th>BYTES</th>
                    <th>ACTIONS</th>
                  </tr>
                </thead>
                <tbody>
                  {corpusFiles.map((file) => (
                    <tr
                      key={file.id}
                      style={{
                        background: selectedFileId === file.id ? '#131e2f' : undefined,
                        cursor: 'pointer',
                      }}
                      onClick={() => setSelectedFileId(file.id)}
                    >
                      <td>
                        <strong>{file.name}</strong>
                      </td>
                      <td>
                        <span className="tag-badge">{file.language}</span>
                      </td>
                      <td>{file.size} B</td>
                      <td>
                        {corpusFiles.length > 1 && (
                          <button
                            className="text-button"
                            style={{ color: '#ef9ea7' }}
                            onClick={(e) => {
                              e.stopPropagation();
                              syntraEngine.removeCorpusFile(file.id);
                            }}
                            title="Delete file from corpus"
                          >
                            <Trash2 size={13} />
                          </button>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>

              {selectedCorpusFile && (
                <div style={{ marginTop: 16 }}>
                  <span className="field-label">File Preview: {selectedCorpusFile.name}</span>
                  <pre
                    style={{
                      background: '#090d14',
                      padding: 12,
                      borderRadius: 6,
                      border: '1px solid #1c2738',
                      font: '11px/1.6 var(--mono)',
                      color: '#a8c6ec',
                      maxHeight: 190,
                      overflowY: 'auto',
                    }}
                  >
                    {selectedCorpusFile.content}
                  </pre>
                </div>
              )}

              <div style={{ marginTop: 14 }}>
                <button
                  className="secondary-button"
                  disabled={dataBusy || isTraining}
                  onClick={() => prepareData()}
                >
                  {dataBusy ? <LoaderCircle size={14} className="spin" /> : <Database size={14} />}
                  Encode Corpus into data/processed/tokens.bin
                </button>
              </div>
            </div>

            {/* Training Control & Metrics */}
            <div className="panel">
              <div className="panel-heading">
                <div className="panel-title">
                  <Zap size={17} /> Training Loop & Evaluation Metrics
                </div>
                <span className="panel-kicker">ADAMW + COSINE LR</span>
              </div>

              <div className="training-state">
                <span className={isTraining ? 'status-indicator online' : 'status-indicator'} />
                STATUS: {training.status.toUpperCase()}
              </div>
              <div className="progress-track" style={{ height: 8 }}>
                <div className="progress-fill" style={{ width: trainingProgress + '%' }} />
              </div>

              <div className="training-metrics" style={{ marginTop: 14 }}>
                <span>
                  Step
                  <strong>
                    {training.step} / {config.max_steps} ({trainingProgress}%)
                  </strong>
                </span>
                <span>
                  Train Loss
                  <strong>{training.train_loss?.toFixed(4) ?? '—'}</strong>
                </span>
                <span>
                  Val Loss
                  <strong>{training.validation_loss?.toFixed(4) ?? '—'}</strong>
                </span>
              </div>

              <div className="action-row" style={{ marginTop: 16 }}>
                {isTraining ? (
                  <button className="secondary-button" style={{ marginTop: 0 }} onClick={pauseTraining}>
                    <Pause size={14} /> Pause Training
                  </button>
                ) : (
                  <button
                    className="primary-button"
                    style={{ width: '100%' }}
                    onClick={() => void startTraining()}
                  >
                    <Play size={14} />{' '}
                    {training.step > 0 && training.step < training.max_steps
                      ? 'Resume Training'
                      : 'Start Training from Scratch'}
                  </button>
                )}
              </div>

              <div style={{ marginTop: 22 }}>
                <div
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    marginBottom: 8,
                  }}
                >
                  <span className="field-label">Training Loss History (CSV metrics)</span>
                  {metrics.length > 0 && (
                    <button
                      className="text-button"
                      style={{ display: 'flex', alignItems: 'center', gap: 4 }}
                      onClick={exportMetricsCsv}
                    >
                      <Download size={12} /> Export CSV
                    </button>
                  )}
                </div>

                <div
                  style={{
                    maxHeight: 220,
                    overflowY: 'auto',
                    border: '1px solid #1f2a3a',
                    borderRadius: 6,
                  }}
                >
                  <table className="data-table" style={{ margin: 0 }}>
                    <thead>
                      <tr>
                        <th>STEP</th>
                        <th>TRAIN LOSS</th>
                        <th>VAL LOSS</th>
                        <th>LEARNING RATE</th>
                      </tr>
                    </thead>
                    <tbody>
                      {metrics.length === 0 ? (
                        <tr>
                          <td colSpan={4} style={{ textAlign: 'center', color: '#68778f' }}>
                            No evaluation rows yet. Start training to populate metrics.
                          </td>
                        </tr>
                      ) : (
                        metrics.map((row) => (
                          <tr key={row.step}>
                            <td>{row.step}</td>
                            <td>{row.train_loss.toFixed(4)}</td>
                            <td>{row.validation_loss.toFixed(4)}</td>
                            <td>{row.learning_rate.toExponential(2)}</td>
                          </tr>
                        ))
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 4: CLI TERMINAL SIMULATOR */}
        {activeTab === 'cli' && (
          <div className="panel" style={{ width: '100%' }}>
            <div className="panel-heading">
              <div className="panel-title">
                <TerminalSquare size={17} /> SYNTRA CLI Terminal Simulator
              </div>
              <span className="panel-kicker">LOCAL REPL</span>
            </div>

            <p className="hint-text" style={{ marginBottom: 14 }}>
              Execute native SYNTRA CLI commands interactively. Click quick commands or type below.
            </p>

            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 14 }}>
              <button
                className="secondary-button"
                style={{ width: 'auto', marginTop: 0, padding: '5px 10px', fontSize: 10 }}
                onClick={() => executeCliCommand('syntra --help')}
              >
                syntra --help
              </button>
              <button
                className="secondary-button"
                style={{ width: 'auto', marginTop: 0, padding: '5px 10px', fontSize: 10 }}
                onClick={() => executeCliCommand('syntra prepare')}
              >
                syntra prepare
              </button>
              <button
                className="secondary-button"
                style={{ width: 'auto', marginTop: 0, padding: '5px 10px', fontSize: 10 }}
                onClick={() => executeCliCommand('syntra train')}
              >
                syntra train
              </button>
              <button
                className="secondary-button"
                style={{ width: 'auto', marginTop: 0, padding: '5px 10px', fontSize: 10 }}
                onClick={() =>
                  executeCliCommand('syntra generate --prompt "class Stack:" --max-new-tokens 60')
                }
              >
                syntra generate
              </button>
              <button
                className="secondary-button"
                style={{ width: 'auto', marginTop: 0, padding: '5px 10px', fontSize: 10 }}
                onClick={() => executeCliCommand('syntra evaluate')}
              >
                syntra evaluate
              </button>
              <button
                className="secondary-button"
                style={{ width: 'auto', marginTop: 0, padding: '5px 10px', fontSize: 10 }}
                onClick={() => executeCliCommand('clear')}
              >
                clear
              </button>
            </div>

            <div className="terminal-window">
              {terminalHistory.map((item, idx) => (
                <div key={idx} className={`terminal-line ${item.type}`}>
                  {item.text}
                </div>
              ))}
              <div ref={terminalEndRef} />
            </div>

            <form onSubmit={handleCliSubmit} className="terminal-input-row">
              <span className="terminal-prompt">syntra&gt;</span>
              <input
                type="text"
                className="terminal-input"
                placeholder="type a command, e.g. syntra prepare, syntra train, syntra generate..."
                value={cliInput}
                onChange={(e) => setCliInput(e.target.value)}
                autoFocus
              />
            </form>
          </div>
        )}

        <footer className="page-footer">
          <span>© 2026 SYNTRA · AN EXPERIMENT IN INDEPENDENT CODING INTELLIGENCE</span>
          <span>
            <span className="live-dot" /> LOCAL-FIRST BY DESIGN
          </span>
        </footer>
      </main>
    </div>
  );
}

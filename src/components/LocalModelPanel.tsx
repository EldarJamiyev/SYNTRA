import { useEffect, useState } from 'react';
import { BrainCircuit, CheckCircle2, Cpu, LoaderCircle, RefreshCw, Sparkles, TriangleAlert } from 'lucide-react';
import './LocalModelPanel.css';

type TrainingStatus = {
  status: string;
  step: number;
  max_steps: number;
  train_loss: number | null;
  validation_loss: number | null;
  message: string;
  elapsed_seconds?: number;
};
type ModelStatus = {
  service: string;
  backend: string;
  checkpoint: string;
  checkpoint_exists: boolean;
  trained_steps: number;
  parameter_count: number;
  config: { seq_len: number; n_layer: number; n_head: number; n_embd: number } | null;
  train_loss: number | null;
  validation_loss: number | null;
  training?: TrainingStatus;
  pretrained_weights: boolean;
  external_ai_api: boolean;
  note: string;
};
type GeneratedOutput = {
  text: string;
  sourceContent: string;
  trained_steps: number;
  parameter_count: number;
  device: string;
  sourceHash: string;
  generationMode: string;
  notice: string;
};

async function request<T>(route: string, body?: unknown): Promise<T> {
  const response = await fetch(route, {
    method: body === undefined ? 'GET' : 'POST',
    headers: body === undefined ? {} : { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const value = await response.json() as T & { error?: string };
  if (!response.ok) throw new Error(value.error || 'Local model request failed.');
  return value;
}

type Props = {
  root: string;
  relativePath: string;
  onGenerated: (text: string, message: string, sourceContent: string, sourceHash: string) => void;
  onError: (message: string) => void;
};

export default function LocalModelPanel({ root, relativePath, onGenerated, onError }: Props) {
  const [model, setModel] = useState<ModelStatus | null>(null);
  const [modelError, setModelError] = useState('');
  const [task, setTask] = useState('Implement the requested change while preserving existing behavior and keeping the file valid.');
  const [steps, setSteps] = useState(1000);
  const [instructionDataset, setInstructionDataset] = useState('');
  const [busy, setBusy] = useState('');
  const [lastOutput, setLastOutput] = useState<GeneratedOutput | null>(null);

  const refresh = async () => {
    try {
      const value = await request<ModelStatus>('/local-api/ai/status');
      setModel(value);
      setModelError('');
    } catch (cause) {
      setModelError(cause instanceof Error ? cause.message : String(cause));
    }
  };

  useEffect(() => { void refresh(); }, []);
  useEffect(() => {
    if (model?.training?.status !== 'running' && model?.training?.status !== 'starting') return;
    const timer = window.setInterval(() => { void refresh(); }, 2500);
    return () => window.clearInterval(timer);
  }, [model?.training?.status]);

  const startTraining = async () => {
    if (!root.trim()) { onError('Enter and inspect a project path before training the local model.'); return; }
    setBusy('train');
    try {
      await request('/local-api/ai/train', {
        root: root.trim(),
        steps: Math.max(1, Math.min(100000, Math.round(steps))),
        batch_size: 4,
        max_tokens: 2000000,
        device: 'auto',
      });
      await refresh();
    } catch (cause) {
      onError(cause instanceof Error ? cause.message : String(cause));
    } finally { setBusy(''); }
  };

  const generate = async () => {
    if (!model?.checkpoint_exists) {
      onError('No trained neural checkpoint exists yet. Train the local model first.');
      return;
    }
    if (!root.trim() || !relativePath.trim() || !task.trim()) {
      onError('Enter a project path, target file path, and a specific change request.');
      return;
    }
    setBusy('generate');
    try {
      const value = await request<GeneratedOutput>('/local-api/agent/generate', {
        root: root.trim(),
        path: relativePath.trim(),
        goal: task.trim(),
        max_new_tokens: 384,
      });
      if (!value.text.trim()) throw new Error('The model emitted no usable text. Adjust training or the prompt, then try again.');
      setLastOutput(value);
      onGenerated(value.text, value.notice, value.sourceContent, value.sourceHash);
    } catch (cause) {
      onError(cause instanceof Error ? cause.message : String(cause));
    } finally { setBusy(''); }
  };

  const fineTune = async () => {
    if (!model?.checkpoint_exists) { onError('Train a local source-code checkpoint before instruction fine-tuning.'); return; }
    if (!instructionDataset.trim()) { onError('Enter an absolute path to a local prompt/completion JSONL dataset.'); return; }
    setBusy('finetune');
    try {
      await request('/local-api/ai/finetune', {
        dataset: instructionDataset.trim(),
        steps: Math.max(1, Math.min(100000, Math.round(steps))),
        batch_size: 4,
        device: 'auto',
      });
      await refresh();
    } catch (cause) {
      onError(cause instanceof Error ? cause.message : String(cause));
    } finally { setBusy(''); }
  };

  const training = model?.training;
  const isTraining = training?.status === 'running' || training?.status === 'starting';
  const statusLabel = model?.checkpoint_exists ? 'TRAINED CHECKPOINT FOUND' : 'NO TRAINED CHECKPOINT';

  return (
    <div className="builder-panel">
      <div className="builder-panel-title"><BrainCircuit size={18} /><div><h3>Native model and code completion</h3><p>Uses SYNTRA's own randomly initialized Transformer weights. PyTorch provides tensor math and gradient descent; no hosted model or pretrained checkpoint is loaded.</p></div></div>
      <div className="model-runtime-grid">
        <div className="model-runtime-stat"><span>MODEL SERVICE</span><strong>{model ? 'ONLINE' : 'OFFLINE'}</strong><small>{model?.backend || 'Python + PyTorch'}</small></div>
        <div className="model-runtime-stat"><span>CHECKPOINT</span><strong>{statusLabel}</strong><small>{model ? model.trained_steps.toLocaleString() + ' optimizer steps' : 'Not connected'}</small></div>
        <div className="model-runtime-stat"><span>PARAMETERS</span><strong>{model?.parameter_count ? model.parameter_count.toLocaleString() : '—'}</strong><small>{model?.config ? model.config.n_layer + ' layers · ' + model.config.n_head + ' heads · ' + model.config.seq_len + ' byte context' : 'Configure at training time'}</small></div>
        <div className="model-runtime-stat"><span>VALIDATION LOSS</span><strong>{model?.validation_loss == null ? '—' : model.validation_loss.toFixed(4)}</strong><small>{model?.train_loss == null ? 'No measured checkpoint yet' : 'Train loss ' + model.train_loss.toFixed(4)}</small></div>
      </div>
      {modelError && <div className="builder-message error"><TriangleAlert size={15} /> {modelError} Start the local Python model with <code>npm run ai:server</code>. Install dependencies with <code>npm run ai:install</code>, then restart the AI service.</div>}
      {training && <div className="model-training-status">
        <div><strong>{training.status.toUpperCase()}</strong><span>{training.step.toLocaleString()} / {training.max_steps.toLocaleString()} steps · {training.message}</span></div>
        <div className="model-progress"><span style={{ width: (training.max_steps ? Math.min(100, training.step / training.max_steps * 100) : 0) + '%' }} /></div>
      </div>}
      <div className="model-action-row">
        <label>Training steps<input type="number" min={1} max={100000} step={100} value={steps} onChange={event => setSteps(Number(event.target.value))} /></label>
        <button className="builder-primary" disabled={Boolean(busy) || !root.trim() || isTraining} onClick={() => void startTraining()}>
          {busy === 'train' || isTraining ? <LoaderCircle size={15} className="spin" /> : <Cpu size={15} />} Train from local source
        </button>
        <button className="builder-secondary" disabled={Boolean(busy)} onClick={() => void refresh()}><RefreshCw size={15} /> Refresh status</button>
      </div>
      <div className="model-task-area model-finetune-area">
        <label>Instruction dataset (JSONL, local file path)</label>
        <input className="model-dataset-input" value={instructionDataset} onChange={event => setInstructionDataset(event.target.value)} spellCheck={false} placeholder={'C:\\Users\\You\\Documents\\syntra-instructions.jsonl'} />
        <p>Each line must be JSON with string fields named <code>prompt</code> and <code>completion</code>. Fine-tuning masks prompt loss and learns completion tokens. Begin with a trained checkpoint and examples you own or are licensed to use.</p>
        <button className="builder-secondary full-width" disabled={Boolean(busy) || isTraining || !model?.checkpoint_exists || !instructionDataset.trim()} onClick={() => void fineTune()}>
          {busy === 'finetune' ? <LoaderCircle size={15} className="spin" /> : <Sparkles size={15} />} Supervised instruction fine-tuning
        </button>
      </div>
      <div className="model-task-area">
        <label>Requested code change</label>
        <textarea className="builder-goal" value={task} onChange={event => setTask(event.target.value)} rows={3} />
        <button className="builder-secondary full-width" disabled={Boolean(busy) || isTraining || !model?.checkpoint_exists || !relativePath.trim()} onClick={() => void generate()}>
          {busy === 'generate' ? <LoaderCircle size={15} className="spin" /> : <Sparkles size={15} />} Generate from local checkpoint into editor
        </button>
      </div>
      {lastOutput && <div className="model-output-note"><CheckCircle2 size={15} /><span>Generated {lastOutput.text.length.toLocaleString()} characters from step {lastOutput.trained_steps.toLocaleString()} on {lastOutput.device}. Source hash: {lastOutput.sourceHash.slice(0, 16)}…</span></div>}
      <div className="model-caveat"><TriangleAlert size={15} /><span>A trained-from-scratch model is real inference, but it will not behave like a commercial coding model just because it has a checkpoint. Source-code pretraining learns next-byte patterns; reliable instruction-following requires substantially more data and a separate supervised instruction-tuning stage. Treat this output as an untrusted completion, inspect it, create a reviewed proposal, and run tests.</span></div>
    </div>
  );
}

import { useEffect, useState } from 'react';
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  ClipboardCheck,
  FileCode2,
  FolderSearch,
  GitCompare,
  LoaderCircle,
  Play,
  Search,
  ShieldCheck,
  Terminal,
} from 'lucide-react';
import './ProjectBuilder.css';

type WorkspaceFile = { path: string; size: number; language: string };
type WorkspaceInfo = {
  root: string;
  name: string;
  fileCount: number;
  truncated: boolean;
  languageCounts: Record<string, number>;
  packageManager: string;
  frameworks: string[];
  scripts: Record<string, string>;
  manifests: Record<string, boolean>;
  readmeExcerpt: string;
  files: WorkspaceFile[];
};
type SearchHit = { path: string; score: number; size: number; matches: Array<{ line: number; text: string }> };
type SearchResult = { root: string; query: string; searchedFiles: number; results: SearchHit[] };
type Plan = {
  id: string;
  planPath: string;
  goal: string;
  planningMode: string;
  notice: string;
  tasks: Array<{ id: string; phase: string; title: string; description: string; status: string; acceptance: string[] }>;
};
type Proposal = {
  id: string;
  path: string;
  existedBefore: boolean;
  beforeHash: string | null;
  afterHash: string;
  summary: { oldLines: number; newLines: number; addedLines: number; removedLines: number; preview: string };
};
type CheckResult = {
  task: string;
  command: string;
  passed: boolean;
  timedOut: boolean;
  durationMs: number;
  exitCode: number | null;
  output: string;
};
type Health = { status: string; service: string; host: string; port: number; capabilities: string[] };

async function postJson<T>(route: string, body: unknown): Promise<T> {
  const response = await fetch(route, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const payload = await response.json() as T & { error?: string };
  if (!response.ok) throw new Error(payload.error || 'Local workspace request failed with HTTP ' + response.status);
  return payload;
}

async function getJson<T>(route: string): Promise<T> {
  const response = await fetch(route);
  const payload = await response.json() as T & { error?: string };
  if (!response.ok) throw new Error(payload.error || 'Local workspace request failed with HTTP ' + response.status);
  return payload;
}

export default function ProjectBuilder() {
  const [root, setRoot] = useState(() => {
    try { return window.localStorage.getItem('syntra_project_root') || ''; } catch { return ''; }
  });
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState('');
  const [inspection, setInspection] = useState<WorkspaceInfo | null>(null);
  const [query, setQuery] = useState('ai brain');
  const [searchResult, setSearchResult] = useState<SearchResult | null>(null);
  const [goal, setGoal] = useState('Build a maintainable full-stack application with a polished UI, modular backend, automated tests, security checks, and complete documentation.');
  const [plan, setPlan] = useState<Plan | null>(null);
  const [relativePath, setRelativePath] = useState('src/example.ts');
  const [content, setContent] = useState('');
  const [proposal, setProposal] = useState<Proposal | null>(null);
  const [applyConsent, setApplyConsent] = useState(false);
  const [checkTask, setCheckTask] = useState('test');
  const [checkConsent, setCheckConsent] = useState(false);
  const [checkResult, setCheckResult] = useState<CheckResult | null>(null);
  const [busy, setBusy] = useState('');
  const [notice, setNotice] = useState('');
  const [error, setError] = useState('');

  useEffect(() => {
    let mounted = true;
    getJson<Health>('/local-api/health')
      .then(value => { if (mounted) { setHealth(value); setHealthError(''); } })
      .catch(cause => {
        if (mounted) setHealthError(cause instanceof Error ? cause.message : 'Local service unavailable.');
      });
    return () => { mounted = false; };
  }, []);

  const rememberRoot = () => {
    if (!root.trim()) throw new Error('Enter the absolute path to a project folder first.');
    try { window.localStorage.setItem('syntra_project_root', root.trim()); } catch {}
    return root.trim();
  };

  const inspect = async () => {
    setBusy('inspect'); setError(''); setNotice(''); setSearchResult(null);
    try {
      const value = await postJson<WorkspaceInfo>('/local-api/workspace/inspect', { root: rememberRoot() });
      setInspection(value);
      setNotice('Inspected ' + value.fileCount + ' source/configuration files in ' + value.name + '.');
    } catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)); }
    finally { setBusy(''); }
  };

  const search = async () => {
    setBusy('search'); setError(''); setNotice('');
    try {
      const value = await postJson<SearchResult>('/local-api/workspace/search', { root: rememberRoot(), query, limit: 20 });
      setSearchResult(value);
      setNotice('Searched ' + value.searchedFiles + ' readable files.');
    } catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)); }
    finally { setBusy(''); }
  };

  const createPlan = async () => {
    setBusy('plan'); setError(''); setNotice('');
    try {
      const value = await postJson<Plan>('/local-api/workspace/plan', { root: rememberRoot(), goal });
      setPlan(value);
      setNotice('Saved a project workflow plan to ' + value.planPath + '.');
    } catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)); }
    finally { setBusy(''); }
  };

  const loadFile = async () => {
    setBusy('read'); setError(''); setNotice('');
    try {
      const value = await postJson<{ content: string; path: string; sha256: string }>('/local-api/workspace/read', {
        root: rememberRoot(), path: relativePath,
      });
      setContent(value.content);
      setProposal(null);
      setApplyConsent(false);
      setNotice('Loaded ' + value.path + ' · SHA-256 ' + value.sha256.slice(0, 16) + '…');
    } catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)); }
    finally { setBusy(''); }
  };

  const createProposal = async () => {
    setBusy('propose'); setError(''); setNotice('');
    try {
      const value = await postJson<Proposal>('/local-api/workspace/propose', {
        root: rememberRoot(), path: relativePath, content,
      });
      setProposal(value);
      setApplyConsent(false);
      setNotice('Proposal saved. Review the target and change summary before applying.');
    } catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)); }
    finally { setBusy(''); }
  };

  const applyProposal = async () => {
    if (!proposal) return;
    if (!applyConsent) {
      setError('Tick the confirmation box after reviewing this proposed file change.');
      return;
    }
    setBusy('apply'); setError(''); setNotice('');
    try {
      const result = await postJson<{ applied: boolean; path: string; backupCreated: boolean }>('/local-api/workspace/apply', {
        root: rememberRoot(), proposalId: proposal.id, confirm: true,
      });
      setNotice('Applied ' + result.path + (result.backupCreated ? ' · previous version backed up.' : ' · new file created.'));
      setProposal(null);
      setApplyConsent(false);
      await inspect();
    } catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)); }
    finally { setBusy(''); }
  };

  const runCheck = async () => {
    if (!checkConsent) {
      setError('Confirm that you understand the selected project script will execute on this computer.');
      return;
    }
    setBusy('check'); setError(''); setNotice(''); setCheckResult(null);
    try {
      const value = await postJson<CheckResult>('/local-api/workspace/check', {
        root: rememberRoot(), task: checkTask, timeoutMs: 120000,
      });
      setCheckResult(value);
      setNotice(value.passed ? checkTask.toUpperCase() + ' passed in ' + (value.durationMs / 1000).toFixed(1) + ' seconds.' : checkTask.toUpperCase() + ' failed. Review the captured output.');
    } catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)); }
    finally { setBusy(''); }
  };

  const isBusy = Boolean(busy);
  const offline = !health;

  return (
    <section className="builder-page">
      <div className="builder-hero">
        <div>
          <div className="builder-eyebrow"><ShieldCheck size={14} /> LOCAL PROJECT OPERATIONS</div>
          <h2>Build applications, <span>not just snippets.</span></h2>
          <p>Inspect the whole repository, locate relevant code, plan vertical slices, propose file changes, then verify with the project's own build and test scripts.</p>
        </div>
        <div className={'builder-connection ' + (health ? 'connected' : 'disconnected')}>
          <span className="builder-connection-dot" />
          <div>
            <strong>{health ? 'LOCAL TOOLS ONLINE' : 'LOCAL TOOLS OFFLINE'}</strong>
            <small>{health ? health.host + ':' + health.port : 'Start npm run server'}</small>
          </div>
        </div>
      </div>

      <div className="builder-callout">
        <AlertTriangle size={18} />
        <div>
          <strong>Current model limitation</strong>
          <p>The browser model is a byte-transition baseline, not a trained neural coding model. The workspace and verification tools are real; autonomous model-driven implementation is a future milestone. SYNTRA does not call external AI APIs.</p>
        </div>
      </div>

      {healthError && <div className="builder-message error"><AlertTriangle size={16} /> {healthError}. Run <code>npm run server</code> in a separate terminal, then refresh this page.</div>}
      {error && <div className="builder-message error"><AlertTriangle size={16} /> {error}</div>}
      {notice && <div className="builder-message success"><CheckCircle2 size={16} /> {notice}</div>}

      <div className="builder-panel workspace-panel">
        <div className="builder-panel-title"><FolderSearch size={18} /><div><h3>Project workspace</h3><p>Choose an absolute local path. SYNTRA will not scan outside this directory.</p></div></div>
        <div className="builder-path-row">
          <input value={root} onChange={event => setRoot(event.target.value)} placeholder={'C:\\Users\\You\\Documents\\YourProject'} spellCheck={false} />
          <button className="builder-primary" disabled={isBusy || offline} onClick={() => void inspect()}>
            {busy === 'inspect' ? <LoaderCircle size={15} className="spin" /> : <FolderSearch size={15} />}
            Inspect project
          </button>
        </div>
      </div>

      {inspection && (
        <div className="builder-overview-grid">
          <div className="builder-stat"><span>PROJECT</span><strong>{inspection.name}</strong><small>{inspection.packageManager}</small></div>
          <div className="builder-stat"><span>READABLE FILES</span><strong>{inspection.fileCount.toLocaleString()}</strong><small>{inspection.truncated ? 'Listing capped at 5,000 files' : 'Source/config files indexed'}</small></div>
          <div className="builder-stat"><span>FRAMEWORKS</span><strong>{inspection.frameworks.join(' · ') || 'Not detected'}</strong><small>Detected from manifests</small></div>
          <div className="builder-stat"><span>AVAILABLE CHECKS</span><strong>{Object.keys(inspection.scripts).join(' · ') || 'No npm scripts'}</strong><small>Only explicit checks are run</small></div>
        </div>
      )}

      <div className="builder-columns">
        <div className="builder-panel">
          <div className="builder-panel-title"><Search size={18} /><div><h3>Repository search</h3><p>Find files and line-level matches across the codebase.</p></div></div>
          <div className="builder-form-row">
            <input value={query} onChange={event => setQuery(event.target.value)} onKeyDown={event => { if (event.key === 'Enter') void search(); }} placeholder="e.g. ai brain, /api/rca, incident correlator" />
            <button className="builder-secondary" disabled={isBusy || offline || !query.trim()} onClick={() => void search()}>{busy === 'search' ? <LoaderCircle size={15} className="spin" /> : <Search size={15} />} Search</button>
          </div>
          {searchResult && <div className="builder-result-list">
            {!searchResult.results.length && <p className="builder-empty">No matches found. Try a symbol name, route, module, or feature term.</p>}
            {searchResult.results.map(item => (
              <div className="builder-search-hit" key={item.path}>
                <div className="builder-hit-head"><button className="builder-file-link" onClick={() => { setRelativePath(item.path); void postJson<{content:string}>('/local-api/workspace/read',{root, path:item.path}).then(value=>setContent(value.content)).catch(cause=>setError(cause instanceof Error ? cause.message : String(cause))); }}>{item.path}</button><span>score {item.score}</span></div>
                {item.matches.map(match => <div className="builder-match" key={item.path + ':' + match.line}><code>{match.line}</code><pre>{match.text}</pre></div>)}
              </div>
            ))}
          </div>}
        </div>

        <div className="builder-panel">
          <div className="builder-panel-title"><ClipboardCheck size={18} /><div><h3>Implementation plan</h3><p>Create a persistent, reviewable development workflow in the project.</p></div></div>
          <textarea className="builder-goal" value={goal} onChange={event => setGoal(event.target.value)} rows={4} />
          <button className="builder-secondary full-width" disabled={isBusy || offline || goal.trim().length < 8} onClick={() => void createPlan()}>
            {busy === 'plan' ? <LoaderCircle size={15} className="spin" /> : <ClipboardCheck size={15} />} Create project plan
          </button>
          {plan && <div className="builder-plan-list">
            <div className="builder-plan-notice">{plan.notice}</div>
            {plan.tasks.map(task => <div className="builder-task" key={task.id}>
              <div className="builder-task-index">{task.id}</div>
              <div><strong>{task.title}</strong><p>{task.description}</p><small>{task.acceptance.join(' · ')}</small></div>
            </div>)}
          </div>}
        </div>
      </div>

      <div className="builder-panel">
        <div className="builder-panel-title"><GitCompare size={18} /><div><h3>Reviewable file changes</h3><p>Load an existing file or propose a new file. Changes are never applied automatically.</p></div></div>
        <div className="builder-form-row">
          <input value={relativePath} onChange={event => setRelativePath(event.target.value)} placeholder="relative/path/to/file.ts" spellCheck={false} />
          <button className="builder-secondary" disabled={isBusy || offline || !relativePath.trim()} onClick={() => void loadFile()}>
            {busy === 'read' ? <LoaderCircle size={15} className="spin" /> : <FileCode2 size={15} />} Load file
          </button>
        </div>
        <textarea className="builder-code-editor" value={content} onChange={event => setContent(event.target.value)} spellCheck={false} placeholder="Enter complete proposed file contents here. The AI-driven patch generator will connect here in a later milestone." />
        <div className="builder-action-row">
          <span>Maximum text proposal: 512 KB. The target file hash is rechecked before apply.</span>
          <button className="builder-primary" disabled={isBusy || offline || !relativePath.trim()} onClick={() => void createProposal()}>
            {busy === 'propose' ? <LoaderCircle size={15} className="spin" /> : <GitCompare size={15} />} Create proposal
          </button>
        </div>
        {proposal && <div className="builder-proposal">
          <div className="builder-proposal-top">
            <div><strong>Proposal ready: {proposal.path}</strong><p>{proposal.existedBefore ? 'Existing file update' : 'New file'} · +{proposal.summary.addedLines} / −{proposal.summary.removedLines} lines</p></div>
            <span className="builder-proposal-id">{proposal.id.slice(0, 8)}</span>
          </div>
          <div className="builder-hash-row"><span>Base SHA</span><code>{proposal.beforeHash || 'NEW FILE'}</code></div>
          <div className="builder-hash-row"><span>Proposed SHA</span><code>{proposal.afterHash}</code></div>
          <pre className="builder-preview">{proposal.summary.preview || '(empty file)'}</pre>
          <label className="builder-consent"><input type="checkbox" checked={applyConsent} onChange={event => setApplyConsent(event.target.checked)} /> I reviewed the target path and proposal summary and authorize this file change.</label>
          <button className="builder-primary" disabled={isBusy || !applyConsent} onClick={() => void applyProposal()}>
            {busy === 'apply' ? <LoaderCircle size={15} className="spin" /> : <CheckCircle2 size={15} />} Apply confirmed proposal
          </button>
        </div>}
      </div>

      <div className="builder-panel">
        <div className="builder-panel-title"><Terminal size={18} /><div><h3>Build and test verification</h3><p>Run an allowlisted project check. The selected project script may execute code on this computer.</p></div></div>
        <div className="builder-check-row">
          <select value={checkTask} onChange={event => setCheckTask(event.target.value)}>
            <option value="build">Build</option><option value="test">Test</option><option value="lint">Lint</option><option value="typecheck">Type check</option>
          </select>
          <label className="builder-consent"><input type="checkbox" checked={checkConsent} onChange={event => setCheckConsent(event.target.checked)} /> I trust this project's scripts and authorize this check.</label>
          <button className="builder-primary" disabled={isBusy || offline || !checkConsent} onClick={() => void runCheck()}>
            {busy === 'check' ? <LoaderCircle size={15} className="spin" /> : <Play size={15} />} Run check
          </button>
        </div>
        {checkResult && <div className="builder-check-result">
          <div className="builder-check-result-head">
            <span className={checkResult.passed ? 'check-pill passed' : 'check-pill failed'}>{checkResult.passed ? 'PASSED' : checkResult.timedOut ? 'TIMED OUT' : 'FAILED'}</span>
            <strong>{checkResult.command}</strong>
            <small>{(checkResult.durationMs / 1000).toFixed(1)}s · exit {checkResult.exitCode ?? 'unknown'}</small>
          </div>
          <pre>{checkResult.output || '(No command output captured.)'}</pre>
        </div>}
      </div>

      <div className="builder-footnote"><Activity size={15} /><span>Project operations are local. File reads are bounded, secret-like files are blocked, file updates require explicit approval, and applied edits create backups. The current model itself is not yet capable of autonomously driving this workflow.</span><ArrowRight size={15} /></div>
    </section>
  );
}

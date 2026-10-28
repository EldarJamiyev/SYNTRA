import { ArrowUpRight, BrainCircuit, FolderSearch, Github } from 'lucide-react';
import ProjectBuilder from './components/ProjectBuilder';

export default function App() {
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark"><BrainCircuit size={23} strokeWidth={1.7} /></div>
          <div>
            <div className="brand-name">SYNTRA<span>®</span></div>
            <div className="brand-subtitle">LOCAL AI WORKSPACE</div>
          </div>
        </div>

        <div className="workspace-label">WORKSPACE</div>
        <div className="nav-item active" aria-current="page">
          <FolderSearch size={17} />
          Project Builder
          <span className="nav-dot" />
        </div>

        <div className="sidebar-bottom">
          <div className="local-card">
            <div className="local-title"><span className="live-dot" /> LOCAL-FIRST</div>
            <p>Randomly initialized local model weights, local training, local checkpoints, and no hosted AI inference.</p>
          </div>
          <a
            className="github-link"
            href="https://github.com/EldarJamiyev/SYNTRA"
            target="_blank"
            rel="noreferrer"
          >
            <Github size={16} /> Repository <ArrowUpRight size={14} />
          </a>
          <div className="sidebar-version">SYNTRA WORKSPACE <span>v0.2.0</span></div>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div className="breadcrumb">SYNTRA <span>/</span> <strong>Project Builder</strong></div>
          <div className="topbar-status">
            <span className="status-indicator" />
            SERVICE CONNECTION · CHECK IN WORKSPACE
          </div>
        </header>

        <section className="page-heading">
          <div className="eyebrow"><span className="eyebrow-line" /> NATIVE MODEL + SOFTWARE WORKSPACE</div>
          <h1>Build. Test. <span>Iterate.</span></h1>
          <p>
            Work against a real local codebase. Train your own small neural model, generate continuations from its checkpoint,
            review file changes, preserve backups, and run the actual project checks.
          </p>
        </section>

        <ProjectBuilder />

        <footer className="page-footer">
          <span>© 2026 SYNTRA · LOCAL-FIRST SOFTWARE WORKSPACE</span>
          <span><span className="live-dot" /> NO HOSTED MODEL INFERENCE</span>
        </footer>
      </main>
    </div>
  );
}

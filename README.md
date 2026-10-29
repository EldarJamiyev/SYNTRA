# SYNTRA — Simple local AI chat

SYNTRA is a deliberately minimal chat interface backed by a local model and a repository-aware coding agent. It shows the conversation, a message field, and one send control—without a dashboard full of toggles.

## Windows launch

Requirements: Node.js 20+ and Git. Python 3.11+ is required only for the native model fallback. A compatible local model in Ollama is preferred automatically.

In your existing SYNTRA repository folder, run:

~~~powershell
git pull
~~~

Then double-click **Start-SYNTRA.bat**. Open http://127.0.0.1:3000 and keep the service windows open. On future runs, double-click the same file again.

The launcher checks the page signature, detects stale app processes on the chat port, and installs newly required frontend dependencies instead of blindly accepting an unrelated page.

## Chat and coding behavior

Normal requests such as writing an email, explaining a concept, or brainstorming stay in chat. Explicit software tasks—such as fixing an API endpoint, updating a component, or changing the project UI—are routed to the repository agent.

Assistant replies support Markdown, GitHub-flavored tables and lists, inline code, fenced code blocks, and safe links. Local Ollama replies stream into the conversation while the model generates; the native research model displays its response when generation completes. User prompts are rendered as literal text so HTML-like snippets are not silently removed. Raw assistant HTML is not rendered, and external links open with appropriate isolation.

For coding tasks, SYNTRA maps the project files (including recognised CI/configuration folders such as .github and .devcontainer) and imports, asks the selected local model for a structured patch, validates exact source anchors and paths, and runs available project checks in a temporary copy. Changes are auto-applied only when discovered checks pass and the patch does not modify tests or verification configuration. Patches that rewrite package-manager scripts are blocked from running those scripts; test/CI/config changes require review and explicit /apply. The agent keeps backups and audit records under .syntra. Multi-file changes are preflighted and applied as a transaction with a shared backup directory; if a later write fails, already-written files are rolled back when safe to do so.

## Local model

SYNTRA prefers an installed local Ollama model and does not call a hosted AI API. Its model selection excludes cloud-only models and rejects weights larger than 70% of physical RAM, leaving room for the OS and inference context/cache. That is a conservative file-size filter, not a guarantee that every remaining model will fit in practice. Model discovery is briefly cached between the status, token-count, and generation stages to avoid repeating the same probe. Inference speed and answer quality still depend on your hardware and the installed model.

If Ollama is unavailable, SYNTRA can use its own local PyTorch runtime. The native model starts with randomly initialized weights: training it from scratch on a small local corpus does not make it comparable to Claude or other frontier models.

## Important boundary

This app is designed to run locally at http://127.0.0.1:3000; the Node service binds to loopback and is not a hosted AI endpoint. A static GitHub Pages copy cannot execute your local model or workspace agent. Use the local launcher for real chat and code changes.

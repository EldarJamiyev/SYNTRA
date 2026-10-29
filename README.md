# SYNTRA — Simple local AI chat

SYNTRA provides a minimal chat interface backed by a local model and a repository-aware coding agent. The frontend intentionally contains only a small title, the conversation, a message field, and a send control.

## Windows launch

Requirements: Node.js 20+ and Git. Python 3.11+ is required for the native model fallback. A compatible locally installed model in Ollama is preferred automatically.

In PowerShell, update the existing repository once:

~~~powershell
git pull
~~~

Then double-click **Start-SYNTRA.bat**. It opens the chat at http://127.0.0.1:3000. Keep the service windows open. On future runs, double-click the same file again.

If an old app is already occupying port 3000, the launcher checks for the expected SYNTRA Chat page signature and reports the conflict instead of treating any random page as SYNTRA.

## Local model

The runtime prefers a suitable pretrained coding model installed in local Ollama. Prompts are sent only to the Ollama service at 127.0.0.1:11434, or to SYNTRA's own local PyTorch runtime as fallback. No hosted model API is used. You can install a local model using Ollama, for example `ollama pull qwen3-coder:30b`; this requires local storage and enough memory, and inference speed depends on hardware.

SYNTRA's coding agent understands the project tree, proposes structured file changes, validates paths and exact source anchors, and runs available checks in a temporary copy before applying verified changes.

## Limitation

The native PyTorch Transformer starts without pretrained weights. Training it from scratch on this repository does not give it frontier-model capability. SYNTRA's quality depends on the local model selected and the checks available for each project.

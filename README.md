# SYNTRA — Local Coding AI and Application Builder

SYNTRA is intended to become a local-first coding AI that can develop complete applications, including projects with the scale and layered architecture of EJI/EJII: source code, UI, APIs, tools, tests, documentation, and deployment workflows.

It must not depend on a pretrained model, hosted inference, Ollama, Anthropic/OpenAI APIs, or any other external AI inference service. The intended model starts from random initialization, and all SYNTRA-owned training, inference, app-building tools, workspace state, and UI live in this repository.

## Important current-state disclosure

The model in the current browser prototype is a byte-transition baseline (unigram/bigram/trigram counts), not a neural transformer. Its measured cross-entropy is now calculated from its stored transitions; no gradient descent or transformer-weight training occurs in this browser version. This baseline is useful for tokenizer and UI experiments, but it is **not capable of independently building EJI-scale applications**.

The repository now adds a real local project-workspace layer: inspect a project, search source files, read bounded text files, create a staged plan, propose file changes, apply only after explicit confirmation, back up prior file contents, and run allowlisted project checks. The model-driven agent that would autonomously coordinate these tools is not yet implemented.

## Application-building workflow

The intended loop is:

1. Discover project manifests, entry points, dependencies, architecture, tests, and conventions.
2. Search and read relevant parts of a large repository without sending the entire codebase into a single context window.
3. Convert the requested feature into testable requirements and a dependency-aware plan.
4. Work in small vertical slices, proposing reviewable changes instead of blindly overwriting files.
5. Apply approved changes only if the target file has not changed since proposal creation; create a backup and audit record.
6. Run the selected build/test/lint/typecheck script, inspect real output, fix failures, and repeat verification.
7. Preserve project-specific decisions, rejected options, test outcomes, and acceptance evidence.

## What is included now

- Browser interface for the initial tokenizer / byte-transition research prototype.
- Local Node.js project workspace API, with no third-party server framework.
- Project inspection and source search that skip common dependency/build folders and likely secret files.
- Bounded text reads with SHA-256 content hashes.
- Persistent deterministic workflow plans under the selected project's .syntra directory.
- File-change proposals, explicit apply confirmation, stale-content protection, backups, and an action audit log.
- Allowlisted build, test, lint, and typecheck checks with a timeout and bounded captured output.
- CLI sharing the same workspace implementation.
- Built-in Node.js tests and a frontend build workflow.

## Requirements

- Node.js 20+ and npm.
- For frontend development, install the package dependencies with npm install.
- The local workspace API listens on 127.0.0.1:3001 and the Vite UI listens on 127.0.0.1:3000. They are loopback-only by default.

## Run SYNTRA

Open PowerShell in the repository root and install frontend dependencies:

~~~powershell
npm install
~~~

Start the local project-workspace service in the first terminal:

~~~powershell
npm run server
~~~

Start the UI in a second terminal:

~~~powershell
npm run dev
~~~

Open http://127.0.0.1:3000 and choose Project Builder. Enter an absolute path to a project you own or are authorized to modify, for example your local EJI checkout.

## CLI

~~~powershell
node bin/syntra.mjs help
node bin/syntra.mjs inspect --root "C:\Users\Eldar\Documents\EJI"
node bin/syntra.mjs search --root "C:\Users\Eldar\Documents\EJI" --query "ai brain"
node bin/syntra.mjs read --root "C:\Users\Eldar\Documents\EJI" --file "src\core\ai-brain.js"
node bin/syntra.mjs plan --root "C:\Users\Eldar\Documents\EJI" --goal "Improve evidence-backed incident correlation and test the changes."
node bin/syntra.mjs check --root "C:\Users\Eldar\Documents\EJI" --task test
~~~

To propose a file change, place the proposed contents into a separate text file, run the propose command, review the returned proposal ID and summary, then apply it with the explicit --confirm flag. Do not apply unreviewed changes.

## Safety boundaries

- The local API binds to loopback, not a public or LAN interface.
- Workspace roots must be absolute paths, and paths are constrained to the selected root.
- Sensitive credential files and symbolic links are blocked; dependency and build folders are skipped during indexing.
- Proposals do not alter project files until explicit confirmation.
- Applying a proposal checks the target SHA-256 hash and backs up the previous file before writing.
- Build/test checks execute scripts from the selected project, so only run them after you trust the project and its dependencies.
- SYNTRA does not automatically execute generated code.

## Development checks

~~~powershell
npm test
npm run build
~~~

See docs/APP_BUILDER_ARCHITECTURE.md for the target architecture and the distinction between implemented tools and future model-driven behavior.

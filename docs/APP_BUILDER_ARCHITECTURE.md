# Chat and Coding Agent

The user-facing interface is deliberately reduced to the conversation and one message composer. It does not expose training charts, model statistics, project dashboards or editor controls.

The Node service accepts chat messages and maintains an in-memory turn status so the UI can display a minimal “thinking” indication while work runs. A workspace switch can be entered in chat as `/workspace C:\\path\\to\\project`.

For coding requests, the local agent creates a repository import graph, ranks related files, dynamically limits retrieved context using the active BPE tokenizer, asks for structured JSON find/replace patches, validates exact anchors and safe paths, and runs configured checks in a temporary project copy. Failed checks are returned to the next bounded repair attempt. Passing checks permit hash-checked changes to be applied with backups; unverified proposals require the explicit `/apply` chat command.

The neural model is the source of generated text. It starts from random weights and must be trained locally; the agent architecture reduces operational failures but cannot substitute for learned coding knowledge. No quality claim is made from a successful infrastructure test alone.

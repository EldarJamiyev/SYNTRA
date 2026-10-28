# SYNTRA architecture

## Scope

SYNTRA is a coding-model experiment trained from random initialization. The repository defines the tokenizer, model architecture, training loop, checkpoint format, autoregressive sampling logic, local API, CLI, and user interface. It uses PyTorch for tensor arithmetic, optimized operations, and automatic differentiation. It does not use a pretrained model or download model weights.

This does not claim every numerical primitive is independently reimplemented. A custom tensor runtime, automatic differentiation engine, memory allocator, and GPU kernel stack would be a separate engineering and performance project.

## Data flow

1. Local text/code files are placed in data/raw.
2. Corpus preparation filters supported text files, limits file and total sizes, and produces a 16-bit token stream at data/processed/tokens.bin.
3. The byte tokenizer maps four reserved tokens plus 256 byte values to 260 token IDs.
4. SyntraModel maps tokens and positions to embeddings, runs causal self-attention and feed-forward blocks, then predicts the next token.
5. Training optimizes next-token cross entropy using AdamW, gradient clipping, a cosine learning-rate schedule, validation batches, and checkpoints.
6. Generation repeatedly predicts and samples the next token from the latest context.
7. The local API exposes status and generation endpoints to SYNTRA Studio.

## Important limitations

- A byte tokenizer is simple and lossless but inefficient compared with a learned subword tokenizer. It uses many tokens for ordinary source code.
- A randomly initialized model has no prior coding knowledge. It must learn patterns from suitable data.
- The starter architecture has a short context window and no repository index, retrieval, agent loop, or automated file-editing ability yet.
- Loss and perplexity measure next-token prediction, not whether generated code is correct, secure, or useful. Task-level tests are required.
- CPU training is expected to be slow. Increasing virtual memory cannot substitute for compute throughput.

## Security

The server defaults to 127.0.0.1. Keep it on loopback unless network exposure is deliberately configured and secured. Generated code must be treated as untrusted text. This version does not execute model-generated code. Do not add automatic shell execution without isolation and explicit approval controls.

# SYNTRA Studio

SYNTRA is an experimental coding-focused language model studio. This project contains the complete decoder-only causal transformer model architecture, lossless UTF-8 byte tokenizer, training loop, checkpoint handling, inference/generation playground, CLI emulator, and evaluation suite rebuilt as a standalone React web application.

**No pretrained model, pretrained weights, hosted inference, or external AI API is used.** The tokenizer, transformer architecture, training procedure, checkpoint persistence, and token sampling loop run 100% locally.

## Features

- **Decoder-Only Causal Transformer**: Learned token and position embeddings, multi-head attention with causal masking, GELU feedforward layers, and weight-tied LM head.
- **Byte Tokenizer**: 260-token UTF-8 byte tokenizer with four reserved tokens (`<PAD>`, `<BOS>`, `<EOS>`, `<UNK>`) and byte offset 4.
- **In-Browser Training Engine**: Training from random initialization, real-time training and validation loss tracking, cosine learning-rate decay, and resumable checkpoints stored locally.
- **Code Generation Playground**: Autoregressive sampling with configurable `max_new_tokens`, `temperature`, and `top_k`, code templates, and instant clipboard copy.
- **Corpus Manager**: Prepare, inspect, and manage raw source code files with UTF-8 token stream serialization.
- **Architecture & Tokenizer Sandbox**: Live byte breakdown of code text and detailed layer parameter breakdown (~7.2M parameters).
- **Interactive CLI Terminal**: Run `syntra prepare`, `syntra train`, `syntra generate`, `syntra evaluate`, and `syntra --help` in an interactive shell.

## Running Locally

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

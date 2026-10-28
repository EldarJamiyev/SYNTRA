"""Real command-line interface for data preparation, training, inference and serving."""
import argparse
import json
import os
from pathlib import Path
import sys

from .config import ModelConfig
from .data import load_corpus
from .runtime import evaluate, generate, status, tokenize_text
from .instruction import finetune
from .training import train


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="syntra", description="SYNTRA local neural coding-model tools.")
    sub = parser.add_subparsers(dest="command", required=True)

    data = sub.add_parser("inspect-data", help="Index a local source corpus without reading likely secrets.")
    data.add_argument("--root", required=True)
    data.add_argument("--max-tokens", type=int, default=5_000_000)

    training = sub.add_parser("train", help="Train a decoder-only Transformer from random initialization.")
    training.add_argument("--root", required=True, help="Absolute local source-code corpus path.")
    training.add_argument("--steps", type=int, default=1000)
    training.add_argument("--batch-size", type=int, default=4)
    training.add_argument("--seq-len", type=int, default=4096)
    training.add_argument("--vocab-size", type=int, default=1024)
    training.add_argument("--layers", type=int, default=4)
    training.add_argument("--heads", type=int, default=4)
    training.add_argument("--width", type=int, default=128)
    training.add_argument("--learning-rate", type=float, default=3e-4)
    training.add_argument("--max-tokens", type=int, default=2_000_000)
    training.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    training.add_argument("--seed", type=int, default=1337)
    training.add_argument("--resume", action="store_true")

    tokenizer_parser = sub.add_parser("tokenize", help="Report real BPE token count and context use for text.")
    tokenizer_parser.add_argument("--text", required=True)

    inference = sub.add_parser("generate", help="Generate real tokens from the trained local checkpoint.")
    inference.add_argument("--prompt", required=True)
    inference.add_argument("--max-new-tokens", type=int, default=256)
    inference.add_argument("--temperature", type=float, default=0.7)
    inference.add_argument("--top-k", type=int, default=40)

    fine = sub.add_parser("finetune", help="Supervised fine-tune the local checkpoint on prompt/completion JSONL.")
    fine.add_argument("--dataset", required=True, help="Absolute path to a JSONL prompt/completion dataset.")
    fine.add_argument("--steps", type=int, default=1000)
    fine.add_argument("--batch-size", type=int, default=4)
    fine.add_argument("--learning-rate", type=float, default=1e-4)
    fine.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])

    sub.add_parser("status", help="Report actual local checkpoint and training metrics.")
    evaluation = sub.add_parser("evaluate", help="Run a fresh sampled loss evaluation on a local corpus.")
    evaluation.add_argument("--root", required=True, help="Absolute local corpus directory to evaluate.")
    evaluation.add_argument("--batches", type=int, default=16)
    evaluation.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    serve = sub.add_parser("serve", help="Start the loopback-only local AI API.")
    serve.add_argument("--host", choices=["127.0.0.1"], default="127.0.0.1")
    serve.add_argument("--port", type=int, default=3002)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "inspect-data":
        corpus = load_corpus([args.root], max_tokens=args.max_tokens)
        print(json.dumps({
            "files": len(corpus.files), "source_bytes": corpus.source_bytes,
            "tokens": len(corpus.token_ids), "skipped_files": corpus.skipped_files,
            "sample_files": corpus.files[:20]
        }, indent=2))
        return 0
    if args.command == "train":
        config = ModelConfig(vocab_size=args.vocab_size, seq_len=args.seq_len, n_layer=args.layers, n_head=args.heads, n_embd=args.width)
        config.validate()
        checkpoint = Path(os.environ.get("SYNTRA_CHECKPOINT", str(
            Path(__file__).resolve().parent.parent / "data" / "checkpoints" / "syntra.pt"
        )))
        result = train(
            roots=[args.root], checkpoint=checkpoint, steps=args.steps,
            batch_size=args.batch_size, config=config, learning_rate=args.learning_rate,
            max_tokens=args.max_tokens, device_name=args.device, seed=args.seed,
            resume=args.resume,
            callback=lambda item: print(json.dumps(item), flush=True),
        )
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "finetune":
        checkpoint = Path(os.environ.get("SYNTRA_CHECKPOINT", str(
            Path(__file__).resolve().parent.parent / "data" / "checkpoints" / "syntra.pt"
        )))
        result = finetune(
            dataset_path=args.dataset,
            checkpoint=checkpoint,
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device_name=args.device,
            callback=lambda item: print(json.dumps(item), flush=True),
        )
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "tokenize":
        print(json.dumps(tokenize_text(args.text), indent=2))
        return 0
    if args.command == "generate":
        print(json.dumps(generate(args.prompt, args.max_new_tokens, args.temperature, args.top_k), indent=2))
        return 0
    if args.command == "status":
        print(json.dumps(status(), indent=2))
        return 0
    if args.command == "evaluate":
        print(json.dumps(evaluate(args.root, batches=args.batches, device_name=args.device), indent=2))
        return 0
    if args.command == "serve":
        os.environ["SYNTRA_AI_PORT"] = str(args.port)
        from .server import main as serve_main
        serve_main()
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())

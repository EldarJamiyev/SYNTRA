"""Command-line entry point for SYNTRA."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from syntra.config import SyntraConfig
from syntra.data import prepare_corpus
from syntra.evaluation import evaluate_checkpoint, load_checkpoint
from syntra.generation import generate_text
from syntra.training import train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="syntra",
        description="Train and run the SYNTRA coding model from random initialization.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="Tokenize a local source corpus.")
    prepare.add_argument("--raw-dir", default="data/raw")
    prepare.add_argument("--out", default="data/processed/tokens.bin")
    prepare.add_argument("--max-file-bytes", type=int, default=2_000_000)
    prepare.add_argument("--max-total-bytes", type=int, default=50_000_000)

    train_parser = commands.add_parser("train", help="Train a model from random weights.")
    train_parser.add_argument("--config", default="config/default.json")
    train_parser.add_argument("--resume", default=None)

    generate = commands.add_parser("generate", help="Generate text from a trained checkpoint.")
    generate.add_argument("--checkpoint", default="checkpoints/latest.pt")
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--max-new-tokens", type=int, default=128)
    generate.add_argument("--temperature", type=float, default=0.8)
    generate.add_argument("--top-k", type=int, default=40)
    generate.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    evaluate = commands.add_parser("evaluate", help="Measure held-out next-token loss.")
    evaluate.add_argument("--checkpoint", default="checkpoints/latest.pt")
    evaluate.add_argument("--data-file", default="data/processed/tokens.bin")

    serve = commands.add_parser("serve", help="Start the loopback-only local API.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--checkpoint", default="checkpoints/latest.pt")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        if args.command == "prepare":
            result = prepare_corpus(
                args.raw_dir,
                args.out,
                max_file_bytes=args.max_file_bytes,
                max_total_bytes=args.max_total_bytes,
            )
            print(json.dumps(result, indent=2))
            return

        if args.command == "train":
            config = SyntraConfig.load(args.config)
            checkpoint = train(config, resume_from=args.resume)
            print("Saved: " + str(checkpoint))
            return

        if args.command == "generate":
            model, config, device, _ = load_checkpoint(
                args.checkpoint, requested_device=args.device
            )
            output = generate_text(
                model,
                args.prompt,
                config,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                device=device,
            )
            sys.stdout.write(output)
            if not output.endswith("\n"):
                sys.stdout.write("\n")
            return

        if args.command == "evaluate":
            print(json.dumps(evaluate_checkpoint(args.checkpoint, args.data_file), indent=2))
            return

        if args.command == "serve":
            if not 1 <= args.port <= 65535:
                raise ValueError("port must be between 1 and 65535.")
            os.environ["SYNTRA_CHECKPOINT"] = str(Path(args.checkpoint).resolve())
            import uvicorn
            uvicorn.run("syntra.api:app", host=args.host, port=args.port, reload=False)
            return
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print("SYNTRA error: " + str(exc), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

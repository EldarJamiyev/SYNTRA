from pathlib import Path

from syntra.config import SyntraConfig
from syntra.data import prepare_corpus
from syntra.evaluation import load_checkpoint
from syntra.training import train


def test_tiny_training_run_saves_loadable_checkpoint(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    sample = "def add(a, b):\n    return a + b\n\n"
    (raw / "training.py").write_text(sample * 80, encoding="utf-8")
    token_file = tmp_path / "processed" / "tokens.bin"
    prepared = prepare_corpus(raw, token_file, max_file_bytes=100_000, max_total_bytes=200_000)
    assert prepared["tokens"] > 100

    config = SyntraConfig(
        seed=7,
        vocab_size=260,
        block_size=8,
        n_layer=1,
        n_head=2,
        n_embd=16,
        dropout=0.0,
        batch_size=2,
        learning_rate=0.001,
        max_steps=2,
        eval_interval=1,
        eval_batches=1,
        grad_clip=1.0,
        num_threads=1,
        device="cpu",
        data_file=str(token_file),
        checkpoint_dir=str(tmp_path / "checkpoints"),
    )
    progress = []
    checkpoint = train(config, progress_callback=progress.append)

    assert checkpoint.is_file()
    assert len(progress) == 2
    model, saved_config, device, payload = load_checkpoint(checkpoint, requested_device="cpu")
    assert saved_config.n_embd == 16
    assert device.type == "cpu"
    assert payload["step"] == 2
    assert model.parameter_count() > 100

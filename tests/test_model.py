import torch

from syntra.config import SyntraConfig
from syntra.model import SyntraModel


def small_config() -> SyntraConfig:
    return SyntraConfig(
        block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0,
        batch_size=2, max_steps=1, eval_batches=1,
    )


def test_model_forward_and_loss() -> None:
    config = small_config()
    model = SyntraModel(config)
    inputs = torch.randint(0, config.vocab_size, (2, 8))
    targets = torch.randint(0, config.vocab_size, (2, 8))
    logits, loss = model(inputs, targets)
    assert logits.shape == (2, 8, config.vocab_size)
    assert loss is not None and torch.isfinite(loss)


def test_model_rejects_long_context() -> None:
    config = small_config()
    model = SyntraModel(config)
    inputs = torch.zeros((1, config.block_size + 1), dtype=torch.long)
    try:
        model(inputs)
    except ValueError as exc:
        assert "exceeds block_size" in str(exc)
    else:
        raise AssertionError("Expected input-length validation to fail.")


def test_model_has_trainable_parameters() -> None:
    model = SyntraModel(small_config())
    assert model.parameter_count() > 1000

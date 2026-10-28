import pytest

from syntra.config import SyntraConfig


def test_default_configuration_validates() -> None:
    assert SyntraConfig().validate().block_size >= 8


def test_attention_dimensions_must_divide() -> None:
    with pytest.raises(ValueError, match="divisible"):
        SyntraConfig(n_embd=383, n_head=6).validate()


def test_vocab_must_match_byte_tokenizer() -> None:
    with pytest.raises(ValueError, match="vocab_size=260"):
        SyntraConfig(vocab_size=100).validate()


def test_mapping_ignores_unknown_fields() -> None:
    config = SyntraConfig.from_mapping({"n_layer": 2, "future_flag": True})
    assert config.n_layer == 2

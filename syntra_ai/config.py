from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class ModelConfig:
    # Default tokenizer is learned from the user's local corpus.
    vocab_size: int = 1024
    seq_len: int = 4096
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.05
    bias: bool = True
    use_rope: bool = False
    use_rmsnorm: bool = False
    use_swiglu: bool = False

    @property
    def block_size(self) -> int:
        return self.seq_len

    def validate(self) -> None:
        if type(self.vocab_size) is not int or self.vocab_size < 258 or self.vocab_size > 8192:
            raise ValueError("vocab_size must be between 258 and 8,192.")
        if type(self.seq_len) is not int or self.seq_len < 16 or self.seq_len > 16384:
            raise ValueError("seq_len must be between 16 and 16,384 tokens.")
        if type(self.n_layer) is not int or self.n_layer < 1 or self.n_layer > 24:
            raise ValueError("n_layer must be between 1 and 24.")
        if type(self.n_head) is not int or self.n_head < 1 or self.n_head > 16:
            raise ValueError("n_head must be between 1 and 16.")
        if type(self.n_embd) is not int or self.n_embd < 32 or self.n_embd > 1024:
            raise ValueError("n_embd must be between 32 and 1,024.")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head.")
        if isinstance(self.dropout, bool) or not isinstance(self.dropout, (int, float)) or not math.isfinite(float(self.dropout)) or not 0.0 <= float(self.dropout) < 0.5:
            raise ValueError("dropout must be between 0 and 0.5.")
        if self.use_rope and (self.n_embd // self.n_head) % 2 != 0:
            raise ValueError("head_dim (n_embd // n_head) must be even when use_rope is enabled.")

    def to_dict(self) -> dict:
        self.validate()
        result = {
            "vocab_size": self.vocab_size,
            "seq_len": self.seq_len,
            "n_layer": self.n_layer,
            "n_head": self.n_head,
            "n_embd": self.n_embd,
            "dropout": self.dropout,
        }
        if not self.bias:
            result["bias"] = False
        if self.use_rope:
            result["use_rope"] = True
        if self.use_rmsnorm:
            result["use_rmsnorm"] = True
        if self.use_swiglu:
            result["use_swiglu"] = True
        return result

    @classmethod
    def from_dict(cls, value: dict) -> "ModelConfig":
        if not isinstance(value, dict):
            raise ValueError("Model configuration must be a dictionary.")
        normalized = dict(value)
        if "seq_len" not in normalized and "block_size" in normalized:
            normalized["seq_len"] = normalized["block_size"]
        accepted = {key: normalized[key] for key in cls.__dataclass_fields__ if key in normalized}
        config = cls(**accepted)
        config.validate()
        return config

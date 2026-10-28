from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 258  # 256 UTF-8 byte values + BOS + EOS
    seq_len: int = 1024
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.05

    def validate(self) -> None:
        if self.vocab_size != 258:
            raise ValueError("SYNTRA currently uses its fixed 258-token byte vocabulary.")
        if self.seq_len < 16 or self.seq_len > 2048:
            raise ValueError("seq_len must be between 16 and 2048.")
        if self.n_layer < 1 or self.n_layer > 24:
            raise ValueError("n_layer must be between 1 and 24.")
        if self.n_head < 1 or self.n_head > 16:
            raise ValueError("n_head must be between 1 and 16.")
        if self.n_embd < 32 or self.n_embd > 1024:
            raise ValueError("n_embd must be between 32 and 1024.")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head.")
        if not 0.0 <= self.dropout < 0.5:
            raise ValueError("dropout must be between 0 and 0.5.")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "ModelConfig":
        accepted = {key: value[key] for key in cls.__dataclass_fields__ if key in value}
        config = cls(**accepted)
        config.validate()
        return config

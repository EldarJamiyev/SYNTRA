"""Dependency-free UTF-8 byte tokenizer.

Token IDs 0-3 are reserved special tokens. Each UTF-8 byte maps to one
token ID in the range 4-259. This tokenizer has no learned merges and can
represent every byte sequence.
"""

from __future__ import annotations

from dataclasses import dataclass

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3
BYTE_OFFSET = 4
VOCAB_SIZE = 260


@dataclass(frozen=True)
class ByteTokenizer:
    vocab_size: int = VOCAB_SIZE

    def __post_init__(self) -> None:
        if self.vocab_size != VOCAB_SIZE:
            raise ValueError("ByteTokenizer requires vocab_size=260.")

    def encode(self, text: str, *, add_bos: bool = True, add_eos: bool = False) -> list[int]:
        token_ids: list[int] = []
        if add_bos:
            token_ids.append(BOS_ID)
        token_ids.extend(byte + BYTE_OFFSET for byte in text.encode("utf-8"))
        if add_eos:
            token_ids.append(EOS_ID)
        return token_ids

    def decode(self, token_ids: list[int] | tuple[int, ...]) -> str:
        decoded = bytearray()
        for token_id in token_ids:
            if BYTE_OFFSET <= int(token_id) < VOCAB_SIZE:
                decoded.append(int(token_id) - BYTE_OFFSET)
        return decoded.decode("utf-8", errors="replace")

    def token_to_string(self, token_id: int) -> str:
        if token_id == PAD_ID:
            return "<PAD>"
        if token_id == BOS_ID:
            return "<BOS>"
        if token_id == EOS_ID:
            return "<EOS>"
        if token_id == UNK_ID:
            return "<UNK>"
        if BYTE_OFFSET <= token_id < VOCAB_SIZE:
            value = token_id - BYTE_OFFSET
            if 32 <= value <= 126:
                return chr(value)
            return "0x" + format(value, "02x")
        raise ValueError("Token ID is outside the vocabulary.")

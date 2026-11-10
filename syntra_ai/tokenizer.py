"""Trainable byte-pair tokenizer for source code, with lossless UTF-8 round-tripping.

The base alphabet is the 256 byte values. Merge IDs are learned from the explicitly
selected local corpus; BOS/EOS remain reserved, stable IDs 256 and 257.
"""
from collections import Counter
from dataclasses import dataclass
import re
from typing import Iterable

BOS_ID = 256
EOS_ID = 257
FIRST_MERGE_ID = 258
FORMAT = "syntra-byte-bpe-v1"
_PIECE = re.compile(r"\s+|[^\W\d]\w*|\d+|[^\w\s]+", re.UNICODE)


def split_pieces(text: str) -> list[str]:
    pieces = _PIECE.findall(text)
    if "".join(pieces) != text:
        # The fallback preserves every code point even for unusual Unicode input.
        return list(text)
    return pieces


@dataclass(frozen=True)
class ByteBPETokenizer:
    merges: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        seen: set[tuple[int, int]] = set()
        vocab = FIRST_MERGE_ID
        for index, pair in enumerate(self.merges):
            if len(pair) != 2 or any(
                type(token) is not int or token < 0 or token in (BOS_ID, EOS_ID) or token >= vocab
                for token in pair
            ):
                raise ValueError(f"Invalid BPE merge at index {index}.")
            if pair in seen:
                raise ValueError(f"Duplicate BPE merge at index {index}.")
            seen.add(pair)
            vocab += 1
        if vocab > 8192:
            raise ValueError("Tokenizer vocabulary may not exceed 8,192 IDs.")

    @property
    def vocab_size(self) -> int:
        return FIRST_MERGE_ID + len(self.merges)

    @property
    def ranks(self) -> dict[tuple[int, int], int]:
        return {pair: rank for rank, pair in enumerate(self.merges)}

    @classmethod
    def empty(cls) -> "ByteBPETokenizer":
        return cls(())

    @classmethod
    def train(
        cls,
        documents: Iterable[str],
        vocab_size: int = 1024,
        max_training_bytes: int = 100_000,
        min_pair_frequency: int = 2,
    ) -> "ByteBPETokenizer":
        if not FIRST_MERGE_ID <= vocab_size <= 8192:
            raise ValueError("vocab_size must be between 258 and 8,192.")
        if max_training_bytes < 1024:
            raise ValueError("max_training_bytes must be at least 1,024.")
        # Bound tokenizer-training CPU/RAM independently of the full model corpus.
        sample = bytearray()
        for document in documents:
            if len(sample) >= max_training_bytes:
                break
            remaining = max_training_bytes - len(sample)
            piece = document.encode("utf-8")[:remaining]
            sample.extend(piece)
        text = bytes(sample).decode("utf-8", errors="ignore")
        sequences = [
            list(piece.encode("utf-8"))
            for piece in split_pieces(text)
            if piece
        ]
        merges: list[tuple[int, int]] = []
        target_merges = vocab_size - FIRST_MERGE_ID
        while len(merges) < target_merges:
            pair_counts: Counter[tuple[int, int]] = Counter()
            for sequence in sequences:
                if len(sequence) > 1:
                    pair_counts.update(zip(sequence, sequence[1:]))
            if not pair_counts:
                break
            pair, frequency = max(pair_counts.items(), key=lambda item: (item[1], -item[0][0], -item[0][1]))
            if frequency < min_pair_frequency:
                break
            new_id = FIRST_MERGE_ID + len(merges)
            replaced: list[list[int]] = []
            for sequence in sequences:
                output: list[int] = []
                index = 0
                while index < len(sequence):
                    if index + 1 < len(sequence) and (sequence[index], sequence[index + 1]) == pair:
                        output.append(new_id)
                        index += 2
                    else:
                        output.append(sequence[index])
                        index += 1
                replaced.append(output)
            sequences = replaced
            merges.append(pair)
        return cls(tuple(merges))

    @classmethod
    def from_dict(cls, value: dict | None) -> "ByteBPETokenizer":
        if not value:
            return cls.empty()
        if value.get("format") != FORMAT:
            raise ValueError("Unsupported SYNTRA tokenizer format.")
        merges_value = value.get("merges")
        if not isinstance(merges_value, list):
            raise ValueError("Tokenizer merges must be a list.")
        merges: list[tuple[int, int]] = []
        for index, pair in enumerate(merges_value):
            if not isinstance(pair, list) or len(pair) != 2 or any(type(token) is not int for token in pair):
                raise ValueError(f"Malformed tokenizer merge at index {index}.")
            merges.append((pair[0], pair[1]))
        tokenizer = cls(tuple(merges))
        declared = value.get("vocab_size")
        if declared is not None and declared != tokenizer.vocab_size:
            raise ValueError("Tokenizer vocabulary size does not match its merge table.")
        return tokenizer

    def to_dict(self) -> dict:
        return {
            "format": FORMAT,
            "vocab_size": self.vocab_size,
            "merges": [[left, right] for left, right in self.merges],
        }

    def _merge_piece(
        self,
        piece: str,
        ranks: dict[tuple[int, int], int],
        new_ids: dict[tuple[int, int], int],
    ) -> list[int]:
        tokens = list(piece.encode("utf-8"))
        if len(tokens) < 2 or not self.merges:
            return tokens
        while len(tokens) > 1:
            best_rank = None
            best_pair = None
            for left, right in zip(tokens, tokens[1:]):
                pair = (left, right)
                rank = ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_pair = rank, pair
                    if rank == 0:
                        break
            if best_pair is None:
                break
            new_id = new_ids[best_pair]
            merged: list[int] = []
            index = 0
            while index < len(tokens):
                if index + 1 < len(tokens) and (tokens[index], tokens[index + 1]) == best_pair:
                    merged.append(new_id)
                    index += 2
                else:
                    merged.append(tokens[index])
                    index += 1
            tokens = merged
        return tokens

    def encode(self, text: str) -> list[int]:
        result: list[int] = []
        ranks = self.ranks
        new_ids = {pair: FIRST_MERGE_ID + rank for rank, pair in enumerate(self.merges)}
        for piece in split_pieces(text):
            result.extend(self._merge_piece(piece, ranks, new_ids))
        return result

    def decode(self, token_ids: Iterable[int]) -> str:
        expansions: dict[int, bytes] = {value: bytes([value]) for value in range(256)}
        for index, (left, right) in enumerate(self.merges):
            if left not in expansions or right not in expansions:
                raise ValueError("Tokenizer merges are not in dependency order.")
            expansions[FIRST_MERGE_ID + index] = expansions[left] + expansions[right]
        output = bytearray()
        for raw_id in token_ids:
            token_id = int(raw_id)
            if 0 <= token_id < 256:
                output.append(token_id)
            elif token_id == BOS_ID or token_id == EOS_ID:
                continue
            elif FIRST_MERGE_ID <= token_id < self.vocab_size:
                output.extend(expansions[token_id])
            else:
                raise ValueError(f"Token ID {token_id} is outside this tokenizer's vocabulary.")
        return bytes(output).decode("utf-8", errors="replace")

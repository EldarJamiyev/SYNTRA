"""Self-contained byte-pair tokenizer trained from a local UTF-8 text file.

The initial alphabet is the 256 byte values. Merge ranks are learned only from
the supplied local corpus. BOS/EOS use stable IDs 256 and 257.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Iterable

BOS_ID = 256
EOS_ID = 257
FIRST_MERGE_ID = 258
MAX_VOCAB_SIZE = 8192
MAX_TRAINING_BYTES = 10 * 1024 * 1024
MAX_TOKENIZER_FILE_BYTES = 2 * 1024 * 1024
FORMAT = "syntra-local-byte-bpe-v1"


@dataclass(frozen=True)
class BytePairTokenizer:
    """Byte-level BPE tokenizer whose merge table is fully serialized locally."""

    merges: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.merges, tuple):
            raise ValueError("Tokenizer merges must be an immutable tuple.")
        seen: set[tuple[int, int]] = set()
        next_id = FIRST_MERGE_ID
        for index, pair in enumerate(self.merges):
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError(f"Malformed tokenizer merge at index {index}.")
            left, right = pair
            if (
                type(left) is not int
                or type(right) is not int
                or left < 0
                or right < 0
                or left >= next_id
                or right >= next_id
                or left in (BOS_ID, EOS_ID)
                or right in (BOS_ID, EOS_ID)
            ):
                raise ValueError(f"Invalid token reference in merge {index}.")
            if pair in seen:
                raise ValueError(f"Duplicate BPE merge at index {index}.")
            seen.add(pair)
            next_id += 1
        if next_id > MAX_VOCAB_SIZE:
            raise ValueError(f"Tokenizer vocabulary may not exceed {MAX_VOCAB_SIZE:,} IDs.")

    @property
    def vocab_size(self) -> int:
        return FIRST_MERGE_ID + len(self.merges)

    @property
    def ranks(self) -> dict[tuple[int, int], int]:
        return {pair: rank for rank, pair in enumerate(self.merges)}

    @classmethod
    def empty(cls) -> "BytePairTokenizer":
        """Return the lossless byte-only tokenizer (no learned merges)."""
        return cls(())

    @classmethod
    def train_from_text(
        cls,
        text: str,
        *,
        vocab_size: int = 1024,
        max_training_bytes: int = 100_000,
        min_pair_frequency: int = 2,
    ) -> "BytePairTokenizer":
        """Learn deterministic BPE merges from a bounded local text sample."""
        if not isinstance(text, str):
            raise TypeError("Tokenizer training input must be a string.")
        if type(vocab_size) is not int or not FIRST_MERGE_ID <= vocab_size <= MAX_VOCAB_SIZE:
            raise ValueError(f"vocab_size must be between {FIRST_MERGE_ID} and {MAX_VOCAB_SIZE}.")
        if (
            type(max_training_bytes) is not int
            or not 1024 <= max_training_bytes <= MAX_TRAINING_BYTES
        ):
            raise ValueError(
                f"max_training_bytes must be between 1,024 and {MAX_TRAINING_BYTES:,}."
            )
        if type(min_pair_frequency) is not int or min_pair_frequency < 1:
            raise ValueError("min_pair_frequency must be a positive integer.")

        raw_sample = text.encode("utf-8")[:max_training_bytes]
        if not raw_sample:
            raise ValueError("The tokenizer training text is empty.")
        try:
            sample = raw_sample.decode("utf-8")
        except UnicodeDecodeError as exc:
            # Truncation may end inside the final UTF-8 codepoint; never discard
            # or replace invalid bytes from the middle of the selected sample.
            if exc.reason == "unexpected end of data":
                sample = raw_sample[:exc.start].decode("utf-8")
            else:
                raise ValueError("The tokenizer training text must be valid UTF-8.") from None
        if not sample:
            raise ValueError("The selected tokenizer training prefix contains no complete UTF-8 text.")

        sequences = [list(sample.encode("utf-8"))]
        merges: list[tuple[int, int]] = []
        merge_target = vocab_size - FIRST_MERGE_ID

        while len(merges) < merge_target:
            pair_counts: Counter[tuple[int, int]] = Counter()
            for sequence in sequences:
                if len(sequence) > 1:
                    pair_counts.update(zip(sequence, sequence[1:]))
            if not pair_counts:
                break
            pair, frequency = max(
                pair_counts.items(),
                key=lambda item: (item[1], -item[0][0], -item[0][1]),
            )
            if frequency < min_pair_frequency:
                break

            new_id = FIRST_MERGE_ID + len(merges)
            replacement_sequences: list[list[int]] = []
            for sequence in sequences:
                merged: list[int] = []
                index = 0
                while index < len(sequence):
                    if (
                        index + 1 < len(sequence)
                        and (sequence[index], sequence[index + 1]) == pair
                    ):
                        merged.append(new_id)
                        index += 2
                    else:
                        merged.append(sequence[index])
                        index += 1
                replacement_sequences.append(merged)
            sequences = replacement_sequences
            merges.append(pair)

        return cls(tuple(merges))

    @classmethod
    def train_from_file(
        cls,
        dataset_path: str | os.PathLike[str] = "dataset.txt",
        *,
        vocab_size: int = 1024,
        max_training_bytes: int = 100_000,
        min_pair_frequency: int = 2,
    ) -> "BytePairTokenizer":
        """Learn merges from a bounded prefix of one local UTF-8 corpus file."""
        if type(vocab_size) is not int or not FIRST_MERGE_ID <= vocab_size <= MAX_VOCAB_SIZE:
            raise ValueError(f"vocab_size must be between {FIRST_MERGE_ID} and {MAX_VOCAB_SIZE}.")
        if (
            type(max_training_bytes) is not int
            or not 1024 <= max_training_bytes <= MAX_TRAINING_BYTES
        ):
            raise ValueError(
                f"max_training_bytes must be between 1,024 and {MAX_TRAINING_BYTES:,}."
            )
        if type(min_pair_frequency) is not int or min_pair_frequency < 1:
            raise ValueError("min_pair_frequency must be a positive integer.")

        path = Path(dataset_path).expanduser()
        try:
            resolved_path = path.resolve(strict=True)
            if not resolved_path.is_file():
                raise ValueError("The tokenizer training dataset must be a regular file.")
            with resolved_path.open("rb") as handle:
                raw_sample = handle.read(max_training_bytes + 1)
        except (OSError, RuntimeError) as exc:
            raise ValueError(f"Cannot read local tokenizer dataset: {exc}") from None

        sample = raw_sample[:max_training_bytes]
        if not sample:
            raise ValueError("The tokenizer training dataset is empty.")
        # Reject corrupted text instead of silently training on replacement glyphs.
        try:
            sample.decode("utf-8")
        except UnicodeDecodeError as exc:
            # A bounded read can end in the middle of a multi-byte UTF-8 codepoint.
            if len(raw_sample) > max_training_bytes and exc.reason == "unexpected end of data":
                sample = sample[:exc.start]
                if not sample:
                    raise ValueError("The selected training prefix contains no complete UTF-8 text.") from None
                sample.decode("utf-8")
            else:
                raise ValueError("The tokenizer training dataset must contain valid UTF-8 text.") from None

        sequences = [list(sample)]
        merges: list[tuple[int, int]] = []
        merge_target = vocab_size - FIRST_MERGE_ID

        while len(merges) < merge_target:
            pair_counts: Counter[tuple[int, int]] = Counter()
            for sequence in sequences:
                if len(sequence) > 1:
                    pair_counts.update(zip(sequence, sequence[1:]))
            if not pair_counts:
                break
            pair, frequency = max(
                pair_counts.items(),
                key=lambda item: (item[1], -item[0][0], -item[0][1]),
            )
            if frequency < min_pair_frequency:
                break

            new_id = FIRST_MERGE_ID + len(merges)
            replacement_sequences: list[list[int]] = []
            for sequence in sequences:
                merged: list[int] = []
                index = 0
                while index < len(sequence):
                    if (
                        index + 1 < len(sequence)
                        and (sequence[index], sequence[index + 1]) == pair
                    ):
                        merged.append(new_id)
                        index += 2
                    else:
                        merged.append(sequence[index])
                        index += 1
                replacement_sequences.append(merged)
            sequences = replacement_sequences
            merges.append(pair)

        return cls(tuple(merges))

    @classmethod
    def from_dict(cls, value: dict) -> "BytePairTokenizer":
        if not isinstance(value, dict) or value.get("format") != FORMAT:
            raise ValueError("Unsupported local SYNTRA tokenizer format.")
        raw_merges = value.get("merges")
        if not isinstance(raw_merges, list) or len(raw_merges) > MAX_VOCAB_SIZE - FIRST_MERGE_ID:
            raise ValueError("Tokenizer merges must be a bounded JSON array.")
        merges: list[tuple[int, int]] = []
        for index, pair in enumerate(raw_merges):
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or any(type(token) is not int for token in pair)
            ):
                raise ValueError(f"Malformed tokenizer merge at index {index}.")
            merges.append((pair[0], pair[1]))
        tokenizer = cls(tuple(merges))
        declared_size = value.get("vocab_size")
        if declared_size is not None and (
            type(declared_size) is not int or declared_size != tokenizer.vocab_size
        ):
            raise ValueError("Tokenizer vocabulary size does not match its merge table.")
        return tokenizer

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "BytePairTokenizer":
        """Load a local tokenizer JSON file with a strict size limit."""
        source = Path(path).expanduser()
        try:
            resolved = source.resolve(strict=True)
            if not resolved.is_file():
                raise ValueError("Tokenizer file must be a regular file.")
            size = resolved.stat().st_size
            if size < 1 or size > MAX_TOKENIZER_FILE_BYTES:
                raise ValueError(
                    f"Tokenizer file must be between 1 and {MAX_TOKENIZER_FILE_BYTES:,} bytes."
                )
            value = json.loads(resolved.read_text(encoding="utf-8"))
        except (OSError, RuntimeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot load local tokenizer file: {exc}") from None
        return cls.from_dict(value)

    def to_dict(self) -> dict:
        return {
            "format": FORMAT,
            "vocab_size": self.vocab_size,
            "base_vocab_size": FIRST_MERGE_ID,
            "merges": [[left, right] for left, right in self.merges],
        }

    def save(self, path: str | os.PathLike[str]) -> Path:
        """Atomically save tokenizer vocabulary and merge ranks as UTF-8 JSON."""
        destination = Path(path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
        ) + "\n"
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_name = handle.name
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, destination)
        except OSError:
            if temporary_name is not None:
                try:
                    os.unlink(temporary_name)
                except OSError:
                    pass
            raise
        return destination.resolve()

    def _merge_bytes(
        self,
        raw_bytes: bytes,
        ranks: dict[tuple[int, int], int],
    ) -> list[int]:
        tokens = list(raw_bytes)
        while len(tokens) > 1:
            best_rank: int | None = None
            best_pair: tuple[int, int] | None = None
            for left, right in zip(tokens, tokens[1:]):
                rank = ranks.get((left, right))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_pair = (left, right)
            if best_pair is None:
                break

            new_id = FIRST_MERGE_ID + best_rank
            merged: list[int] = []
            index = 0
            while index < len(tokens):
                if (
                    index + 1 < len(tokens)
                    and (tokens[index], tokens[index + 1]) == best_pair
                ):
                    merged.append(new_id)
                    index += 2
                else:
                    merged.append(tokens[index])
                    index += 1
            tokens = merged
        return tokens

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        """Encode arbitrary Unicode text losslessly through its UTF-8 bytes."""
        if not isinstance(text, str):
            raise TypeError("Tokenizer input must be a string.")
        if type(add_bos) is not bool or type(add_eos) is not bool:
            raise ValueError("add_bos and add_eos must be booleans.")
        ranks = self.ranks
        encoded = self._merge_bytes(text.encode("utf-8"), ranks)
        if add_bos:
            encoded.insert(0, BOS_ID)
        if add_eos:
            encoded.append(EOS_ID)
        return encoded

    def decode(
        self,
        token_ids: Iterable[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token IDs to Unicode text; invalid IDs are rejected."""
        if type(skip_special_tokens) is not bool:
            raise ValueError("skip_special_tokens must be a boolean.")

        expansions: dict[int, bytes] = {
            token_id: bytes((token_id,)) for token_id in range(256)
        }
        for index, (left, right) in enumerate(self.merges):
            if left not in expansions or right not in expansions:
                raise ValueError("Tokenizer merges are not in dependency order.")
            expansions[FIRST_MERGE_ID + index] = expansions[left] + expansions[right]

        output = bytearray()
        for raw_id in token_ids:
            if type(raw_id) is not int:
                try:
                    token_id = int(raw_id)
                except (TypeError, ValueError, OverflowError):
                    raise ValueError("Token IDs must be integers.") from None
                if token_id != raw_id:
                    raise ValueError("Token IDs must be integers.")
            else:
                token_id = raw_id

            if 0 <= token_id < 256:
                output.append(token_id)
            elif token_id in (BOS_ID, EOS_ID):
                if not skip_special_tokens:
                    raise ValueError("Special tokens cannot be decoded as text.")
            elif FIRST_MERGE_ID <= token_id < self.vocab_size:
                output.extend(expansions[token_id])
            else:
                raise ValueError(
                    f"Token ID {token_id} is outside this tokenizer's vocabulary."
                )
        return bytes(output).decode("utf-8", errors="replace")


__all__ = [
    "BOS_ID",
    "EOS_ID",
    "FIRST_MERGE_ID",
    "MAX_VOCAB_SIZE",
    "BytePairTokenizer",
]

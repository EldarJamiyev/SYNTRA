from pathlib import Path

from syntra.data import prepare_corpus, read_token_stream
from syntra.tokenizer import BOS_ID, EOS_ID, ByteTokenizer


def test_prepare_and_read_local_corpus(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "example.py").write_text("def square(x):\n    return x * x\n", encoding="utf-8")
    output = tmp_path / "processed" / "tokens.bin"

    result = prepare_corpus(raw, output, max_file_bytes=4096, max_total_bytes=8192)
    tokens = read_token_stream(output)
    assert result["files"] == 1
    assert result["tokens"] == len(tokens)
    assert int(tokens[0]) == BOS_ID
    assert EOS_ID in tokens.tolist()
    decoded = ByteTokenizer().decode(tokens.tolist())
    assert "def square" in decoded
    assert "return x * x" in decoded


def test_prepare_fails_on_empty_corpus(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "notes.bin").write_bytes(b"not a supported source extension")
    try:
        prepare_corpus(raw, tmp_path / "tokens.bin")
    except ValueError as exc:
        assert "No usable source files" in str(exc)
    else:
        raise AssertionError("Expected an empty corpus to be rejected.")

from syntra.tokenizer import BOS_ID, EOS_ID, ByteTokenizer, VOCAB_SIZE


def test_vocab_size_and_special_tokens() -> None:
    tokenizer = ByteTokenizer()
    assert tokenizer.vocab_size == VOCAB_SIZE == 260
    assert tokenizer.encode("x", add_bos=True, add_eos=True)[0] == BOS_ID
    assert tokenizer.encode("x", add_bos=True, add_eos=True)[-1] == EOS_ID


def test_utf8_round_trip() -> None:
    tokenizer = ByteTokenizer()
    samples = [
        "",
        "print('hello')",
        "def greet(name: str) -> str:\n    return 'hi ' + name\n",
        "emoji: 🚀 and Azerbaijani: ə, ş, ğ",
        "\x00\x01",
    ]
    for sample in samples:
        encoded = tokenizer.encode(sample, add_bos=True, add_eos=True)
        assert tokenizer.decode(encoded) == sample


def test_token_range() -> None:
    tokenizer = ByteTokenizer()
    assert all(4 <= token_id < 260 for token_id in tokenizer.encode("SYNTRA", add_bos=False))

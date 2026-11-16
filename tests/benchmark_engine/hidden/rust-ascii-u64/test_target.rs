use syntra_ascii_u64::parse_ascii_u64;

#[test]
fn accepts_only_unsigned_ascii_decimal_digits() {
    assert_eq!(parse_ascii_u64("0"), Some(0));
    assert_eq!(parse_ascii_u64("00042"), Some(42));
    assert_eq!(parse_ascii_u64("18446744073709551615"), Some(u64::MAX));
    for invalid in [
        "",
        "+1",
        "-1",
        " 1",
        "1 ",
        "1\n",
        "18446744073709551616",
        "\u{FF11}",
    ] {
        assert_eq!(parse_ascii_u64(invalid), None, "Input should be rejected: {invalid:?}");
    }
}

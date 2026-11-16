use syntra_ascii_u64::parse_ascii_u64;

#[test]
fn ordinary_decimal_values_still_parse() {
    assert_eq!(parse_ascii_u64("42"), Some(42));
    assert_eq!(parse_ascii_u64("0"), Some(0));
}

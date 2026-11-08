use syntra_safe_slice::safe_str_slice;

#[test]
fn utf8_boundaries_and_invalid_ranges_are_checked_without_panics() {
    assert_eq!(safe_str_slice("A🙂B", 1, 5), Some("🙂"));
    assert_eq!(safe_str_slice("A🙂B", 2, 5), None);
    assert_eq!(safe_str_slice("A🙂B", 1, 4), None);
    assert_eq!(safe_str_slice("abc", 2, 1), None);
    assert_eq!(safe_str_slice("abc", 0, 4), None);
}

use syntra_safe_slice::safe_str_slice;

#[test]
fn ascii_substring_behavior_remains_unchanged() {
    assert_eq!(safe_str_slice("network", 1, 4), Some("etw"));
    assert_eq!(safe_str_slice("network", 2, 2), Some(""));
}

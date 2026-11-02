use syntra_benchmark_rust_key_normalize::normalize_key;

#[test]
fn trims_and_lowercases_keys() {
    assert_eq!(normalize_key("  SYNTRA-Key  "), "syntra-key");
}

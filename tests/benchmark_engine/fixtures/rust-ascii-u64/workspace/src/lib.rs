pub fn parse_ascii_u64(value: &str) -> Option<u64> {
    value.trim().parse::<u64>().ok()
}

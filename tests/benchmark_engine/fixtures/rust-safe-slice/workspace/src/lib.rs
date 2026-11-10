/// Return the requested byte range without allocating or copying.
pub fn safe_str_slice(value: &str, start: usize, end: usize) -> Option<&str> {
    Some(&value[start..end])
}

pub fn bounded_window<T>(items: &[T], start: usize, limit: usize) -> &[T] {
    &items[start..start + limit]
}

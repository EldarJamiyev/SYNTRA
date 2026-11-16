use syntra_bounded_window::bounded_window;

#[test]
fn ordinary_bounded_window_behavior_is_preserved() {
    let items = [1, 2, 3, 4];
    assert_eq!(bounded_window(&items, 1, 2), &[2, 3]);
    assert_eq!(bounded_window(&items, 0, 4), &items);
}

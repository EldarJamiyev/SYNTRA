use syntra_bounded_window::bounded_window;

#[test]
fn clamps_ranges_and_avoids_overflow() {
    let items = [10, 20, 30, 40];
    assert_eq!(bounded_window(&items, 1, 2), &[20, 30]);
    assert_eq!(bounded_window(&items, 2, usize::MAX), &[30, 40]);
    assert_eq!(bounded_window(&items, 4, 1), &[]);
    assert_eq!(bounded_window(&items, usize::MAX, 2), &[]);
    assert_eq!(bounded_window(&items, 2, 0), &[]);
    assert_eq!(bounded_window(&items, 0, usize::MAX), &items);
    let borrowed = bounded_window(&items, 1, 2);
    assert_eq!(borrowed.as_ptr(), items[1..].as_ptr(), "The result must borrow the original storage without copying.");
}

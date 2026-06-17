from walk import cursor_positions, held_out


def test_positions_are_interior_word_boundaries():
    text = "the quick brown fox jumps over the lazy dog today"
    pos = cursor_positions(text, 3)
    assert len(pos) == 3
    assert all(text[p] == " " for p in pos)          # at a space
    assert all(0 < p < len(text) - 1 for p in pos)   # interior, room to continue
    assert pos == sorted(pos)                          # increasing => growing prefix


def test_held_out_is_text_after_cursor():
    text = "hello world foo"
    p = text.index(" ")            # after "hello"
    assert held_out(text, p).startswith(" world")

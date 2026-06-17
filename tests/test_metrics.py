from metrics import (next_word, next_word_match, prefix_overlap_tokens,
                     truncate_at_boundary, precision_coverage)


def test_next_word_strips_and_extracts():
    assert next_word("  fox jumped") == "fox"
    assert next_word("") == ""


def test_next_word_match_is_case_insensitive():
    assert next_word_match("Fox runs", "fox jumped") is True
    assert next_word_match("dog", "fox") is False
    assert next_word_match("", "") is False        # empty truth never matches


def test_prefix_overlap_counts_leading_matches():
    assert prefix_overlap_tokens("the quick red", "the quick brown") == 2
    assert prefix_overlap_tokens("a b c", "x y z") == 0


def test_truncate_stops_at_sentence_end():
    assert truncate_at_boundary("done here. and more") == "done here."
    assert truncate_at_boundary("no boundary at all") == "no boundary at all"


def test_precision_coverage_trades_coverage_for_precision():
    # (confidence, correct)
    recs = [(-0.1, True), (-0.5, True), (-2.0, False), (-3.0, False)]
    rows = precision_coverage(recs, thresholds=[-5.0, -1.0])
    assert rows[0] == (-5.0, 1.0, 0.5)   # show all 4: coverage 1.0, precision 0.5
    assert rows[1] == (-1.0, 0.5, 1.0)   # show 2 confident: coverage 0.5, precision 1.0

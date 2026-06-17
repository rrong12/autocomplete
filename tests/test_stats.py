from stats import summarize


def test_summarize_reports_median_p90_mean():
    s = summarize([10, 20, 30, 40, 100])
    assert s["n"] == 5
    assert s["median"] == 30
    assert s["p90"] == 100        # 90th-percentile index into sorted samples
    assert round(s["mean"], 1) == 40.0


def test_summarize_empty_is_safe():
    s = summarize([])
    assert s["n"] == 0 and s["median"] == 0.0

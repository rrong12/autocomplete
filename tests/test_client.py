from client import parse_timings


def test_parse_timings_extracts_prefill_and_decode():
    resp = {"timings": {"prompt_n": 12, "prompt_ms": 30.0,
                        "predicted_n": 8, "predicted_ms": 40.0}}
    t = parse_timings(resp)
    assert t.prompt_n == 12
    assert t.prompt_ms == 30.0
    assert t.predicted_n == 8
    assert t.predicted_ms == 40.0

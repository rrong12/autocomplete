import statistics


def summarize(samples: list[float]) -> dict:
    s = sorted(samples)
    if not s:
        return {"n": 0, "median": 0.0, "p90": 0.0, "mean": 0.0}
    idx = min(len(s) - 1, int(0.9 * len(s)))
    return {"n": len(s), "median": statistics.median(s),
            "p90": s[idx], "mean": statistics.fmean(s)}

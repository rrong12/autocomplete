import re


def next_word(text: str) -> str:
    m = re.match(r"\s*([A-Za-z0-9']+)", text)
    return m.group(1) if m else ""


def next_word_match(suggestion: str, truth: str) -> bool:
    t = next_word(truth)
    return t != "" and next_word(suggestion).lower() == t.lower()


def _tokens(s: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", s.lower())


def prefix_overlap_tokens(suggestion: str, truth: str) -> int:
    n = 0
    for a, b in zip(_tokens(suggestion), _tokens(truth)):
        if a != b:
            break
        n += 1
    return n


def truncate_at_boundary(text: str) -> str:
    best = None
    for end in (". ", "! ", "? ", "\n"):
        i = text.find(end)
        if i != -1:
            best = i + 1 if best is None else min(best, i + 1)
    return text[:best].rstrip() if best is not None else text.rstrip()


def precision_coverage(records, thresholds):
    """records: list of (confidence, correct). confidence = min token logprob of the
    suggestion. Returns [(threshold, coverage, precision)] — raising the bar shows
    fewer suggestions (lower coverage) but more of them are right (higher precision)."""
    total = len(records)
    out = []
    for thr in thresholds:
        shown = [c for c in records if c[0] >= thr]
        cov = len(shown) / total if total else 0.0
        prec = sum(1 for c in shown if c[1]) / len(shown) if shown else 0.0
        out.append((thr, cov, prec))
    return out

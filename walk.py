def cursor_positions(text: str, n_points: int) -> list[int]:
    """Character offsets at interior word boundaries, evenly spaced. Each offset is
    a point where we cut the text into (prefix, held-out continuation)."""
    spaces = [i for i, c in enumerate(text) if c == " "]
    interior = [s for s in spaces if 0 < s < len(text) - 1]
    if n_points <= 0 or not interior:
        return []
    if n_points >= len(interior):
        return interior
    step = len(interior) / n_points
    return [interior[int(i * step)] for i in range(n_points)]


def held_out(text: str, pos: int) -> str:
    """The true continuation the model is trying to predict."""
    return text[pos:]

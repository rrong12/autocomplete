from dataclasses import dataclass, field

from client import complete
from walk import cursor_positions, held_out


@dataclass
class Record:
    position: int
    prefix_len: int
    suggestion: str
    truth: str
    ttft_ms: list = field(default_factory=list)
    prefill_ms: list = field(default_factory=list)
    decode_ms: list = field(default_factory=list)
    total_ms: list = field(default_factory=list)
    min_logprob: float = 0.0


def run_walk(passage: str, n_points=8, repetitions=5, warmup=1, **params) -> list[Record]:
    records = []
    for pos in cursor_positions(passage, n_points):
        prefix = passage[:pos]
        truth = held_out(passage, pos)
        rec = Record(pos, len(prefix), "", truth)
        for i in range(warmup + repetitions):
            c = complete(prefix, **params)
            if i < warmup:               # discard warm-up
                continue
            rec.suggestion = c.text
            rec.ttft_ms.append(c.ttft_ms)
            rec.prefill_ms.append(c.timings.prompt_ms)
            rec.decode_ms.append(c.timings.predicted_ms)
            rec.total_ms.append(c.timings.prompt_ms + c.timings.predicted_ms)
            if c.token_logprobs:
                rec.min_logprob = min(c.token_logprobs)
        records.append(rec)
    return records

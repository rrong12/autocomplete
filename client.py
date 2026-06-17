import json
import time
from dataclasses import dataclass, field

import requests

from config import SERVER


@dataclass
class Timings:
    prompt_n: int
    prompt_ms: float
    predicted_n: int
    predicted_ms: float


@dataclass
class Completion:
    text: str
    timings: Timings
    ttft_ms: float
    token_logprobs: list = field(default_factory=list)


def parse_timings(resp: dict) -> Timings:
    t = resp["timings"]
    return Timings(t["prompt_n"], t["prompt_ms"], t["predicted_n"], t["predicted_ms"])


def complete(prefix, *, n_predict=16, temperature=0.0, top_p=0.95, top_k=40,
             cache_prompt=True, stop=None, n_probs=1, server=SERVER) -> Completion:
    payload = {"prompt": prefix, "n_predict": n_predict, "temperature": temperature,
               "top_p": top_p, "top_k": top_k, "cache_prompt": cache_prompt,
               "stop": stop if stop is not None else ["\n"], "n_probs": n_probs,
               "stream": True}
    parts, logprobs, ttft, final = [], [], None, None
    start = time.perf_counter()
    with requests.post(f"{server}/completion", json=payload, stream=True) as r:
        for raw in r.iter_lines():
            if not raw:
                continue
            line = raw.decode()
            if not line.startswith("data: "):
                continue
            chunk = json.loads(line[6:])
            if chunk.get("content"):
                if ttft is None:
                    ttft = (time.perf_counter() - start) * 1000.0
                parts.append(chunk["content"])
            # NOTE: logprob field names vary by llama.cpp version — verified against
            # the running server during integration (Task 2 Step 5).
            for cp in chunk.get("completion_probabilities", []):
                if "logprob" in cp:
                    logprobs.append(cp["logprob"])
            if chunk.get("stop"):
                final = chunk
    return Completion("".join(parts), parse_timings(final), ttft or 0.0, logprobs)

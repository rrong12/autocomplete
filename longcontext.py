"""Supplement for Q2: the brief's short passage can't exercise KV-cache reuse
(prefill is cheap at ~70 tokens). Here we tile the real article into a long
context and measure, at several prefix lengths:
  - COLD prefill  (cache_prompt=false): full reprocess, scales with length
  - INCREMENTAL prefill (cache_prompt=true, +a few new tokens): only the new
    tokens are processed -> ~constant, independent of document length.
That gap IS the KV-cache lever. Needs the server started with a large enough
--ctx-size (we use 4096). prefill latency depends only on token COUNT, so tiling
real text is a valid way to reach long contexts.
"""
import csv
import os

from client import complete
from passages import PASSAGES
from stats import summarize

RESULTS = os.path.join(os.path.dirname(__file__), "results")
REPS = 5


def build_long(copies=8):
    return "\n\n".join([PASSAGES["article"]] * copies)


def cold(prefix):
    c = complete(prefix, n_predict=1, cache_prompt=False, temperature=0.0,
                 stop=[], n_probs=0)
    return c.timings.prompt_n, c.timings.prompt_ms


def incremental(prefix, addition=" the team agreed to"):
    complete(prefix, n_predict=1, cache_prompt=True, temperature=0.0, stop=[], n_probs=0)
    c = complete(prefix + addition, n_predict=1, cache_prompt=True, temperature=0.0,
                 stop=[], n_probs=0)
    return c.timings.prompt_n, c.timings.prompt_ms


def main():
    text = build_long()
    complete(text[:2000], n_predict=4, cache_prompt=False, temperature=0.0,
             stop=[], n_probs=0)   # warm-up: compile Metal kernels, discard
    rows = []
    for fr in (0.15, 0.3, 0.45, 0.6, 0.75, 0.9):
        cut = text.rfind(" ", 0, int(len(text) * fr))
        prefix = text[:cut]
        n_cold = [cold(prefix) for _ in range(REPS)]
        n_inc = [incremental(prefix) for _ in range(REPS)]
        ntok = n_cold[0][0]
        rows.append([ntok,
                     round(summarize([x[1] for x in n_cold])["median"], 1),
                     round(summarize([x[1] for x in n_inc])["median"], 1)])
        print(f"  {ntok:5d} tok: cold {rows[-1][1]:8.1f} ms | incremental {rows[-1][2]:6.1f} ms")
    with open(os.path.join(RESULTS, "longcontext.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["prompt_tokens", "cold_prefill_ms", "incremental_prefill_ms"])
        w.writerows(rows)
    print("wrote results/longcontext.csv")


if __name__ == "__main__":
    main()

"""Controlled experiments for the autocomplete loop. Each function runs walks under
one condition and writes a CSV into results/. Latency uses median over repetitions
(warm-up discarded). Run with the llama-server already up (see run_server.sh)."""
import csv
import os

from config import DEFAULT, GRANULARITIES
from harness import run_walk
from metrics import (next_word_match, prefix_overlap_tokens,
                     precision_coverage, truncate_at_boundary)
from passages import PASSAGES
from stats import summarize

RESULTS = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS, exist_ok=True)

N_POINTS = 10
REPS = 5
WARMUP = 1


def _write(name, header, rows):
    with open(os.path.join(RESULTS, f"{name}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  wrote results/{name}.csv ({len(rows)} rows)")


def exp_cache(passage="article"):
    """KV-cache reuse on vs off: prefill + TTFT vs prefix length."""
    rows = []
    for on in (True, False):
        p = {**DEFAULT, "cache_prompt": on}
        for r in run_walk(PASSAGES[passage], N_POINTS, REPS, WARMUP, **p):
            rows.append([on, r.prefix_len,
                         round(summarize(r.prefill_ms)["median"], 1),
                         round(summarize(r.ttft_ms)["median"], 1)])
    _write("cache", ["cache_on", "prefix_len", "prefill_ms_median", "ttft_ms_median"], rows)


def exp_granularity(passage="article"):
    """word / phrase_sentence / multiline: latency + example suggestion per setting."""
    rows = []
    for name, npred in GRANULARITIES.items():
        p = {**DEFAULT, "n_predict": npred}
        for r in run_walk(PASSAGES[passage], N_POINTS, REPS, WARMUP, **p):
            s = summarize(r.total_ms)
            rows.append([name, npred, r.prefix_len, round(s["median"], 1),
                         round(s["p90"], 1), r.suggestion.strip()])
    _write("granularity", ["granularity", "n_predict", "prefix_len",
                           "total_ms_median", "total_ms_p90", "suggestion"], rows)


def exp_decoding(passage="article"):
    """greedy vs sampled: latency similar, suggestion character differs."""
    rows = []
    for label, extra in [("greedy", {"temperature": 0.0}),
                         ("sampled", {"temperature": 0.7, "top_p": 0.9})]:
        p = {**DEFAULT, **extra}
        for r in run_walk(PASSAGES[passage], N_POINTS, REPS, WARMUP, **p):
            rows.append([label, r.prefix_len,
                         round(summarize(r.total_ms)["median"], 1), r.suggestion.strip()])
    _write("decoding", ["decoding", "prefix_len", "total_ms_median", "suggestion"], rows)


def exp_accuracy():
    """Held-out next-word match + prefix overlap across all passages, plus the
    confidence (min-logprob) for each suggestion -> precision/coverage curve."""
    rows, conf = [], []
    for pname, text in PASSAGES.items():
        for r in run_walk(text, N_POINTS, 3, WARMUP, **DEFAULT):
            correct = next_word_match(r.suggestion, r.truth)
            rows.append([pname, r.prefix_len, int(correct),
                         prefix_overlap_tokens(r.suggestion, r.truth),
                         round(r.min_logprob, 3),
                         truncate_at_boundary(r.suggestion).strip(),
                         r.truth[:40].strip()])
            conf.append((r.min_logprob, correct))
    _write("accuracy", ["passage", "prefix_len", "next_word_match", "prefix_overlap",
                        "min_logprob", "suggestion", "truth_head"], rows)
    thresholds = [-6, -5, -4, -3, -2.5, -2, -1.5, -1, -0.5]
    pc = precision_coverage(conf, thresholds)
    _write("precision_coverage", ["min_logprob_threshold", "coverage", "precision"],
           [[t, round(c, 3), round(p, 3)] for t, c, p in pc])


def exp_latency_profile(label, passage="article"):
    """Full latency breakdown for whatever model the server is currently running.
    Run once per quant (restart the server between) -> latency_q4.csv, latency_q8.csv."""
    rows = []
    for r in run_walk(PASSAGES[passage], N_POINTS, REPS, WARMUP, **DEFAULT):
        rows.append([label, r.prefix_len,
                     round(summarize(r.ttft_ms)["median"], 1),
                     round(summarize(r.prefill_ms)["median"], 1),
                     round(summarize(r.decode_ms)["median"], 1),
                     round(summarize(r.total_ms)["median"], 1)])
    _write(f"latency_{label}", ["model", "prefix_len", "ttft_ms",
                                "prefill_ms", "decode_ms", "total_ms"], rows)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "latency":
        exp_latency_profile(sys.argv[2])          # e.g. python experiments.py latency q8
    else:
        print("running Q4 experiment suite:")
        exp_cache()
        exp_granularity()
        exp_decoding()
        exp_accuracy()
        exp_latency_profile("q4")
        print("done.")

# Local Autocomplete with llama.cpp — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python harness that drives a local `llama-server` (base `gemma-4-E2B`) through a growing-prefix autocomplete loop, and measure both latency and suggestion quality honestly.

**Architecture:** A thin client wraps `llama-server`'s native `/completion` endpoint (streaming, server-side timings, logprobs). Pure modules handle cursor-walking, accuracy metrics, boundary truncation, and stats and are unit-tested. An orchestration layer runs the walk and the controlled experiments (KV-cache on/off, granularity sweep, Q4-vs-Q8, decoding); an analysis layer emits tables and charts. A write-up answers the five take-home questions, every claim tied to a measured number.

**Tech Stack:** Python 3 (`requests`, `matplotlib`, stdlib `statistics`/`re`), `pytest`; llama.cpp (`llama-server`) on Apple M1 Metal; GGUF models (Q4_K_M primary, Q8_0 for comparison).

**Spec:** `docs/superpowers/specs/2026-06-15-local-autocomplete-design.md`

**Verification philosophy:** Pure functions → red-green TDD (full test + impl shown). I/O and measurement → run-and-observe (run the command, confirm the stated observation). Each task says which it uses.

---

## File structure (decomposition locked here)

| File | Responsibility | Tested how |
|------|----------------|------------|
| `run_server.sh` | Launch `llama-server` with a model path + Metal flags | run-and-observe |
| `requirements.txt` | Python deps | — |
| `config.py` | Server URL + default sampling/params constants | — |
| `passages.py` | 2-3 realistic passages as `PASSAGES: dict[str,str]` | — |
| `client.py` | `parse_timings()` (pure) + `complete()` (I/O) → suggestion, timings, TTFT, logprobs | TDD + run-and-observe |
| `walk.py` | `cursor_positions()`, `held_out()` — choose growing-prefix cut points | TDD |
| `metrics.py` | `next_word()`, `next_word_match()`, `prefix_overlap_tokens()`, `truncate_at_boundary()`, `precision_coverage()` | TDD |
| `stats.py` | `summarize()` — median/p90/mean over repetitions | TDD |
| `harness.py` | `run_walk()` — drive the loop, collect `Record`s with warm-up + repetitions | run-and-observe |
| `experiments.py` | The controlled comparisons; writes CSVs into `results/` | run-and-observe |
| `analyze.py` | Aggregate → markdown tables + matplotlib charts into `results/` | run-and-observe |
| `README.md` | Exact run instructions | — |
| `WRITEUP.md` | The 5 answers, citing measured numbers | — |

`tests/` holds `test_walk.py`, `test_metrics.py`, `test_stats.py`, `test_client.py`.

---

## Task 0: Environment + model acquisition (DE-RISK FIRST)

**This task resolves the project's biggest unknown: does a base `gemma-4-E2B` GGUF exist and run?** If it does not, STOP and surface to the user before building further.

**Files:** Create `run_server.sh`

- [ ] **Step 1: Install llama.cpp** — `brew install llama.cpp` then `llama-server --version`. Expected: a version prints (binary on PATH).

- [ ] **Step 2: Locate the GGUF.** Search Hugging Face for a GGUF build of base `gemma-4-E2B` (e.g. via `huggingface-cli` or the HF web search). Record the resolved repo + filename in `README.md`. **Decision gate:** if only an `-it` GGUF exists and no base, STOP and tell the user — we decide together (convert, or proceed with what exists, documented honestly). Do not silently substitute.

- [ ] **Step 3: Download the Q4 model** into `models/` (gitignored). Prefer Q4_K_M (~3 GB — fits 8 GB RAM with headroom). Confirm the file size with `ls -lh models/`.

- [ ] **Step 4: Write `run_server.sh`:**

```bash
#!/usr/bin/env bash
# Launch llama-server for the autocomplete harness.
# Usage: ./run_server.sh <path-to-gguf> [port]
set -euo pipefail
MODEL="${1:?path to .gguf required}"
PORT="${2:-8080}"
exec llama-server \
  --model "$MODEL" \
  --port "$PORT" \
  --ctx-size 2048 \
  --n-gpu-layers 999 \
  --parallel 1 \
  --no-webui
```

- [ ] **Step 5: Smoke-test the server (run-and-observe).** In one terminal: `./run_server.sh models/<file>.gguf`. In another:

```bash
curl -s http://127.0.0.1:8080/completion \
  -d '{"prompt":"The quick brown","n_predict":8,"cache_prompt":true}' | python3 -m json.tool
```

Expected: JSON containing a `content` continuation AND a `timings` object with `prompt_ms` and `predicted_ms`. **If `timings` is absent, the rest of the plan's measurement depends on it — note the server version and adjust.**

- [ ] **Step 6: Commit** `git add run_server.sh README.md && git commit -m "chore: llama.cpp setup + model acquisition, server smoke test"`

---

## Task 1: Python scaffold

**Files:** Create `requirements.txt`, `config.py`, `passages.py`

- [ ] **Step 1: `requirements.txt`:**

```
requests
matplotlib
pytest
```

- [ ] **Step 2: venv + install** — `python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt`. Expected: installs cleanly.

- [ ] **Step 3: `config.py`:**

```python
SERVER = "http://127.0.0.1:8080"

# Phrase-to-sentence default (the chosen granularity); other granularities live
# in the experiments sweep, not here.
DEFAULT = dict(n_predict=16, temperature=0.0, top_p=0.95, top_k=40,
               cache_prompt=True, stop=["\n", ". ", "! ", "? "], n_probs=1)

GRANULARITIES = {"word": 3, "phrase_sentence": 16, "multiline": 64}
```

- [ ] **Step 4: `passages.py`** — 2-3 realistic passages (prose paragraph + short email + one more). Real text, several sentences each:

```python
PASSAGES = {
    "prose": (
        "The morning fog had not yet lifted when she stepped onto the platform. "
        "The train was late again, and the small crowd shuffled in the cold, "
        "checking phones and exchanging the resigned glances of people who commute "
        "together but have never spoken. She pulled her coat tighter and watched "
        "the empty tracks curve away into the grey."
    ),
    "email": (
        "Hi Marcus, thanks for sending over the draft proposal yesterday. "
        "I read through it on the train this morning and overall it looks strong. "
        "I have a couple of small concerns about the timeline in section three, "
        "and I think we should add a paragraph about budget before we send it to the client. "
        "Could we find thirty minutes tomorrow to talk it through?"
    ),
    "notes": (
        "Meeting notes: the team agreed to ship the beta by the end of the month. "
        "Remaining blockers are the login flow and the slow search endpoint. "
        "Priya will own the login fix and Sam will profile the database queries. "
        "We decided to postpone the redesign until after launch."
    ),
}
```

- [ ] **Step 5: Commit** `git add requirements.txt config.py passages.py && git commit -m "feat: python scaffold, config, sample passages"`

---

## Task 2: client.py — talk to llama-server

**Files:** Create `client.py`, `tests/test_client.py`

- [ ] **Step 1: Write the failing test (pure parser):** `tests/test_client.py`

```python
from client import parse_timings

def test_parse_timings_extracts_prefill_and_decode():
    resp = {"timings": {"prompt_n": 12, "prompt_ms": 30.0,
                        "predicted_n": 8, "predicted_ms": 40.0}}
    t = parse_timings(resp)
    assert t.prompt_n == 12
    assert t.prompt_ms == 30.0
    assert t.predicted_n == 8
    assert t.predicted_ms == 40.0
```

- [ ] **Step 2: Run it, verify it fails** — `pytest tests/test_client.py -v`. Expected: FAIL (cannot import `parse_timings`).

- [ ] **Step 3: Implement `client.py`:**

```python
import json, time
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
            # NOTE: logprob field names vary by llama.cpp version — verify against
            # the installed server in Step 5 and adjust the key if needed.
            for cp in chunk.get("completion_probabilities", []):
                if "logprob" in cp:
                    logprobs.append(cp["logprob"])
            if chunk.get("stop"):
                final = chunk
    return Completion("".join(parts), parse_timings(final), ttft or 0.0, logprobs)
```

- [ ] **Step 4: Run the test, verify it passes** — `pytest tests/test_client.py -v`. Expected: PASS.

- [ ] **Step 5: Integration smoke (run-and-observe).** With the server running, in a Python shell: `from client import complete; c = complete("The quick brown", n_predict=8); print(repr(c.text), c.ttft_ms, c.timings)`. Expected: a continuation string, a positive `ttft_ms`, and a `Timings` with non-zero `prompt_ms`/`predicted_ms`. **If `token_logprobs` is empty, fix the field name per the Step 3 note and re-run.**

- [ ] **Step 6: Commit** `git add client.py tests/test_client.py && git commit -m "feat: llama-server client with timings + streaming TTFT"`

---

## Task 3: walk.py — growing-prefix cursor positions

**Files:** Create `walk.py`, `tests/test_walk.py`

- [ ] **Step 1: Failing test:** `tests/test_walk.py`

```python
from walk import cursor_positions, held_out

def test_positions_are_interior_word_boundaries():
    text = "the quick brown fox jumps over the lazy dog today"
    pos = cursor_positions(text, 3)
    assert len(pos) == 3
    assert all(text[p] == " " for p in pos)      # at a space
    assert all(0 < p < len(text) - 1 for p in pos)  # interior, room to continue
    assert pos == sorted(pos)                      # increasing => growing prefix

def test_held_out_is_text_after_cursor():
    text = "hello world foo"
    p = text.index(" ")            # after "hello"
    assert held_out(text, p).startswith(" world")
```

- [ ] **Step 2: Run, verify fail** — `pytest tests/test_walk.py -v`. Expected: FAIL (import error).

- [ ] **Step 3: Implement `walk.py`:**

```python
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
```

- [ ] **Step 4: Run, verify pass** — `pytest tests/test_walk.py -v`. Expected: PASS.

- [ ] **Step 5: Commit** `git add walk.py tests/test_walk.py && git commit -m "feat: growing-prefix cursor walk"`

---

## Task 4: metrics.py — accuracy + truncation

**Files:** Create `metrics.py`, `tests/test_metrics.py` (the `precision_coverage` function is added in Task 7.)

- [ ] **Step 1: Failing test:** `tests/test_metrics.py`

```python
from metrics import next_word, next_word_match, prefix_overlap_tokens, truncate_at_boundary

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
```

- [ ] **Step 2: Run, verify fail** — `pytest tests/test_metrics.py -v`. Expected: FAIL.

- [ ] **Step 3: Implement `metrics.py`:**

```python
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
```

- [ ] **Step 4: Run, verify pass** — `pytest tests/test_metrics.py -v`. Expected: PASS.

- [ ] **Step 5: Commit** `git add metrics.py tests/test_metrics.py && git commit -m "feat: accuracy metrics + boundary truncation"`

---

## Task 5: stats.py — median/p90 over repetitions

**Files:** Create `stats.py`, `tests/test_stats.py`

- [ ] **Step 1: Failing test:** `tests/test_stats.py`

```python
from stats import summarize

def test_summarize_reports_median_p90_mean():
    s = summarize([10, 20, 30, 40, 100])
    assert s["n"] == 5
    assert s["median"] == 30
    assert s["p90"] == 100        # 90th percentile index into sorted samples
    assert round(s["mean"], 1) == 40.0

def test_summarize_empty_is_safe():
    s = summarize([])
    assert s["n"] == 0 and s["median"] == 0.0
```

- [ ] **Step 2: Run, verify fail** — `pytest tests/test_stats.py -v`. Expected: FAIL.

- [ ] **Step 3: Implement `stats.py`:**

```python
import statistics

def summarize(samples: list[float]) -> dict:
    s = sorted(samples)
    if not s:
        return {"n": 0, "median": 0.0, "p90": 0.0, "mean": 0.0}
    idx = min(len(s) - 1, int(0.9 * len(s)))
    return {"n": len(s), "median": statistics.median(s),
            "p90": s[idx], "mean": statistics.fmean(s)}
```

- [ ] **Step 4: Run, verify pass** — `pytest tests/test_stats.py -v`. Expected: PASS.

- [ ] **Step 5: Commit** `git add stats.py tests/test_stats.py && git commit -m "feat: latency stats (median/p90/mean)"`

---

## Task 6: harness.py — drive the walk (run-and-observe)

**Files:** Create `harness.py`

- [ ] **Step 1: Implement `harness.py`:**

```python
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
```

- [ ] **Step 2: Run-and-observe.** With server up: `python3 -c "from harness import run_walk; from config import DEFAULT; from passages import PASSAGES; rs=run_walk(PASSAGES['prose'], n_points=4, repetitions=3, **DEFAULT); [print(r.prefix_len, repr(r.suggestion[:40]), sorted(r.total_ms)) for r in rs]"`. Expected: 4 rows, each with a plausible continuation and 3 latency samples. Eyeball that suggestions are coherent continuations.

- [ ] **Step 3: Commit** `git add harness.py && git commit -m "feat: walk harness with warm-up + repetitions"`

---

## Task 7: experiments.py + precision/coverage metric

**Files:** Create `experiments.py`; modify `metrics.py` (add `precision_coverage`) and `tests/test_metrics.py`

- [ ] **Step 1: Failing test for `precision_coverage`:** add to `tests/test_metrics.py`

```python
from metrics import precision_coverage

def test_precision_coverage_trades_coverage_for_precision():
    # (confidence, correct)
    recs = [(-0.1, True), (-0.5, True), (-2.0, False), (-3.0, False)]
    rows = precision_coverage(recs, thresholds=[-5.0, -1.0])
    # threshold -5 shows all 4 (coverage 1.0, precision 0.5)
    assert rows[0] == (-5.0, 1.0, 0.5)
    # threshold -1 shows the two confident ones (coverage 0.5, precision 1.0)
    assert rows[1] == (-1.0, 0.5, 1.0)
```

- [ ] **Step 2: Run, verify fail** — `pytest tests/test_metrics.py::test_precision_coverage_trades_coverage_for_precision -v`. Expected: FAIL.

- [ ] **Step 3: Add `precision_coverage` to `metrics.py`:**

```python
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
```

- [ ] **Step 4: Run, verify pass** — `pytest tests/test_metrics.py -v`. Expected: PASS (all metrics tests).

- [ ] **Step 5: Implement `experiments.py`** — each function runs a walk under one condition and writes a CSV into `results/`. Concrete structure:

```python
import csv, os
from harness import run_walk
from stats import summarize
from config import DEFAULT, GRANULARITIES
from passages import PASSAGES

os.makedirs("results", exist_ok=True)

def _write(name, rows, header):
    with open(f"results/{name}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)

def exp_cache(passage="prose"):
    """KV-cache reuse on vs off: prefill time vs prefix length."""
    rows = []
    for on in (True, False):
        p = {**DEFAULT, "cache_prompt": on}
        for r in run_walk(PASSAGES[passage], **p):
            rows.append([on, r.prefix_len, summarize(r.prefill_ms)["median"],
                         summarize(r.ttft_ms)["median"]])
    _write("cache", rows, ["cache_on", "prefix_len", "prefill_ms_median", "ttft_ms_median"])

def exp_granularity(passage="prose"):
    """word / phrase_sentence / multiline: latency vs how much we predict."""
    rows = []
    for name, npred in GRANULARITIES.items():
        p = {**DEFAULT, "n_predict": npred}
        for r in run_walk(PASSAGES[passage], **p):
            s = summarize(r.total_ms)
            rows.append([name, npred, r.prefix_len, s["median"], s["p90"], repr(r.suggestion)])
    _write("granularity", rows,
           ["granularity", "n_predict", "prefix_len", "total_ms_median", "total_ms_p90", "suggestion"])

def exp_decoding(passage="prose"):
    """greedy vs sampled: latency similar, suggestion character differs."""
    rows = []
    for label, extra in [("greedy", {"temperature": 0.0}),
                         ("sampled", {"temperature": 0.7, "top_p": 0.9})]:
        p = {**DEFAULT, **extra}
        for r in run_walk(PASSAGES[passage], **p):
            rows.append([label, r.prefix_len, summarize(r.total_ms)["median"], repr(r.suggestion)])
    _write("decoding", rows, ["decoding", "prefix_len", "total_ms_median", "suggestion"])

if __name__ == "__main__":
    exp_cache(); exp_granularity(); exp_decoding()
    print("wrote results/cache.csv, results/granularity.csv, results/decoding.csv")
```

- [ ] **Step 6: Run-and-observe the in-RAM experiments** — `python3 experiments.py`. Expected: three CSVs in `results/`. Open `results/cache.csv` and confirm prefill_ms grows with prefix_len when `cache_on=False` but stays roughly flat when `True` — the headline result.

- [ ] **Step 7: Quant Q4-vs-Q8 (manual, download-test-delete).** Run `exp_cache`/granularity once with the server on Q4 (rename outputs to `*_q4.csv`), then: stop server → download Q8_0 → start server on Q8 → re-run → save as `*_q8.csv` → **delete the Q8 file** to reclaim disk. **If the machine swaps badly on Q8 (watch Activity Monitor / `vm_stat`), record "Q8 caused memory pressure on 8 GB" as the finding** and move on — do not report swap-contaminated latency as clean.

- [ ] **Step 8: Model head-to-head (manual).** Download the `-it` Q4 GGUF, start the server on it, run `python3 -c "from client import complete; [print(p, '=>', repr(complete(p, n_predict=16).text)) for p in ['The quick brown', 'Hi Marcus, thanks for', 'Meeting notes: the team']]"`, save the base-vs-`-it` outputs side by side into `results/model_headtohead.md`, then **delete the `-it` file**.

- [ ] **Step 9: Commit** `git add experiments.py metrics.py tests/test_metrics.py results/ && git commit -m "feat: latency/accuracy experiments + precision-coverage metric"`

---

## Task 8: analyze.py — tables + charts (run-and-observe)

**Files:** Create `analyze.py`

- [ ] **Step 1: Implement `analyze.py`** — read the CSVs, print markdown summary tables, and render the key chart (prefill vs prefix length, cache on/off):

```python
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _read(name):
    with open(f"results/{name}.csv") as f:
        return list(csv.DictReader(f))

def chart_cache():
    rows = _read("cache")
    fig, ax = plt.subplots()
    for on in ("True", "False"):
        pts = [(int(r["prefix_len"]), float(r["prefill_ms_median"]))
               for r in rows if r["cache_on"] == on]
        pts.sort()
        ax.plot([x for x, _ in pts], [y for _, y in pts],
                marker="o", label=f"cache {'on' if on=='True' else 'off'}")
    ax.set_xlabel("prefix length (chars)"); ax.set_ylabel("prefill ms (median)")
    ax.set_title("KV-cache reuse: prefill vs prefix length"); ax.legend()
    fig.savefig("results/cache.png", dpi=120, bbox_inches="tight")
    print("wrote results/cache.png")

def table_granularity():
    print("\n| granularity | n_predict | median ms | p90 ms |")
    print("|---|---|---|---|")
    seen = {}
    for r in _read("granularity"):
        seen.setdefault(r["granularity"], (r["n_predict"], [], []))
        seen[r["granularity"]][1].append(float(r["total_ms_median"]))
        seen[r["granularity"]][2].append(float(r["total_ms_p90"]))
    for g, (npred, med, p90) in seen.items():
        print(f"| {g} | {npred} | {sum(med)/len(med):.0f} | {max(p90):.0f} |")

if __name__ == "__main__":
    chart_cache(); table_granularity()
```

- [ ] **Step 2: Run-and-observe** — `python3 analyze.py`. Expected: `results/cache.png` exists and shows the off-curve rising while the on-curve stays low; the granularity table prints with latency increasing word → phrase → multiline.

- [ ] **Step 3: Commit** `git add analyze.py results/ && git commit -m "feat: analysis tables + cache chart"`

---

## Task 9: WRITEUP.md + README.md

**Files:** Create `WRITEUP.md`, update `README.md`

- [ ] **Step 1: `README.md`** — exact run order: install llama.cpp, model repo+file used, `./run_server.sh models/<file>`, `pip install -r requirements.txt`, `python3 experiments.py`, `python3 analyze.py`, `pytest`. State the machine (MacBook Air M1, 8 GB).

- [ ] **Step 2: `WRITEUP.md`** — answer the 5 questions in Robert's voice, each claim citing a number from `results/` or a shown example:
  1. **Model choice** — base over `-it`; cite `results/model_headtohead.md`.
  2. **Latency** — prefill vs decode vs TTFT; cite `results/cache.png` and the median/p90 numbers; explain KV-cache reuse and the granularity/output-cap effect.
  3. **Accuracy** — held-out next-word rate + examples (state the proxy's bias); the precision/coverage confidence-gating result; decoding levers (temp/top-p/top-k) and their latency cost.
  4. **Middle-of-text** — show the causal model ignoring the suffix; explain FIM and name FIM-capable models.
  5. **Production** — debounce, cancellation, warm-up, persistent KV, speculative decoding, telemetry on acceptance, memory lifecycle on 8 GB, privacy.

- [ ] **Step 3: Final full-suite check** — `pytest -v` (all pure-logic tests pass) and confirm every `results/` artifact referenced in `WRITEUP.md` exists.

- [ ] **Step 4: Commit** `git add WRITEUP.md README.md && git commit -m "docs: write-up answering the five questions + run instructions"`

---

## Self-review (completed)

- **Spec coverage:** §4 surface → Task 0/2; §5 model + head-to-head → Task 0/7.8; §7 core loop → Task 3/6; §8.1 cache → Task 7 `exp_cache`; §8.2 granularity → Task 7 `exp_granularity`; §8.3 quant → Task 7.7; §9 accuracy (proxy, precision/coverage, truncation) → Task 4/7; §10 decoding → Task 7 `exp_decoding`; §11 middle-of-text + §12 production → Task 9 write-up. Speculative decoding (§8.5) is stretch → covered in write-up Q5 only, consistent with spec.
- **Placeholder scan:** model repo/filename is an unavoidable discovery (Task 0 Step 2, gated), not a lazy placeholder. Logprob field name flagged as version-dependent with a concrete verify step. No other gaps.
- **Type/name consistency:** `complete()`/`Completion`/`Timings`/`Record`/`cursor_positions`/`held_out`/`summarize`/`precision_coverage` names match across all tasks.

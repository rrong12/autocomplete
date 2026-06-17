# Local Autocomplete with llama.cpp — Design Spec

**Date:** 2026-06-15
**Author:** Robert (with Claude as pair)
**Status:** Approved design, pre-implementation

This document is both the design and a teaching artifact. It is written so that the
reasoning behind every decision is explicit — the take-home grades *thinking about
latency, accuracy, and measurement*, so the "why" matters as much as the "what".

---

## 1. What we are building

An inline autocomplete loop, the way Gmail Smart Compose or Cursor Tab works: given the
text a user has typed so far (the **prefix**), predict a good continuation. The model is a
small LLM running **locally** via llama.cpp — no network, for speed and privacy.

Two things matter equally:
- **Latency** — the suggestion must feel instant while typing. Late = useless.
- **Accuracy** — the suggestion must be a genuinely good continuation. Wrong = noise.

We are not building a UI and not simulating keystrokes. We build a **harness**: a driver
that walks a cursor through a realistic passage, and at a series of points feeds the
growing prefix to the model and records (a) how long it took and (b) how good the
suggestion was.

## 2. Mental model (the concepts the whole project rests on)

**Autocomplete = text continuation.** Feed the model "The quick brown" and it predicts
"fox". That is the entire task.

**Inference has two phases:**
- **Prefill** — the model reads the whole prefix to build its internal state (the KV
  cache). Cost scales with prefix length. Parallelizable, but still the dominant cost for
  long prefixes.
- **Decode** — the model emits tokens one at a time, each depending on the last. Sequential
  and unavoidably serial.

**TTFT (time to first token)** is the metric that maps to "feels instant." It is dominated
by prefill. Total latency = TTFT + decode time for the rest of the suggestion.

**KV-cache reuse is the biggest latency lever.** Each prefix is the previous prefix plus a
few new characters. Instead of re-reading the entire prefix every keystroke, we cache the
model's state and process only the *new* tokens. With reuse, prefill cost stops growing
with document length. This is the headline result we will measure.

**Decoding turns probabilities into tokens.** The model outputs a probability distribution
over the next token; the *decoding strategy* picks the actual token. Greedy (argmax) is
fast and high-confidence. Sampling (temperature/top-p/top-k) adds variety at the risk of
noise. For autocomplete we generally want low temperature and **early stopping** (stop at a
clean boundary so it doesn't ramble).

## 3. Environment and constraints

- **Machine:** MacBook Air, Apple M1 (4 performance + 4 efficiency cores), **8 GB RAM**,
  macOS 14.2.1. Metal GPU available via llama.cpp.
- **Disk:** ~11 GB free. Be economical with model downloads.

**Why the 8 GB constraint is a feature, not just a limitation:** on-device autocomplete
must coexist with the user's actual app, browser, and OS in consumer RAM. This makes the
**quantization** choice a real product decision: a Q4 quant (~3 GB) is the realistic ship
target; a Q8 quant (~5 GB) is borderline on 8 GB and may swap. We will measure both where
feasible and report honestly if Q8 is too tight to run cleanly — that tradeoff *is* part of
the answer.

## 4. llama.cpp surface

We drive **`llama-server`** (local HTTP) via its native `/completion` endpoint. Rationale:
- It mirrors how this would actually ship (a local service the app calls).
- It returns server-side **`timings`** (`prompt_ms` = prefill, `predicted_ms` = decode,
  tokens/sec) — precise measurement without guessing.
- It supports **`cache_prompt`** (our KV-reuse on/off knob) and **`n_probs`** (per-token
  logprobs, for the confidence-gating analysis).
- Streaming gives a clean client-side **TTFT** measurement.

Rejected alternatives: `llama-cli` subprocess (re-processes the prefix every call — kills
the latency story); in-process `llama-cpp-python` (more control but more code and fiddly
Metal install — not worth it here).

## 5. Model

**Base `gemma-4-E2B`**, not the `-it` (instruction-tuned) variant.

- The **base** model is trained to *continue* text — exactly the autocomplete task.
- The **`-it`** model is trained to follow chat instructions; fed raw text it tends to
  "respond to" or "complete a task," not naturally continue. It also needs a chat template
  wrapped around the input.

**Decision (confirmed 2026-06-15):** run a **lightweight head-to-head** — both models on
~6 shared prefixes, outputs side by side — so the difference is *visible* (the evidence for
write-up Q1). This is rigor proportional to a near-foregone decision: enough to show we
checked, not a wasteful full benchmark. The `-it` model is downloaded only for this, then
deleted (disk is tight at ~11 GB free).

> **Setup risk (honest):** we must confirm a downloadable GGUF of the base variant (and a
> second quant) actually exists. Verified at setup; if the exact name differs in practice,
> we surface it and decide together rather than papering over it.

## 6. Architecture

```
local-autocomplete/
  run_server.sh    # launches llama-server with model + Metal flags (documents HOW we ran it)
  passages.py      # 2-3 realistic passages (prose, short email, one more genre)
  harness.py       # core loop: walk cursor, send requests, stream, measure, collect
  experiments.py   # the controlled comparisons (cache on/off, output cap, quant, decoding)
  analyze.py       # aggregate stats (median/p90), accuracy metric, charts
  README.md        # exact run instructions
  results/         # generated: latency tables, example suggestions, plots
  WRITEUP.md       # answers to the 5 questions, in Robert's voice, every claim measured
```

Each file has one job and a clear interface, so pieces can be understood and run
independently.

## 7. The core loop (harness.py)

1. Take a realistic passage.
2. Walk a cursor through it, stopping at ~8-10 points (sentence/word boundaries). At each
   stop, `prefix = passage[:cursor]`.
3. POST the prefix to `/completion` with streaming on.
4. Record per request: **TTFT** (client-side), **total** wall-clock, server `timings`
   (prefill ms, decode ms, tokens/sec), and the suggestion text (+ logprobs).

**Measurement rigor** (the credibility upgrade): discard **warm-up** runs, then take **N
repetitions** per cursor point and report **median + p90**, never a single noisy sample.
Fix a seed for any sampling so runs are reproducible.

## 8. Latency experiments (experiments.py)

1. **KV-cache reuse on vs off** (`cache_prompt` true/false) — headline result; expect
   prefill to stay flat with reuse and grow with prefix length without it.
2. **Granularity / output-length sweep** (confirmed 2026-06-15) — word (~1-3 tokens) vs
   **phrase-to-sentence** (~8-16, stop at sentence/newline) vs multi-line (~32-64). One
   sweep, two payoffs: it shows how latency scales with how much we commit to predicting
   (the decode-cost story) *and* yields example suggestions at each granularity (the product
   story). **Phrase-to-sentence is the default** for all other experiments. Discipline:
   granularity is its own isolated sweep — we do NOT re-run cache/quant/decoding at every
   granularity (that combinatorial blowup would be over-building).
3. **Quantization Q4 vs Q8** (confirmed 2026-06-15) — latency *and* the 8 GB memory tradeoff
   (see §3). Q4 is the primary/ship model; we compare against Q8, and **if Q8 forces
   swapping on 8 GB we report that as a finding** rather than a clean number — the failure
   case is itself relevant to on-device shipping. Download-test-delete to fit disk.
4. **GPU offload / threads** — recorded from server config, not a sweep.
5. **Speculative decoding — STRETCH only.** Run it *only* if a tokenizer-compatible draft
   model exists; otherwise cover it rigorously in the Q5 discussion. We will not fabricate
   a number.

**Charts** (matplotlib): prefill-time vs prefix-length with cache on/off overlaid (the most
persuasive image), plus a per-point latency breakdown.

## 9. Accuracy (analyze.py)

Latency is easy to measure; quality is the honest-reporting challenge.

- **Held-out proxy metric:** because we walk a *real* passage, the true continuation is
  known. Compute **next-word exact-match rate** and **token prefix-overlap** between the
  suggestion and the held-out real text, across many cursor points and 2-3 passages → an
  actual aggregate number. **Stated bias:** a different-but-valid continuation scores as
  "wrong," so this number is a floor, not the truth. We complement it with eyeballed
  examples.
- **Confidence gating via logprobs → precision/coverage curve:** only surface a suggestion
  when model confidence clears a threshold; show how raising the bar trades coverage for
  acceptance quality. This directly addresses "good enough that people actually accept it"
  and is product-grade reasoning, not just a metric.
- **Boundary-aware truncation:** cut the suggestion at a clean word/sentence end; show
  before/after examples.

## 10. Decoding depth (write-up Q3)

Explain and *show*: temperature, top_p, top_k, min_p, repetition penalty. Greedy vs sampled
side by side with example suggestions and their latency cost (near-identical latency,
different character → for autocomplete, low-temp/greedy usually wins).

## 11. Middle-of-text (write-up Q4) — demonstrated, not asserted

Gemma is **causal / left-to-right**: it only sees text *before* the cursor. We will
actually feed it a mid-document case (real text before *and* after the cursor) and **show
it ignoring the suffix**. Then explain **FIM (Fill-in-the-Middle)** — models trained with
prefix/suffix/middle tokens — and name the ones that do it right (CodeGemma, StarCoder,
DeepSeek-Coder, Codestral). Evidence beats a paragraph of claims.

## 12. Production (write-up Q5)

Discussion grounded in what we measured: debouncing keystrokes, cancelling in-flight
requests, prompt/KV caching and warm-up, persistent KV cache, speculative decoding,
batching/concurrency across users, **acceptance-rate telemetry** (tie back to §9), memory
lifecycle on constrained devices (tie back to §3), privacy of local text, and fallbacks.

## 13. Deliverables → take-home mapping

- **Code + run instructions** → `run_server.sh`, `harness.py`, `experiments.py`,
  `analyze.py`, `README.md`
- **Latency numbers + example suggestions** → `results/` (tables, examples, plots)
- **Write-up (the 5 questions)** → `WRITEUP.md`, in Robert's voice, every claim tied to a
  measured number or shown example; explicit about proxies and stretch items.
- **Machine reported** → MacBook Air M1, 8 GB (§3).

## 14. Out of scope (so we do not over-build)

No UI. No keystroke/timing simulation. No multi-model leaderboard. No quantization sweep
beyond Q4/Q8. Speculative decoding is run only if trivially feasible, else discussed. The
bar is "really good and honest," not "exhaustive."

## 15. Honesty commitments

- Single numbers are suspect → report distributions (median/p90) over repetitions.
- The accuracy proxy is a floor, not truth → say so, show examples.
- If Q8 won't fit 8 GB cleanly, or no draft model exists, or the GGUF name differs → report
  it plainly rather than hide it.

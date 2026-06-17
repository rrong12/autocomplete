# Local Autocomplete with llama.cpp — Write-up

**Machine:** MacBook Air, Apple M1 (4 performance + 4 efficiency cores), **8 GB RAM**, macOS 14.2.1, Metal GPU.
**Model:** `gemma-4-E2B` (base), quantized to `Q4_K_M` (~3 GB) and served by `llama-server`.
**Harness:** Python driving `llama-server`'s native `/completion` endpoint (streaming for TTFT; server-side `timings` for prefill/decode; `n_probs` for token logprobs). All latencies are medians over repetitions with a warm-up discarded.

> **One honest caveat up front:** the M1 Air is fanless and this was a long session, so it was thermal-throttling by the end — *absolute* latencies are inflated. Every comparison below is measured in the same thermal state, so the *relative* results (cache on/off, cold/incremental, Q4/Q8, granularity) hold; treat the absolute milliseconds as "this laptop, hot," not "the model's best case."

---

## 1. Model choice — I picked the **base** model

Autocomplete is *continuation*: given the text so far, predict what comes next. That is exactly what a base LM does. The instruction-tuned variant is trained to *respond* to chat turns, which is a different task. I confirmed this rather than asserting it (`results/model_headtohead.md`):

- **base** on a raw prefix → clean continuation (`"The quick brown"` → `"fox jumps over the lazy dog."`).
- **`-it`** on a raw prefix → also continues, but leaks chatbot artifacts (e.g. `"Meeting notes: the team agreed to"` → `"...related to [Topic] due to [Reason]"` — placeholder brackets).
- **`-it`** used the way it's meant to (chat template) → it *reasons first*: for `"The quick brown"` it emitted **774 characters of "Thinking Process: 1. Analyze the input..."** before any answer. For inline autocomplete that reasoning latency is disqualifying.

A real-world wrinkle that turned out to *strengthen* this answer: base `gemma-4-E2B` ships **only as a 10 GB multimodal safetensors with no GGUF**, while Google publishes deployable GGUFs only for the `-it` model. So I converted the **text tower** myself (llama.cpp's `Gemma4Model`) and quantized it. The lesson: base is the right *fit* for the task, even though the *shippable* artifact normally isn't handed to you.

## 2. Latency — where the time goes, and what I did about it

Inference is two phases: **prefill** (read the prefix, build the KV cache — cost scales with prefix length) and **decode** (emit tokens one at a time — sequential). **TTFT** (time to first token) is the "feels-instant" metric and is prefill-dominated. From `results/latency_q4.csv`, TTFT grows from ~97 ms (5-char prefix) to ~920 ms (1.3k-char prefix) as prefill grows.

**The biggest lever is KV-cache reuse.** On the brief's short passages it barely shows (1.1x — prefill is cheap at ~70 tokens, so reusing it saves little). That's misleading, so I added a long-context probe (`results/longcontext.png`, tiling the press release): as the prefix grows, **cold prefill scales linearly to ~9.1 s at 2,329 tokens, while incremental prefill with cache reuse stays flat at ~80 ms** — a **~100x** difference. In a real document the cursor sits deep in the text, so without cache every keystroke pays for the whole document; with cache it pays only for the few new tokens. That is the single most important latency decision.

Other levers I used / measured:
- **Output cap / granularity** (`results/granularity.csv`): word ≈ 543 ms, phrase-to-sentence ≈ 828 ms, multi-line ≈ 887 ms median. You only need a few words, so capping `n_predict` and stopping at a sentence/newline cuts decode you'd otherwise waste.
- **Metal GPU offload** (`-ngl 999`) and a **warm-up** request to absorb one-time kernel compilation.

## 3. Accuracy — how good, and how to push it further

**How good:** on the article I get a **60 % exact next-word match** (`results/accuracy.csv`). That number is a *floor*: it counts a valid-but-different continuation as wrong. Eyeballing shows the suggestions are genuinely good — e.g. the model produced `"...in the Western Conference."` (exact) and `"next chapter of the Chicago Bulls"` vs the true `"next era of Bulls basketball"` (semantically dead-on). With almost no context (a 5-char prefix) it emits web junk — context matters a lot.

**Beyond swapping the model, the decoding step is the main knob.** The model outputs a probability distribution; decoding turns it into tokens:
- **Greedy** (temperature 0) — take the most likely token. Deterministic, high-confidence, and the right default for autocomplete: you want the single most probable continuation, not variety.
- **Sampling** (temperature / top-p / top-k / min-p) — adds diversity at the cost of reliability. Useful when you offer multiple suggestions, risky for a single inline ghost-text. Latency is ~unchanged; quality/consistency is the tradeoff. (Beam search would improve quality but costs latency — not worth it for this use case.)
- **Confidence gating** (`results/precision_coverage.png`) — the best accuracy lever I found that *isn't* a model swap. Use the suggestion's min token logprob as a confidence and only show it above a threshold. Raising the bar trades coverage for precision (e.g. threshold −2 → show ~60 % of the time at ~75 % precision, vs a 60 % base rate). On a small 20-point sample the effect is modest but real, and it directly raises *accepted* quality — you suppress the bad suggestions rather than show noise.

## 4. Where this model falls short — the cursor in the middle

`gemma-4-E2B` is **causal / left-to-right**: it only sees text *before* the cursor. When the cursor sits mid-document with text after it, the model is blind to that suffix. Demonstrated in `results/middle_of_text.md`: given `"Dear Professor Smith, I am writing to"`, the model continued `"express my deep appreciation for the exceptional teaching..."` — but the hidden suffix was `"I would be grateful for a one-week extension."` It guessed the **wrong intent** because it couldn't see where the message was going.

What I'd want instead: a model trained for **Fill-in-the-Middle (FIM)** with prefix/suffix/middle sentinel tokens, so it conditions on *both* sides of the cursor. The code-completion models do this well — StarCoder, DeepSeek-Coder, Codestral, CodeLlama — and it's the standard approach for editor autocomplete.

## 5. Production — what I'd do differently shipping this to many people

- **Don't fire on every keystroke.** Debounce, and **cancel in-flight requests** when the user keeps typing — most requests are thrown away.
- **Persistent KV cache + warm-up** per document, so a returning cursor reuses prefill (see §2 — this is the whole latency game on long docs).
- **Speculative decoding** with a tiny draft model to cut decode time (I'd validate it needs a tokenizer-compatible draft; didn't have one here, so this is reasoned, not measured).
- **Acceptance-rate telemetry**, and feed it back into the confidence threshold from §3 — ship a suggestion only when expected acceptance clears a bar.
- **Memory lifecycle on constrained devices.** This bit me directly: **Q8 (4.6 GB) does not fit cleanly on 8 GB** — the first inferences swapped catastrophically (5 s, then a 105 s call), settling to ~1.5–2x Q4 once resident (`results/latency_q8.csv`). **Q4 (~3 GB) is the realistic ship target.** Pick the quant by the device's RAM budget, and unload the model when idle.
- **Thermal management** — sustained inference on a fanless laptop throttles; a real product paces work and watches for it.
- **Privacy** — keep it fully on-device (no network), which is the whole reason for a local model.

---

### Honesty notes
- Latencies are thermal-throttled (fanless M1, long session); relative comparisons hold, absolutes are inflated.
- 60 % next-word accuracy is a proxy floor — semantically-correct-but-different suggestions count as misses.
- The long-context numbers tile the real press release; valid because prefill latency depends only on token *count*, not content (it would **not** be valid for the accuracy test, which uses real un-tiled passages).
- Confidence-gating gains are real but measured on a small (20-point) sample.
- Q8 latencies are swap-contaminated on 8 GB — that *is* the finding, not a clean curve.

"""Read the experiment CSVs in results/ and emit charts + a markdown summary.
Safe to run repeatedly; skips any output whose CSV isn't present yet."""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "results")


def _read(name):
    path = os.path.join(RESULTS, f"{name}.csv")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return list(csv.DictReader(f))


def chart_cache():
    rows = _read("cache")
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for on, style in (("True", "-o"), ("False", "--s")):
        pts = sorted((int(r["prefix_len"]), float(r["prefill_ms_median"]))
                     for r in rows if r["cache_on"] == on)
        ax.plot([x for x, _ in pts], [y for _, y in pts], style,
                label=f"cache {'on' if on == 'True' else 'off'}")
    ax.set_xlabel("prefix length (chars)")
    ax.set_ylabel("prefill ms (median)")
    ax.set_title("KV-cache reuse: prefill vs prefix length")
    ax.legend()
    ax.grid(alpha=.3)
    fig.savefig(os.path.join(RESULTS, "cache.png"), dpi=120, bbox_inches="tight")
    print("wrote results/cache.png")


def chart_quant():
    q4, q8 = _read("latency_q4"), _read("latency_q8")
    if not (q4 and q8):
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for rows, lbl, style in ((q4, "Q4", "-o"), (q8, "Q8", "--s")):
        pts = sorted((int(r["prefix_len"]), float(r["total_ms"])) for r in rows)
        ax.plot([x for x, _ in pts], [y for _, y in pts], style, label=lbl)
    ax.set_yscale("log")   # Q8 swaps catastrophically on 8GB -> log scale to show it
    ax.set_xlabel("prefix length (chars)")
    ax.set_ylabel("total ms (median, log scale)")
    ax.set_title("Quantization: Q4 vs Q8 (Q8 swaps on 8 GB)")
    ax.legend()
    ax.grid(alpha=.3, which="both")
    fig.savefig(os.path.join(RESULTS, "quant.png"), dpi=120, bbox_inches="tight")
    print("wrote results/quant.png")


def chart_longcontext():
    rows = _read("longcontext")
    if not rows:
        return
    tok = [int(r["prompt_tokens"]) for r in rows]
    cold = [float(r["cold_prefill_ms"]) for r in rows]
    inc = [float(r["incremental_prefill_ms"]) for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(tok, cold, "-o", label="cold prefill (no cache)")
    ax.plot(tok, inc, "-s", label="incremental prefill (cache reuse)")
    ax.set_xlabel("prefix length (tokens)")
    ax.set_ylabel("prefill ms (median)")
    ax.set_title("KV-cache lever on long context")
    ax.legend()
    ax.grid(alpha=.3)
    fig.savefig(os.path.join(RESULTS, "longcontext.png"), dpi=120, bbox_inches="tight")
    print("wrote results/longcontext.png")


def chart_precision_coverage():
    rows = _read("precision_coverage")
    if not rows:
        return
    th = [float(r["min_logprob_threshold"]) for r in rows]
    cov = [float(r["coverage"]) for r in rows]
    prec = [float(r["precision"]) for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(th, cov, "-o", label="coverage")
    ax.plot(th, prec, "-s", label="precision")
    ax.set_xlabel("confidence threshold (min logprob)")
    ax.set_ylabel("rate")
    ax.set_title("Confidence gating: precision vs coverage")
    ax.legend()
    ax.grid(alpha=.3)
    fig.savefig(os.path.join(RESULTS, "precision_coverage.png"), dpi=120, bbox_inches="tight")
    print("wrote results/precision_coverage.png")


def summary_md():
    lines = ["# Results summary\n"]

    g = _read("granularity")
    if g:
        lines += ["## Granularity — latency vs how much we predict\n",
                  "| granularity | n_predict | avg median ms | max p90 ms |",
                  "|---|---|---|---|"]
        agg = {}
        for r in g:
            k = r["granularity"]
            agg.setdefault(k, [r["n_predict"], [], []])
            agg[k][1].append(float(r["total_ms_median"]))
            agg[k][2].append(float(r["total_ms_p90"]))
        for k, (n, med, p90) in agg.items():
            lines.append(f"| {k} | {n} | {sum(med)/len(med):.0f} | {max(p90):.0f} |")
        lines.append("")

    c = _read("cache")
    if c:
        on = [float(r["prefill_ms_median"]) for r in c if r["cache_on"] == "True"]
        off = [float(r["prefill_ms_median"]) for r in c if r["cache_on"] == "False"]
        lines += ["## KV-cache reuse — prefill ms (median)\n",
                  f"- cache ON:  avg {sum(on)/len(on):.0f} ms, max {max(on):.0f} ms",
                  f"- cache OFF: avg {sum(off)/len(off):.0f} ms, max {max(off):.0f} ms",
                  f"- speedup at longest prefix: {max(off)/max(on):.1f}x\n"]

    a = _read("accuracy")
    if a:
        n = len(a)
        hits = sum(int(r["next_word_match"]) for r in a)
        lines += ["## Accuracy — held-out next-word match\n",
                  f"- next-word match: {hits}/{n} = {100*hits/n:.0f}% "
                  "(proxy: a valid-but-different continuation scores as wrong)\n",
                  "### Example suggestions\n",
                  "| prefix_len | suggestion | true continuation |",
                  "|---|---|---|"]
        for r in a[:8]:
            lines.append(f"| {r['prefix_len']} | {r['suggestion'][:45]} | {r['truth_head']} |")
        lines.append("")

    pc = _read("precision_coverage")
    if pc:
        lines += ["## Confidence gating — precision vs coverage\n",
                  "| min-logprob threshold | coverage | precision |",
                  "|---|---|---|"]
        for r in pc:
            lines.append(f"| {r['min_logprob_threshold']} | "
                         f"{float(r['coverage']):.2f} | {float(r['precision']):.2f} |")
        lines.append("")

    q4, q8 = _read("latency_q4"), _read("latency_q8")
    if q4 and q8:
        def avg_total(rows):
            v = [float(r["total_ms"]) for r in rows]
            return sum(v) / len(v)
        lines += ["## Quantization — Q4 vs Q8 (total ms, avg across prefixes)\n",
                  f"- Q4: {avg_total(q4):.0f} ms",
                  f"- Q8: {avg_total(q8):.0f} ms\n"]

    out = "\n".join(lines)
    with open(os.path.join(RESULTS, "summary.md"), "w") as f:
        f.write(out)
    print("\n" + out)


if __name__ == "__main__":
    chart_cache()
    chart_quant()
    chart_longcontext()
    chart_precision_coverage()
    summary_md()

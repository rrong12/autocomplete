"""Q4 (write-up): the chosen model is causal / left-to-right — it only sees text
BEFORE the cursor. Here we show it continuing a prefix while blind to the text
AFTER the cursor, which a Fill-in-the-Middle (FIM) model could condition on.
Run against the base server on :8080."""
import requests

SERVER = "http://127.0.0.1:8080"

# (prefix before cursor, suffix after cursor)
CASES = [
    ("I stopped by the store to grab milk, eggs, and",
     " so we could finally bake the birthday cake tonight."),
    ("The interview is scheduled for",
     " so please make sure you are ready by Friday morning."),
    ("Dear Professor Smith, I am writing to",
     " I would be grateful for a one-week extension on the final paper."),
]


def cont(prefix, n=16):
    r = requests.post(f"{SERVER}/completion",
                      json={"prompt": prefix, "n_predict": n, "temperature": 0.0,
                            "cache_prompt": True, "stop": ["\n"]})
    return r.json().get("content", "").strip()


def main():
    lines = ["# Q4: middle-of-text — the causal model ignores the suffix\n",
             "The model sees only text BEFORE the cursor. Below it continues each prefix;",
             "the **suffix** is the text after the cursor that it CANNOT see but a",
             "Fill-in-the-Middle model could condition on.\n"]
    for pre, suf in CASES:
        c = cont(pre)
        lines += [f"### prefix: `{pre}`",
                  f"- model continues → `{c}`",
                  f"- (unseen) suffix → `{suf}`",
                  "- a suffix-aware fill would bridge into that suffix; the causal model can't.\n"]
    out = "\n".join(lines)
    with open("results/middle_of_text.md", "w") as f:
        f.write(out)
    print(out)


if __name__ == "__main__":
    main()

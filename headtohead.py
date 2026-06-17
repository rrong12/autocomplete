"""Q1 evidence: send identical RAW prefixes (no chat template) to the base and the
-it model, and show how each behaves. Base should *continue* the text; -it tends to
*respond* / break character because it was tuned for chat. Run with base on :8080
and -it on :8081 (launch a second llama-server on the -it GGUF).
Usage: python headtohead.py [base_url] [it_url]"""
import sys

import requests

PROMPTS = [
    "The quick brown",
    "Hi Marcus, thanks for sending over the draft proposal",
    "Meeting notes: the team agreed to",
    "The morning fog had not yet lifted when she stepped onto the",
]


def gen(server, prompt, n_predict=24):
    r = requests.post(f"{server}/completion",
                      json={"prompt": prompt, "n_predict": n_predict,
                            "temperature": 0.0, "cache_prompt": True, "stop": []})
    return r.json().get("content", "").replace("\n", " ⏎ ").strip()


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"
    it = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8081"
    lines = ["# Q1: base vs -it on RAW prefixes (no chat template)\n",
             "Same raw prefix POSTed to each model's `/completion`. The base model is "
             "trained to continue text; the instruction-tuned model is trained to "
             "respond to chat, so on a bare prefix it tends to break the continuation.\n"]
    for p in PROMPTS:
        lines.append(f"### Prefix: `{p}`")
        lines.append(f"- **base** → `{gen(base, p)}`")
        lines.append(f"- **-it**  → `{gen(it, p)}`\n")
    out = "\n".join(lines)
    with open("results/model_headtohead.md", "w") as f:
        f.write(out)
    print(out)


if __name__ == "__main__":
    main()

SERVER = "http://127.0.0.1:8080"

# Phrase-to-sentence default (the chosen granularity); other granularities live
# in the experiments sweep, not here.
DEFAULT = dict(n_predict=16, temperature=0.0, top_p=0.95, top_k=40,
               cache_prompt=True, stop=["\n", ". ", "! ", "? "], n_probs=1)

GRANULARITIES = {"word": 3, "phrase_sentence": 16, "multiline": 64}

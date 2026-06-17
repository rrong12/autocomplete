# Q4: middle-of-text — the causal model ignores the suffix

The model sees only text BEFORE the cursor. Below it continues each prefix;
the **suffix** is the text after the cursor that it CANNOT see but a
Fill-in-the-Middle model could condition on.

### prefix: `I stopped by the store to grab milk, eggs, and`
- model continues → `some other things. I was greeted by a very friendly and helpful employee. I`
- (unseen) suffix → ` so we could finally bake the birthday cake tonight.`
- a suffix-aware fill would bridge into that suffix; the causal model can't.

### prefix: `The interview is scheduled for`
- model continues → `10:30 a.m. at the office of the president`
- (unseen) suffix → ` so please make sure you are ready by Friday morning.`
- a suffix-aware fill would bridge into that suffix; the causal model can't.

### prefix: `Dear Professor Smith, I am writing to`
- model continues → `express my deep appreciation for the exceptional teaching and mentorship I received during my time at`
- (unseen) suffix → ` I would be grateful for a one-week extension on the final paper.`
- a suffix-aware fill would bridge into that suffix; the causal model can't.

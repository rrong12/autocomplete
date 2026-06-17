# Q1 evidence: base vs -it (gemma-4-E2B)

Autocomplete = continuation. Identical **raw** prefixes were POSTed to each
model's /completion endpoint (how an autocomplete client would call it).

## Raw prefix -> continuation

### `The quick brown`
- **base** -> fox jumps over the lazy dog. /  / The quick brown fox jumps over the lazy dog. /  / The quick brown fox jumps
- **-it**  -> fox jumps over the lazy dog.

### `Hi Marcus, thanks for sending over the draft proposal`
- **base** -> . I have a few comments: /  / 1. I think the proposal is a bit too long. I would suggest cutting
- **-it**  -> . /  / I've had a chance to review it, and overall, it's a strong foundation. I particularly

### `Meeting notes: the team agreed to`
- **base** -> meet on the 1st of the month to discuss the project. /  / The team agreed to meet on the 1
- **-it**  -> prioritize the Q3 roadmap items related to [Topic] due to [Reason]. /  / Action Item: [Specific Task]

### `The morning fog had not yet lifted when she stepped onto the`
- **base** -> deck of the <em>USS</em> <em>Saratoga</em>. The ship was moored at the pier in the harbor
- **-it**  -> balcony, the air thick with the scent of damp earth and blooming jasmine. A gentle breeze rustled the leaves, carrying

## -it via its chat template (its *intended* use) — unusable here

Called as a chat model, -it is a **reasoning assistant**. For `The quick brown`
it produced **774 chars of "Thinking Process: 1. Analyze the input... 2. Identify
the source..."** before any answer, hitting the 200-token limit still mid-thought.
For inline autocomplete that reasoning latency is fatal.

## Takeaway

- **base** continues text cleanly — the right fit for autocomplete.
- **-it on raw prefixes** also continues, but leaks instruction-tuned artifacts
  (e.g. Meeting notes -> "... related to **[Topic]** due to **[Reason]**").
- **-it used as intended (chat)** reasons before answering -> wrong behavior *and*
  far too slow.

=> We chose **base**. (In practice base ships only as a 10 GB multimodal
safetensors with no GGUF, so we converted the text tower ourselves — see README.)

# Results summary

## Granularity — latency vs how much we predict

| granularity | n_predict | avg median ms | max p90 ms |
|---|---|---|---|
| word | 3 | 543 | 960 |
| phrase_sentence | 16 | 828 | 1734 |
| multiline | 64 | 887 | 1940 |

## KV-cache reuse — prefill ms (median)

- cache ON:  avg 425 ms, max 816 ms
- cache OFF: avg 517 ms, max 888 ms
- speedup at longest prefix: 1.1x

## Accuracy — held-out next-word match

- next-word match: 12/20 = 60% (proxy: a valid-but-different continuation scores as wrong)

### Example suggestions

| prefix_len | suggestion | true continuation |
|---|---|---|
| 5 | 2016 | ANNOUNCE TIAGO SPLITTER AS HEAD COACH |
| 161 | 15th head coach in franchise history. | 25th head coach in franchise history. |
| 325 | John Paxson | Bryson Graham. "Throughout our process, |
| 478 | personality and leadership style would fit wi | teams compete every single night. He ha |
| 604 | and we are excited to have him lead our team. | and we believe his vision is the right |
| 720 | next chapter of the Chicago Bulls." | next era of Bulls basketball."

Splitte |
| 875 | He has also served as an assistant coach for  | Elevated to the role early in the seaso |
| 1003 | in the Western Conference. | in the Western Conference.

"I want to |

## Confidence gating — precision vs coverage

| min-logprob threshold | coverage | precision |
|---|---|---|
| -6 | 1.00 | 0.60 |
| -5 | 1.00 | 0.60 |
| -4 | 1.00 | 0.60 |
| -3 | 0.95 | 0.63 |
| -2.5 | 0.75 | 0.73 |
| -2 | 0.65 | 0.77 |
| -1.5 | 0.45 | 0.67 |
| -1 | 0.20 | 0.75 |
| -0.5 | 0.10 | 0.50 |

## Quantization — Q4 vs Q8 (total ms, avg across prefixes)

- Q4: 828 ms
- Q8: 12342 ms

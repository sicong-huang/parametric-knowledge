# Tier 2 — main vs placebo arm comparison (DiD + no-op instability)

Sources: `/home/mjsheu/parametric-knowledge/premise2/corpus/tier2_continuations.jsonl`, `/home/mjsheu/parametric-knowledge/premise2/corpus/tier2_placebo_continuations.jsonl`.

Paired on (trace_id, hop_index) -- only questions present in both the main and placebo arms, in all 3 conditions, are used. DiD = main's (corrupted-paraphrase) gap minus placebo's, per question; bootstrap 95% CI over questions. If DiD excludes zero and the main gap is positive, the on-chain effect is distinguishable from generic force-decode instability.

| Model | Dataset | n_paired | main gap | placebo gap | DiD (main-placebo) | P(wrong\|orig) main | P(wrong\|orig) placebo |
| :-- | :-- | --: | --: | --: | --: | --: | --: |
| google/gemma-4-E2B-it | 2wikimultihopqa | 2 | +0.000 | -0.125 | +0.125 [+0.000,+0.250] | 0.625 | 0.625 |
| google/gemma-4-E2B-it | hotpotqa | 6 | +0.125 | +0.000 | +0.125 [+0.000,+0.375] | 0.125 | 0.333 |
| google/gemma-4-E2B-it | musique | 1 | +0.000 | +0.000 | +0.000 [+0.000,+0.000] | 0.000 | 0.000 |
| google/gemma-4-E4B-it | 2wikimultihopqa | 5 | +0.200 | -0.050 | +0.250 [-0.100,+0.750] | 0.500 | 0.150 |
| google/gemma-4-E4B-it | hotpotqa | 5 | +0.050 | -0.150 | +0.200 [-0.100,+0.650] | 0.350 | 0.300 |
| google/gemma-4-E4B-it | musique | 4 | -0.062 | +0.125 | -0.188 [-0.562,+0.000] | 0.000 | 0.125 |
| nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 | 2wikimultihopqa | 18 | +0.083 | +0.111 | -0.028 [-0.278,+0.236] | 0.250 | 0.319 |
| nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 | hotpotqa | 7 | +0.000 | -0.071 | +0.071 [-0.071,+0.179] | 0.214 | 0.107 |
| nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 | musique | 3 | +0.500 | -0.167 | +0.667 [+0.500,+1.000] | 0.583 | 0.500 |
| nvidia/NVIDIA-Nemotron-Nano-9B-v2 | 2wikimultihopqa | 41 | +0.110 | +0.043 | +0.067 [-0.055,+0.201] | 0.177 | 0.152 |
| nvidia/NVIDIA-Nemotron-Nano-9B-v2 | hotpotqa | 34 | +0.000 | -0.037 | +0.037 [-0.022,+0.096] | 0.066 | 0.088 |
| nvidia/NVIDIA-Nemotron-Nano-9B-v2 | musique | 13 | +0.038 | -0.038 | +0.077 [-0.096,+0.288] | 0.327 | 0.404 |

**Pooled DiD across all paired questions (n=139): +0.068 [+0.004,+0.135]**

## No-op instability (P(wrong | original), all candidates were em_correct==1)

All of these traces were originally answered correctly with no edit at all. Any nonzero value below is pure temperature-1.0 force-decode resampling noise, not an effect of corruption -- this is the no-op control (exp20.md Decision item (c)), already present as the `original` condition in both arms.

| Model | Dataset | n_q main | P(wrong\|orig) main | n_q placebo | P(wrong\|orig) placebo |
| :-- | :-- | --: | --: | --: | --: |
| google/gemma-4-E2B-it | 2wikimultihopqa | 17 | 0.368 | 6 | 0.333 |
| google/gemma-4-E2B-it | hotpotqa | 23 | 0.228 | 6 | 0.333 |
| google/gemma-4-E2B-it | musique | 2 | 0.000 | 1 | 0.000 |
| google/gemma-4-E4B-it | 2wikimultihopqa | 17 | 0.338 | 7 | 0.107 |
| google/gemma-4-E4B-it | hotpotqa | 23 | 0.315 | 5 | 0.300 |
| google/gemma-4-E4B-it | musique | 9 | 0.139 | 5 | 0.250 |
| nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 | 2wikimultihopqa | 62 | 0.246 | 22 | 0.318 |
| nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 | hotpotqa | 29 | 0.181 | 7 | 0.107 |
| nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 | musique | 8 | 0.531 | 3 | 0.500 |
| nvidia/NVIDIA-Nemotron-Nano-9B-v2 | 2wikimultihopqa | 69 | 0.163 | 46 | 0.136 |
| nvidia/NVIDIA-Nemotron-Nano-9B-v2 | hotpotqa | 60 | 0.104 | 38 | 0.079 |
| nvidia/NVIDIA-Nemotron-Nano-9B-v2 | musique | 24 | 0.323 | 15 | 0.433 |

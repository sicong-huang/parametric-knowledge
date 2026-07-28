# Tier 2 — position and self-correction analysis

Classification pass over existing `tier2_continuations.jsonl` (corrupted condition only). No new generation. Self-correction is a lexical-marker heuristic — treat as a coarse proxy, not a precise rate; spot-check before citing.

## Flip rate by hop position

| Model | Position | n | P(wrong) |
| :-- | :-- | --: | --: |
| google/gemma-4-E2B-it | hop1 | 24 | 0.250 |
| google/gemma-4-E2B-it | hop1(only) | 88 | 0.364 |
| google/gemma-4-E2B-it | last | 16 | 0.062 |
| google/gemma-4-E2B-it | mid | 8 | 0.250 |
| google/gemma-4-E4B-it | hop1 | 12 | 0.250 |
| google/gemma-4-E4B-it | hop1(only) | 140 | 0.300 |
| google/gemma-4-E4B-it | last | 24 | 0.167 |
| google/gemma-4-E4B-it | mid | 8 | 0.000 |
| nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 | hop1 | 68 | 0.250 |
| nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 | hop1(only) | 232 | 0.371 |
| nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 | last | 68 | 0.279 |
| nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 | mid | 4 | 0.000 |
| nvidia/NVIDIA-Nemotron-Nano-9B-v2 | hop1 | 88 | 0.318 |
| nvidia/NVIDIA-Nemotron-Nano-9B-v2 | hop1(only) | 376 | 0.229 |
| nvidia/NVIDIA-Nemotron-Nano-9B-v2 | last | 92 | 0.098 |
| nvidia/NVIDIA-Nemotron-Nano-9B-v2 | mid | 20 | 0.050 |

## Self-correction rate by model (corrupted condition)

| Model | n | n_self_corrected | self_correction_rate |
| :-- | --: | --: | --: |
| google/gemma-4-E2B-it | 136 | 89 | 0.654 |
| google/gemma-4-E4B-it | 184 | 93 | 0.505 |
| nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 | 372 | 199 | 0.535 |
| nvidia/NVIDIA-Nemotron-Nano-9B-v2 | 576 | 177 | 0.307 |

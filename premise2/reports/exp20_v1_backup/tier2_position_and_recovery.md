# Tier 2 — position and self-correction analysis

Classification pass over existing `tier2_continuations.jsonl` (corrupted condition only). No new generation. Self-correction is a lexical-marker heuristic — treat as a coarse proxy, not a precise rate; spot-check before citing.

## Flip rate by hop position

| Model | Position | n | P(wrong) |
| :-- | :-- | --: | --: |
| google/gemma-4-E2B-it | hop1 | 24 | 0.208 |
| google/gemma-4-E2B-it | hop1(only) | 172 | 0.360 |
| google/gemma-4-E2B-it | last | 28 | 0.036 |
| google/gemma-4-E2B-it | mid | 8 | 0.125 |
| google/gemma-4-E4B-it | hop1 | 28 | 0.250 |
| google/gemma-4-E4B-it | hop1(only) | 228 | 0.298 |
| google/gemma-4-E4B-it | last | 36 | 0.194 |
| google/gemma-4-E4B-it | mid | 12 | 0.000 |
| nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 | hop1 | 68 | 0.250 |
| nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 | hop1(only) | 228 | 0.430 |
| nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 | last | 68 | 0.309 |
| nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 | mid | 4 | 0.000 |
| nvidia/NVIDIA-Nemotron-Nano-9B-v2 | hop1 | 92 | 0.315 |
| nvidia/NVIDIA-Nemotron-Nano-9B-v2 | hop1(only) | 372 | 0.234 |
| nvidia/NVIDIA-Nemotron-Nano-9B-v2 | last | 88 | 0.216 |
| nvidia/NVIDIA-Nemotron-Nano-9B-v2 | mid | 20 | 0.050 |

## Self-correction rate by model (corrupted condition)

| Model | n | n_self_corrected | self_correction_rate |
| :-- | --: | --: | --: |
| google/gemma-4-E2B-it | 232 | 149 | 0.642 |
| google/gemma-4-E4B-it | 304 | 128 | 0.421 |
| nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 | 368 | 195 | 0.530 |
| nvidia/NVIDIA-Nemotron-Nano-9B-v2 | 572 | 171 | 0.299 |

# Tier 2 results — force-decode flip rates

Source: `/home/mjsheu/parametric-knowledge/premise2/corpus/tier2_continuations.jsonl` (4012 continuation rows).

Per question (trace_id + hop_index), wrong-rate = mean(is_wrong) over the n={samples} force-decoded continuations for that condition. Bootstrap 95% CI over questions.

| Model | Dataset | n_q | P(wrong\|orig) | P(wrong\|corrupted) | P(wrong\|paraphrase) | corrupted-paraphrase gap | corrupted-original gap |
| :-- | :-- | --: | --: | --: | --: | --: | --: |
| google/gemma-4-E2B-it | 2wikimultihopqa | 13 | 0.327 | 0.308 [0.115,0.519] | 0.327 [0.154,0.519] | -0.019 [-0.192,+0.192] | -0.019 [-0.192,+0.192] |
| google/gemma-4-E2B-it | hotpotqa | 20 | 0.237 | 0.312 [0.150,0.475] | 0.312 [0.163,0.475] | +0.000 [-0.125,+0.125] | +0.075 [-0.025,+0.188] |
| google/gemma-4-E2B-it | musique | 1 | 0.000 | 0.000 [0.000,0.000] | 0.000 [0.000,0.000] | +0.000 [+0.000,+0.000] | +0.000 [+0.000,+0.000] |
| google/gemma-4-E4B-it | 2wikimultihopqa | 15 | 0.367 | 0.333 [0.150,0.517] | 0.267 [0.133,0.400] | +0.067 [-0.100,+0.217] | -0.033 [-0.200,+0.133] |
| google/gemma-4-E4B-it | hotpotqa | 23 | 0.315 | 0.293 [0.163,0.446] | 0.261 [0.152,0.380] | +0.033 [-0.087,+0.163] | -0.022 [-0.130,+0.098] |
| google/gemma-4-E4B-it | musique | 8 | 0.062 | 0.062 [0.000,0.188] | 0.094 [0.000,0.219] | -0.031 [-0.094,+0.000] | +0.000 [+0.000,+0.000] |
| nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 | 2wikimultihopqa | 60 | 0.250 | 0.317 [0.242,0.400] | 0.287 [0.217,0.362] | +0.029 [-0.042,+0.108] | +0.067 [-0.008,+0.150] |
| nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 | hotpotqa | 25 | 0.170 | 0.230 [0.130,0.340] | 0.190 [0.100,0.300] | +0.040 [-0.090,+0.180] | +0.060 [-0.060,+0.190] |
| nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 | musique | 8 | 0.531 | 0.719 [0.375,0.969] | 0.500 [0.250,0.750] | +0.219 [+0.062,+0.438] | +0.188 [+0.000,+0.375] |
| nvidia/NVIDIA-Nemotron-Nano-9B-v2 | 2wikimultihopqa | 66 | 0.170 | 0.235 [0.163,0.314] | 0.167 [0.106,0.239] | +0.068 [+0.000,+0.136] | +0.064 [-0.011,+0.136] |
| nvidia/NVIDIA-Nemotron-Nano-9B-v2 | hotpotqa | 56 | 0.094 | 0.116 [0.054,0.179] | 0.107 [0.054,0.174] | +0.009 [-0.040,+0.062] | +0.022 [-0.036,+0.080] |
| nvidia/NVIDIA-Nemotron-Nano-9B-v2 | musique | 22 | 0.307 | 0.409 [0.239,0.580] | 0.330 [0.182,0.489] | +0.080 [-0.045,+0.216] | +0.102 [-0.011,+0.239] |

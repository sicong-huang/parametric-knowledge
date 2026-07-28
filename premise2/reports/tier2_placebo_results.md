# Tier 2 results — force-decode flip rates

Source: `/home/mjsheu/parametric-knowledge/premise2/corpus/tier2_placebo_continuations.jsonl` (1892 continuation rows).

Per question (trace_id + hop_index), wrong-rate = mean(is_wrong) over the n={samples} force-decoded continuations for that condition. Bootstrap 95% CI over questions.

| Model | Dataset | n_q | P(wrong\|orig) | P(wrong\|corrupted) | P(wrong\|paraphrase) | corrupted-paraphrase gap | corrupted-original gap |
| :-- | :-- | --: | --: | --: | --: | --: | --: |
| google/gemma-4-E2B-it | 2wikimultihopqa | 4 | 0.500 | 0.688 [0.500,0.875] | 0.688 [0.375,0.938] | +0.000 [-0.188,+0.188] | +0.188 [+0.000,+0.375] |
| google/gemma-4-E2B-it | hotpotqa | 6 | 0.333 | 0.208 [0.000,0.542] | 0.208 [0.000,0.542] | +0.000 [-0.125,+0.125] | -0.125 [-0.292,+0.000] |
| google/gemma-4-E2B-it | musique | 1 | 0.000 | 0.000 [0.000,0.000] | 0.000 [0.000,0.000] | +0.000 [+0.000,+0.000] | +0.000 [+0.000,+0.000] |
| google/gemma-4-E4B-it | 2wikimultihopqa | 6 | 0.125 | 0.542 [0.250,0.833] | 0.500 [0.167,0.792] | +0.042 [-0.208,+0.292] | +0.417 [+0.167,+0.708] |
| google/gemma-4-E4B-it | hotpotqa | 5 | 0.300 | 0.100 [0.000,0.300] | 0.250 [0.000,0.550] | -0.150 [-0.450,+0.000] | -0.200 [-0.400,+0.000] |
| google/gemma-4-E4B-it | musique | 5 | 0.250 | 0.200 [0.000,0.400] | 0.200 [0.000,0.600] | +0.000 [-0.300,+0.300] | -0.050 [-0.150,+0.000] |
| nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 | 2wikimultihopqa | 19 | 0.316 | 0.434 [0.289,0.579] | 0.342 [0.211,0.474] | +0.092 [-0.079,+0.276] | +0.118 [-0.026,+0.276] |
| nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 | hotpotqa | 7 | 0.107 | 0.143 [0.000,0.357] | 0.214 [0.036,0.429] | -0.071 [-0.250,+0.071] | +0.036 [-0.143,+0.286] |
| nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 | musique | 3 | 0.500 | 0.250 [0.000,0.750] | 0.417 [0.250,0.750] | -0.167 [-0.250,+0.000] | -0.250 [-0.250,-0.250] |
| nvidia/NVIDIA-Nemotron-Nano-9B-v2 | 2wikimultihopqa | 43 | 0.145 | 0.198 [0.110,0.302] | 0.151 [0.081,0.227] | +0.047 [-0.023,+0.128] | +0.052 [+0.000,+0.110] |
| nvidia/NVIDIA-Nemotron-Nano-9B-v2 | hotpotqa | 37 | 0.081 | 0.054 [0.007,0.122] | 0.108 [0.041,0.196] | -0.054 [-0.115,-0.007] | -0.027 [-0.061,+0.000] |
| nvidia/NVIDIA-Nemotron-Nano-9B-v2 | musique | 15 | 0.433 | 0.383 [0.200,0.567] | 0.450 [0.250,0.650] | -0.067 [-0.167,+0.033] | -0.050 [-0.167,+0.100] |

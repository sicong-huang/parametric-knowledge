# Tier 2 results — force-decode flip rates

Source: `/home/mjsheu/parametric-knowledge/premise2/corpus/tier2_placebo_continuations.jsonl` (9364 continuation rows).

Per question (trace_id + hop_index), wrong-rate = mean(is_wrong) over the n={samples} force-decoded continuations for that condition. Bootstrap 95% CI over questions.

| Model                                 | Dataset         | n_q | P(wrong\|orig) | P(wrong\|corrupted) | P(wrong\|paraphrase) | corrupted-paraphrase gap | corrupted-original gap |
| :------------------------------------ | :-------------- | --: | -------------: | ------------------: | -------------------: | -----------------------: | ---------------------: |
| google/gemma-4-E2B-it                 | 2wikimultihopqa |  95 |          0.426 | 0.471 [0.403,0.542] |  0.416 [0.353,0.482] |   +0.055 [-0.021,+0.129] | +0.045 [-0.029,+0.118] |
| google/gemma-4-E2B-it                 | hotpotqa        |  66 |          0.258 | 0.273 [0.197,0.352] |  0.303 [0.223,0.394] |   -0.030 [-0.106,+0.045] | +0.015 [-0.061,+0.087] |
| google/gemma-4-E2B-it                 | musique         |   7 |          0.536 | 0.536 [0.250,0.821] |  0.429 [0.143,0.714] |   +0.107 [+0.036,+0.179] | +0.000 [-0.107,+0.107] |
| google/gemma-4-E4B-it                 | 2wikimultihopqa |  85 |          0.488 | 0.553 [0.479,0.624] |  0.474 [0.403,0.544] |   +0.079 [+0.012,+0.147] | +0.065 [+0.000,+0.129] |
| google/gemma-4-E4B-it                 | hotpotqa        |  84 |          0.268 | 0.256 [0.196,0.321] |  0.271 [0.202,0.345] |   -0.015 [-0.074,+0.042] | -0.012 [-0.071,+0.045] |
| google/gemma-4-E4B-it                 | musique         |  13 |          0.519 | 0.654 [0.519,0.788] |  0.635 [0.519,0.769] |   +0.019 [-0.096,+0.154] | +0.135 [-0.058,+0.346] |
| nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 | 2wikimultihopqa | 105 |          0.438 | 0.529 [0.467,0.593] |  0.469 [0.410,0.529] |   +0.060 [-0.007,+0.131] | +0.090 [+0.021,+0.164] |
| nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 | hotpotqa        |  69 |          0.268 | 0.377 [0.297,0.460] |  0.272 [0.199,0.351] |   +0.105 [+0.029,+0.185] | +0.109 [+0.029,+0.192] |
| nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 | musique         |  14 |          0.518 | 0.643 [0.446,0.821] |  0.500 [0.339,0.661] |   +0.143 [+0.036,+0.268] | +0.125 [+0.018,+0.232] |
| nvidia/NVIDIA-Nemotron-Nano-9B-v2     | 2wikimultihopqa | 110 |          0.332 | 0.445 [0.382,0.516] |  0.398 [0.339,0.459] |   +0.048 [-0.023,+0.123] | +0.114 [+0.039,+0.191] |
| nvidia/NVIDIA-Nemotron-Nano-9B-v2     | hotpotqa        |  99 |          0.194 | 0.265 [0.202,0.331] |  0.253 [0.189,0.318] |   +0.013 [-0.045,+0.073] | +0.071 [+0.013,+0.129] |
| nvidia/NVIDIA-Nemotron-Nano-9B-v2     | musique         |  26 |          0.385 | 0.490 [0.327,0.644] |  0.577 [0.423,0.731] |   -0.087 [-0.221,+0.038] | +0.106 [-0.010,+0.212] |

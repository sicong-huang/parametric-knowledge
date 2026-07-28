# Tier 2 results — force-decode flip rates

Source: `/home/mjsheu/parametric-knowledge/premise2/corpus/tier2_continuations.jsonl` (4708 continuation rows).

Per question (trace_id + hop_index), wrong-rate = mean(is_wrong) over the n={samples} force-decoded continuations for that condition. Bootstrap 95% CI over questions.

| Model                                 | Dataset         | n_q | P(wrong\|orig) | P(wrong\|corrupted) | P(wrong\|paraphrase) | corrupted-paraphrase gap | corrupted-original gap |
| :------------------------------------ | :-------------- | --: | -------------: | ------------------: | -------------------: | -----------------------: | ---------------------: |
| google/gemma-4-E2B-it                 | 2wikimultihopqa |  22 |          0.375 | 0.386 [0.227,0.557] |  0.330 [0.193,0.477] |   +0.057 [-0.102,+0.239] | +0.011 [-0.148,+0.205] |
| google/gemma-4-E2B-it                 | hotpotqa        |  32 |          0.203 | 0.211 [0.117,0.312] |  0.320 [0.195,0.461] |   -0.109 [-0.227,-0.008] | +0.008 [-0.078,+0.094] |
| google/gemma-4-E2B-it                 | musique         |   4 |          0.562 | 0.500 [0.000,1.000] |  0.500 [0.000,1.000] |   +0.000 [+0.000,+0.000] | -0.062 [-0.188,+0.000] |
| google/gemma-4-E4B-it                 | 2wikimultihopqa |  25 |          0.230 | 0.290 [0.170,0.430] |  0.250 [0.130,0.380] |   +0.040 [-0.050,+0.130] | +0.060 [-0.030,+0.150] |
| google/gemma-4-E4B-it                 | hotpotqa        |  40 |          0.156 | 0.269 [0.169,0.381] |  0.188 [0.106,0.287] |   +0.081 [-0.025,+0.194] | +0.113 [+0.031,+0.206] |
| google/gemma-4-E4B-it                 | musique         |  11 |          0.295 | 0.227 [0.045,0.455] |  0.205 [0.045,0.409] |   +0.023 [-0.068,+0.136] | -0.068 [-0.159,+0.000] |
| nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 | 2wikimultihopqa |  60 |          0.275 | 0.367 [0.279,0.463] |  0.283 [0.204,0.362] |   +0.083 [-0.004,+0.179] | +0.092 [-0.008,+0.196] |
| nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 | hotpotqa        |  25 |          0.160 | 0.310 [0.200,0.420] |  0.210 [0.130,0.300] |   +0.100 [+0.000,+0.200] | +0.150 [+0.040,+0.260] |
| nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 | musique         |   7 |          0.500 | 0.607 [0.286,0.893] |  0.571 [0.286,0.857] |   +0.036 [-0.143,+0.286] | +0.107 [-0.107,+0.429] |
| nvidia/NVIDIA-Nemotron-Nano-9B-v2     | 2wikimultihopqa |  65 |          0.150 | 0.231 [0.158,0.312] |  0.204 [0.138,0.281] |   +0.027 [-0.042,+0.100] | +0.081 [+0.015,+0.154] |
| nvidia/NVIDIA-Nemotron-Nano-9B-v2     | hotpotqa        |  56 |          0.089 | 0.152 [0.080,0.228] |  0.112 [0.049,0.183] |   +0.040 [-0.009,+0.094] | +0.062 [+0.018,+0.116] |
| nvidia/NVIDIA-Nemotron-Nano-9B-v2     | musique         |  22 |          0.295 | 0.477 [0.295,0.659] |  0.295 [0.148,0.466] |   +0.182 [+0.080,+0.307] | +0.182 [+0.080,+0.307] |

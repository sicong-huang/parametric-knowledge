---
id: E005
run_dir: experiment/exp5
condition: direct
model: Qwen/Qwen3.5-4B
datasets: [nq, triviaqa, popqa, hotpotqa, 2wikimultihopqa, musique, bamboogle, simpleqa_verified]
n_examples: 500
status: Done
---

# E005 — Direct answer versus reasoning (Qwen3.5-4B, direct arm)

**Owner:** Michelle Sheu
**Why:** Premise 1 sweep — same as E004 but next size up.
**Setup:** Qwen3.5-4B, vLLM, `enable_thinking=False`. Decoding per Qwen3.5
card's "Instruct (non-thinking) mode, general tasks": temp 0.7, top_p 0.8,
top_k 20, min_p 0, presence_penalty 1.5, rep_penalty 1.0, max 500 tokens.
Same 8 datasets/metrics/API requirements as `exp3`.
**Expected result:** Baseline accuracy per dataset; comparison point for
`exp6` (reasoning arm, same model).
**Result:**

| dataset | n | em | cover_em | judge_acc | ex_recall | avg_output_len | extraction_fail_rate |
|---|---|---|---|---|---|---|---|
| nq | 500 | 0.178 | 0.234 | 0.338 | 0.222 | 62.1 | 0.062 |
| triviaqa | 500 | 0.442 | 0.508 | 0.552 | 0.480 | 72.1 | 0.106 |
| popqa | 500 | 0.190 | 0.206 | 0.240 | 0.200 | 37.5 | 0.038 |
| hotpotqa | 500 | 0.176 | 0.238 | 0.306 | 0.188 | 115.2 | 0.236 |
| 2wikimultihopqa | 500 | 0.258 | 0.324 | 0.298 | 0.294 | 78.5 | 0.110 |
| musique | 500 | 0.024 | 0.080 | 0.090 | 0.044 | 196.5 | 0.462 |
| bamboogle | 125 | 0.272 | 0.320 | 0.360 | 0.304 | 90.7 | 0.144 |

simpleqa_verified (llm_judge, n=500, separate): overall_accuracy 0.078, accuracy_given_attempted 0.081, f1 0.080, not_attempted 20, extraction_fail_rate 0.104.

**What we learned:** Qwen3.5-4B direct arm beats Qwen3.5-2B (`exp3`) across every dataset — a clean model-size scaling effect at fixed direct-answer condition (e.g. triviaqa EM 0.152→0.442, nq 0.074→0.178). musique remains the hardest dataset by far (EM 0.024, extraction-failure 0.462 — near half of outputs fail to close the answer tag), suggesting a genuine reasoning-chain difficulty rather than pure scale, since more parameters barely move it. judge_accuracy tracks EM directionally and stays higher throughout, same pattern as exp3.
**Decision:** Proceed to exp6 (reasoning arm, same model) for the paired comparison at 4B.

**Links**
- Settings: `experiment/exp5/settings.json`
- Raw generations: `experiment/exp5/outputs/`
- Eval results: `experiment/exp5/eval/`
- Summary: `experiment/exp5/summary.json`

---
id: E006
run_dir: experiment/exp8
condition: reasoning
model: Qwen/Qwen3.5-9B
datasets: [nq, triviaqa, popqa, hotpotqa, 2wikimultihopqa, musique, bamboogle, simpleqa_verified]
n_examples: 500
status: Done
---

# E006 — Direct answer versus reasoning (Qwen3.5-9B, reasoning arm)

**Owner:** Michelle Sheu
**Why:** Paired comparison to `exp7`.
**Setup:** Same as `exp7` except `condition: reasoning`. Decoding per Qwen3.5
card's thinking-mode general-tasks recommendation: temp 1.0, top_p 0.95,
top_k 20, min_p 0, presence_penalty 1.5, rep_penalty 1.0, max 32768 tokens.
Largest/slowest run in the sweep — run last.
**Expected result:** Reasoning should improve EM/cover-EM/judge_accuracy/
ex_recall over `exp7`, especially on multi-hop datasets.
**Result:**

| dataset | n | em | cover_em | judge_acc | ex_recall | avg_output_len | extraction_fail_rate |
|---|---|---|---|---|---|---|---|
| nq | 500 | 0.218 | 0.296 | 0.428 | 0.282 | 2090 | 0.000 |
| triviaqa | 500 | 0.566 | 0.630 | 0.704 | 0.626 | 1837 | 0.006 |
| popqa | 500 | 0.224 | 0.258 | 0.282 | 0.244 | 3206 | 0.002 |
| hotpotqa | 500 | 0.258 | 0.296 | 0.444 | 0.288 | 3124 | 0.004 |
| 2wikimultihopqa | 500 | 0.296 | 0.318 | 0.340 | 0.308 | 3775 | 0.002 |
| musique | 500 | 0.096 | 0.114 | 0.176 | 0.104 | 4344 | 0.012 |
| bamboogle | 125 | 0.440 | 0.480 | 0.616 | 0.472 | 2157 | 0.000 |

simpleqa_verified (llm_judge, n=500, separate): overall_accuracy 0.116, accuracy_given_attempted 0.120, f1 0.118, not_attempted 17, extraction_fail_rate 0.002.

**What we learned:** Best result in the whole sweep by every metric — reasoning + largest model size compound: EM up across the board vs `exp7` (musique 0.038→0.096, more than doubling and its best showing yet; triviaqa 0.510→0.566; bamboogle 0.392→0.440). Also the best result vs the smaller reasoning arms (exp4 2B, exp6 4B), so the reasoning benefit appears to *grow* with scale here rather than saturate, unlike what exp5→exp7's direct-arm scaling suggested for nq. Extraction-failure rate near zero everywhere (max 1.2%, on musique — down from 47% in exp7), same formatting-compliance confound as exp4/exp6.
**Decision:** Reasoning + scale both help and compound at 9B, the largest Qwen3.5 tested. Proceed to exp9/exp10 (Gemma-4-E4B-it) to test whether the reasoning benefit is Qwen-specific or generalizes cross-family.

**Links**
- Settings: `experiment/exp8/settings.json`
- Raw generations: `experiment/exp8/outputs/`
- Eval results: `experiment/exp8/eval/`
- Summary: `experiment/exp8/summary.json`

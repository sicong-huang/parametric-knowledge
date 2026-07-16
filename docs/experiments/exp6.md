---
id: E005
run_dir: experiment/exp6
condition: reasoning
model: Qwen/Qwen3.5-4B
datasets: [nq, triviaqa, popqa, hotpotqa, 2wikimultihopqa, musique, bamboogle, simpleqa_verified]
n_examples: 500
status: Done
---

# E005 — Direct answer versus reasoning (Qwen3.5-4B, reasoning arm)

**Owner:** Michelle Sheu
**Why:** Paired comparison to `exp5`.
**Setup:** Same as `exp5` except `condition: reasoning`. Decoding per Qwen3.5
card's thinking-mode general-tasks recommendation: temp 1.0, top_p 0.95,
top_k 20, min_p 0, presence_penalty 1.5, rep_penalty 1.0, max 32768 tokens.
**Expected result:** Reasoning should improve EM/cover-EM/judge_accuracy/
ex_recall over `exp5`, especially on multi-hop datasets.
**Result:**

| dataset | n | em | cover_em | judge_acc | ex_recall | avg_output_len | extraction_fail_rate |
|---|---|---|---|---|---|---|---|
| nq | 500 | 0.182 | 0.250 | 0.366 | 0.234 | 2844 | 0.002 |
| triviaqa | 500 | 0.460 | 0.528 | 0.600 | 0.518 | 2519 | 0.004 |
| popqa | 500 | 0.196 | 0.224 | 0.244 | 0.212 | 4040 | 0.000 |
| hotpotqa | 500 | 0.216 | 0.252 | 0.350 | 0.234 | 3811 | 0.008 |
| 2wikimultihopqa | 500 | 0.268 | 0.300 | 0.322 | 0.280 | 4022 | 0.000 |
| musique | 500 | 0.052 | 0.082 | 0.120 | 0.066 | 5064 | 0.006 |
| bamboogle | 125 | 0.336 | 0.376 | 0.472 | 0.368 | 2779 | 0.000 |

simpleqa_verified (llm_judge, n=500, separate): overall_accuracy 0.074, accuracy_given_attempted 0.077, f1 0.076, not_attempted 22, extraction_fail_rate 0.014.

**What we learned:** Reasoning helps at 4B same as it did at 2B (`exp3`→`exp4`): every dataset improves over `exp5` (e.g. musique EM 0.024→0.052, more than doubling despite still being the weakest dataset; bamboogle 0.272→0.336). The reasoning boost looks roughly similar in relative size to the 2B case, not obviously growing with scale. Extraction-failure rate again collapses to near-zero (was up to 46% on musique direct-arm, now 0.6%), reinforcing the exp4 finding that reasoning's apparent EM lift is partly a formatting-compliance effect, not purely recall. This is now the strongest EM/judge_accuracy result seen so far in the sweep.
**Decision:** Consistent reasoning benefit across two model sizes now; proceed to exp7/exp8 (9B) to check whether the effect holds, shrinks, or grows at the largest size in the sweep.

**Links**
- Settings: `experiment/exp6/settings.json`
- Raw generations: `experiment/exp6/outputs/`
- Eval results: `experiment/exp6/eval/`
- Summary: `experiment/exp6/summary.json`

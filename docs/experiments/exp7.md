---
id: E006
run_dir: experiment/exp7
condition: direct
model: Qwen/Qwen3.5-9B
datasets: [nq, triviaqa, popqa, hotpotqa, 2wikimultihopqa, musique, bamboogle, simpleqa_verified]
n_examples: 500
status: Done
---

# E006 — Direct answer versus reasoning (Qwen3.5-9B, direct arm)

**Owner:** Michelle Sheu
**Why:** Premise 1 sweep — largest Qwen3.5 size, run last (heaviest GPU job).
**Setup:** Qwen3.5-9B, vLLM, `enable_thinking=False`. Decoding per Qwen3.5
card's "Instruct (non-thinking) mode, general tasks": temp 0.7, top_p 0.8,
top_k 20, min_p 0, presence_penalty 1.5, rep_penalty 1.0, max 500 tokens.
Same 8 datasets/metrics/API requirements as `exp3`.
**Expected result:** Baseline accuracy per dataset; comparison point for
`exp8` (reasoning arm, same model).
**Result:**

| dataset | n | em | cover_em | judge_acc | ex_recall | avg_output_len | extraction_fail_rate |
|---|---|---|---|---|---|---|---|
| nq | 500 | 0.182 | 0.300 | 0.418 | 0.274 | 64.5 | 0.062 |
| triviaqa | 500 | 0.510 | 0.586 | 0.632 | 0.566 | 75.5 | 0.112 |
| popqa | 500 | 0.182 | 0.230 | 0.262 | 0.228 | 47.0 | 0.054 |
| hotpotqa | 500 | 0.228 | 0.298 | 0.374 | 0.266 | 133.8 | 0.272 |
| 2wikimultihopqa | 500 | 0.248 | 0.338 | 0.292 | 0.314 | 117.3 | 0.180 |
| musique | 500 | 0.038 | 0.098 | 0.110 | 0.062 | 201.1 | 0.470 |
| bamboogle | 125 | 0.392 | 0.496 | 0.536 | 0.480 | 111.4 | 0.224 |

simpleqa_verified (llm_judge, n=500, separate): overall_accuracy 0.094, accuracy_given_attempted 0.101, f1 0.097, not_attempted 33, extraction_fail_rate 0.128.
Note: this run's eval step was interrupted mid-way by a transient judge-server restart (unrelated `--max-model-len` fix rollout, see `project_doc.md` E007 judge infra note) and was fully rerun end-to-end.

**What we learned:** Direct-arm scaling continues cleanly from 2B (exp3) → 4B (exp5) → 9B (exp7) on most datasets — triviaqa EM 0.152→0.442→0.510, nq 0.074→0.178→0.182 (nq nearly flat 4B→9B, diminishing returns). musique stays the outlier: EM only 0.038 at 9B despite 3x the parameters of exp3, with extraction-failure rate still ~47%, reinforcing that this dataset's difficulty is largely independent of raw scale.
**Decision:** Proceed to exp8 (reasoning arm, same model) for the paired comparison at the largest size in the sweep.

**Links**
- Settings: `experiment/exp7/settings.json`
- Raw generations: `experiment/exp7/outputs/`
- Eval results: `experiment/exp7/eval/`
- Summary: `experiment/exp7/summary.json`

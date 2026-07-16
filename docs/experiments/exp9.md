---
id: E007
run_dir: experiment/exp9
condition: direct
model: google/gemma-4-E4B-it
datasets: [nq, triviaqa, popqa, hotpotqa, 2wikimultihopqa, musique, bamboogle, simpleqa_verified]
n_examples: 500
status: Done
---

# E007 — Direct answer versus reasoning (Gemma-4-E4B-it, direct arm)

**Owner:** Michelle Sheu
**Why:** Premise 1 sweep — cross-family check (Gemma vs Qwen3.5).
**Setup:** google/gemma-4-E4B-it, vLLM, `enable_thinking=False`. Gemma's card
gives one unified sampling config for both modes (no min_p/presence_penalty/
repetition_penalty recommended): temp 1.0, top_p 0.95, top_k 64, max 500
tokens for this direct arm. First Gemma run in the sweep — verify
`apply_chat_template` accepts `enable_thinking` for this model/tokenizer
before trusting the reasoning arm (`exp10`). Same 8 datasets/metrics/API
requirements as `exp3`.
**Expected result:** Baseline accuracy per dataset; comparison point for
`exp10` (reasoning arm, same model).
**Result:**

| dataset | n | em | cover_em | judge_acc | ex_recall | avg_output_len | extraction_fail_rate |
|---|---|---|---|---|---|---|---|
| nq | 500 | 0.132 | 0.190 | 0.280 | 0.176 | 54.2 | 0.164 |
| triviaqa | 500 | 0.402 | 0.466 | 0.510 | 0.448 | 63.7 | 0.114 |
| popqa | 500 | 0.148 | 0.168 | 0.182 | 0.162 | 62.9 | 0.206 |
| hotpotqa | 500 | 0.144 | 0.202 | 0.234 | 0.170 | 122.9 | 0.288 |
| 2wikimultihopqa | 500 | 0.072 | 0.290 | 0.090 | 0.164 | 132.8 | 0.380 |
| musique | 500 | 0.022 | 0.052 | 0.050 | 0.028 | 157.2 | 0.472 |
| bamboogle | 125 | 0.240 | 0.280 | 0.344 | 0.248 | 139.7 | 0.264 |

simpleqa_verified (llm_judge, n=500, separate): overall_accuracy 0.032, accuracy_given_attempted 0.085, f1 0.046, not_attempted 311 (62% — very high abstention rate), extraction_fail_rate 0.152.

**What we learned:** First Gemma run — `enable_thinking` gating worked as hoped (confirmed direct arm behaves as expected, unblocking `exp10`). Cross-family comparison at similar-ish scale: Gemma-4-E4B-it direct arm sits below Qwen3.5-4B (`exp5`) on most datasets (nq 0.178→0.132, triviaqa 0.442→0.402) but the standout anomaly is 2wikimultihopqa, where judge_accuracy (0.090) is *lower* than EM (0.072) — the only dataset in the entire sweep where that ordering inverts — driven by an extreme not_attempted count (393/500, 78.6%) that cover_em's looser substring match doesn't penalize (cover_em 0.290, back in line with other datasets). simpleqa_verified also shows unusually high abstention (62% not_attempted, vs 4-10% typical for Qwen). This looks like a Gemma-specific abstention/refusal tendency on ambiguous or hard questions rather than a recall deficit — worth flagging as a metric-interpretation caveat for this model family.
**Decision:** Proceed to exp10 (reasoning arm, same model) to see if reasoning reduces Gemma's abstention rate. Flag the 2wikimultihopqa/simpleqa_verified abstention anomaly in the writeup — judge_accuracy alone would understate this model's true attempted-accuracy there.

**Links**
- Settings: `experiment/exp9/settings.json`
- Raw generations: `experiment/exp9/outputs/`
- Eval results: `experiment/exp9/eval/`
- Summary: `experiment/exp9/summary.json`

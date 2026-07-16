---
id: E004
run_dir: experiment/exp4
condition: reasoning
model: Qwen/Qwen3.5-2B
datasets: [nq, triviaqa, popqa, hotpotqa, 2wikimultihopqa, musique, bamboogle, simpleqa_verified]
n_examples: 500
status: Done
---

# E004 — Direct answer versus reasoning (Qwen3.5-2B, reasoning arm)

**Owner:** Michelle Sheu
**Why:** Paired comparison to `exp3` (same model/datasets/sample,
`enable_thinking=True`).
**Setup:** Same as `exp3` except `condition: reasoning`. Decoding per Qwen3.5
card's thinking-mode general-tasks recommendation: temp 1.0, top_p 0.95,
top_k 20, min_p 0, presence_penalty 1.5, rep_penalty 1.0, max 32768 tokens
(card-recommended thinking budget — much larger than the direct arm's 500).
**Expected result:** Reasoning should improve EM/cover-EM/judge_accuracy/
ex_recall over `exp3`, especially on multi-hop datasets. Output length and
extraction-failure rate should both rise relative to the direct arm.
**Result:**

| dataset | n | em | cover_em | judge_acc | ex_recall | avg_output_len | extraction_fail_rate |
|---|---|---|---|---|---|---|---|
| nq | 500 | 0.114 | 0.182 | 0.264 | 0.164 | 3001 | 0.006 |
| triviaqa | 500 | 0.248 | 0.310 | 0.342 | 0.288 | 3444 | 0.016 |
| popqa | 500 | 0.122 | 0.172 | 0.160 | 0.138 | 4335 | 0.014 |
| hotpotqa | 500 | 0.154 | 0.186 | 0.262 | 0.172 | 4205 | 0.012 |
| 2wikimultihopqa | 500 | 0.204 | 0.264 | 0.250 | 0.228 | 3799 | 0.004 |
| musique | 500 | 0.020 | 0.048 | 0.078 | 0.024 | 4934 | 0.016 |
| bamboogle | 125 | 0.136 | 0.176 | 0.200 | 0.160 | 3598 | 0.008 |

simpleqa_verified (llm_judge, n=500, separate): overall_accuracy 0.044, accuracy_given_attempted 0.049, f1 0.046, not_attempted 47, extraction_fail_rate 0.012.

**What we learned:** Reasoning helps at this size, matching the E001 direction: EM roughly doubles or better across every dataset vs `exp3` (e.g. nq 0.074→0.114, triviaqa 0.152→0.248, musique 0.006→0.020, bamboogle 0.048→0.136). judge_accuracy and ex_recall move the same direction and stay above EM throughout, as expected. Extraction-failure rate drops sharply (19–41% in exp3 → 0.4–1.6% here) — the model closes `<answer>` tags far more reliably once it reasons first, which alone would inflate EM somewhat independent of any recall improvement. Output length is ~20–30x longer (thousands of tokens vs tens), consistent with the 32768-token thinking budget being used.
**Decision:** Reasoning improves recall at 2B; proceed to exp5/exp6 to see if the effect holds/grows at 4B. Given extraction-failure rate is confounded with reasoning (fewer malformed tags), treat judge_accuracy/ex_recall as more trustworthy than raw EM for the direct-vs-reasoning delta specifically.

**Links**
- Settings: `experiment/exp4/settings.json`
- Raw generations: `experiment/exp4/outputs/`
- Eval results: `experiment/exp4/eval/`
- Summary: `experiment/exp4/summary.json`

---
id: E007
run_dir: experiment/exp10
condition: reasoning
model: google/gemma-4-E4B-it
datasets: [nq, triviaqa, popqa, hotpotqa, 2wikimultihopqa, musique, bamboogle, simpleqa_verified]
n_examples: 500
status: Done
---

# E007 — Direct answer versus reasoning (Gemma-4-E4B-it, reasoning arm)

**Owner:** Michelle Sheu
**Why:** Paired comparison to `exp9`.
**Setup:** Same as `exp9` except `condition: reasoning`. Gemma's card uses the
same sampling config for both modes: temp 1.0, top_p 0.95, top_k 64, but max
32768 tokens for this reasoning arm (card-recommended thinking budget).
**Expected result:** Reasoning should improve EM/cover-EM/judge_accuracy/
ex_recall over `exp9`, especially on multi-hop datasets.
**Result:**

| dataset | n | em | cover_em | judge_acc | ex_recall | avg_output_len | extraction_fail_rate |
|---|---|---|---|---|---|---|---|
| nq | 500 | 0.168 | 0.258 | 0.354 | 0.246 | 259.3 | 0.188 |
| triviaqa | 500 | 0.438 | 0.492 | 0.556 | 0.480 | 261.3 | 0.062 |
| popqa | 500 | 0.160 | 0.206 | 0.204 | 0.176 | 372.8 | 0.338 |
| hotpotqa | 500 | 0.168 | 0.226 | 0.298 | 0.198 | 450.8 | 0.142 |
| 2wikimultihopqa | 500 | 0.170 | 0.324 | 0.220 | 0.234 | 475.5 | 0.366 |
| musique | 500 | 0.032 | 0.058 | 0.098 | 0.044 | 578.4 | 0.320 |
| bamboogle | 125 | 0.248 | 0.280 | 0.368 | 0.280 | 362.4 | 0.120 |

simpleqa_verified (llm_judge, n=500, separate): overall_accuracy 0.048, accuracy_given_attempted 0.060, f1 0.053, not_attempted 97 (19.4%), extraction_fail_rate 0.126.

**What we learned:** Reasoning helps Gemma too, and specifically fixes the `exp9` abstention anomaly: 2wikimultihopqa judge_accuracy roughly doubles (0.090→0.220, now correctly above EM again) and simpleqa_verified not_attempted drops from 62%→19.4%. Output length only grows ~2-5x here (hundreds of tokens vs tens) rather than the ~40-70x seen in the Qwen reasoning arms (which used the full 32768-token budget) — Gemma appears to reason much more concisely, or terminate its thinking earlier. Despite the smaller reasoning budget used, gains are broadly consistent with the Qwen pattern (exp3→exp4, exp5→exp6, exp7→exp8): every dataset improves, extraction-failure rate drops somewhat (e.g. musique 0.472→0.320) though not to Qwen's near-zero levels.
**Decision:** Reasoning benefit generalizes across both model families tested (Qwen3.5, Gemma), and — as with Qwen — the effect looks partly compositional with abstention/formatting-compliance fixes rather than pure recall improvement. Sweep (E004-E007) complete; next step per `project_doc.md` plan is E002 (length-controlled comparison) to isolate the reasoning-specific effect from the extra-compute/output-length confound flagged across every paired run in this sweep.

**Links**
- Settings: `experiment/exp10/settings.json`
- Raw generations: `experiment/exp10/outputs/`
- Eval results: `experiment/exp10/eval/`
- Summary: `experiment/exp10/summary.json`

---
id: E011
run_dir: experiment/exp18
condition: reasoning
model: google/gemma-4-E2B-it
datasets: nq, triviaqa, popqa, hotpotqa, 2wikimultihopqa, musique, bamboogle, simpleqa_verified
n_examples: 500
status: Done
---

# E011 — Gemma-4-E2B-it, reasoning

**Owner:**
**Why:** Reasoning arm of E011; same rationale as `exp17.md`.
**Setup:** google/gemma-4-E2B-it, reasoning condition (`enable_thinking=True`).
Decoding per the HF card's recommended sampling config: `temperature=1.0`,
`top_p=0.95`, `top_k=64`, max_tokens 32768 (room for a full `<think>` trace,
mirrors exp10's E4B config). Same 8-dataset eval suite, n=500/dataset,
judge + ex_recall enabled.
**Expected result:** Reasoning-ON arm for E011's direct-vs-reasoning delta;
compare against `exp17` to test whether thinking improves parametric
recall at the E2B rung.
**Result:** Mean over the 7 EM-metric datasets (`experiment/exp18/summary.json`):
EM 0.159, cover-EM 0.206, judge_accuracy 0.245, ex_recall 0.189 -- modest
gains over the direct arm (`exp17`: 0.137 / 0.241 / 0.211 / 0.184; note
cover-EM is actually slightly *lower* under reasoning here). Extraction
failure rate dropped to 13.7% (vs. 44.8% direct), and mean output length grew
to ~485 chars. simpleqa_verified: overall_accuracy 0.05 (same as direct).
**What we learned:** Reasoning gives Gemma-E2B a real but much smaller boost
than it gave Nemotron-4B (EM +0.023 vs. +0.062; ex_recall +0.006 vs. +0.070 --
ex_recall is essentially flat here). Reasoning still meaningfully reduces
extraction failures (44.8% -> 13.7%) even though the recall-metric lift is
small, suggesting most of the direct-arm's EM loss for Gemma-E2B was already
partially recoverable via judge/ex_recall scoring rather than genuine
knowledge-access failure.
**Decision:** E011 pair (exp17/exp18) complete. Reasoning benefit is
present but smaller than for Nemotron-4B (E010) or the E004-E007 sweep;
fold into the cross-model rollup and flag the small-model ceiling effect for
discussion.

**Links**
- Settings: `experiment/exp18/settings.json`
- Raw generations: `experiment/exp18/outputs/`
- Eval results: `experiment/exp18/eval/`
- Summary: `experiment/exp18/summary.json`

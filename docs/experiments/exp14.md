---
id: E009
run_dir: experiment/exp14
condition: reasoning
model: nvidia/NVIDIA-Nemotron-Nano-9B-v2
datasets: nq, triviaqa, popqa, hotpotqa, 2wikimultihopqa, musique, bamboogle, simpleqa_verified
n_examples: 500
status: Done
---

# E009 — Nemotron-Nano-9B-v2, reasoning

**Owner:**
**Why:** Reasoning-arm counterpart to exp13 (see exp13.md for the model-ladder and
toggle-mechanism rationale).
**Setup:** nvidia/NVIDIA-Nemotron-Nano-9B-v2, reasoning condition (`toggle_style:
sysprompt`, appends `/think`). Decoding per NVIDIA's HF-card recommendation for
reasoning=True: temp 0.6 / top_p 0.95 / max_tokens 32768. Note: NVIDIA's own
`min_thinking_tokens`/`max_thinking_tokens` budget controls are not usable in this repo's
offline vLLM `LLM.generate` path (not standard `SamplingParams` fields; served-endpoint-only
feature) -- relying on `max_tokens` as the sole length cap, so a runaway trace can in
principle exhaust the budget before an `<answer>` tag appears. Watch for this in the
smoke test and note truncated predictions if seen. Same 8-dataset eval suite, n=500/dataset,
judge + ex_recall enabled.
**Expected result:** Test whether reasoning improves recall at the top of the ladder;
compare trace verbosity (avg_output_len) against SmolLM3 (exp12), Qwen3.5-4B (exp6), and
Gemma-4-E4B (exp10) reasoning arms.
**Result:** Mean over the 7 EM-metric datasets (`experiment/exp14/summary.json`):
EM 0.242, cover-EM 0.268, judge_accuracy 0.350, ex_recall 0.263 -- the largest
EM/judge/ex_recall lift of any pair run in this project so far (+0.077 EM,
+0.097 judge_accuracy, +0.077 ex_recall over `exp13`). No `min_thinking_tokens`/
`max_thinking_tokens` truncation issue observed -- extraction failure rate
actually dropped further, to 0.3%. bamboogle showed the single largest
per-dataset jump of the whole sweep (EM 0.064 -> 0.336). simpleqa_verified:
overall_accuracy 0.074 (vs. 0.062 direct).
**What we learned:** Reasoning helps most at this end of the model-size
ladder -- consistent with the earlier E004-E006 Qwen3.5 finding that
reasoning benefit does not saturate with scale, now confirmed cross-family
on Nemotron. Since exp13's direct arm was already well-formatted, this lift
is mostly a genuine recall gain, not a formatting-fix artifact (contrast
with Nemotron-4B/Gemma-E2B in exp15-18, where much of the reasoning benefit
tracked fixing extraction failures).
**Decision:** E009 pair (exp13/exp14) complete. Strongest reasoning benefit
of the whole 4B-9B sweep; fold into the cross-model rollup in
`project_doc.md` and the combined report.

**Links**
- Settings: `experiment/exp14/settings.json`
- Raw generations: `experiment/exp14/outputs/`
- Eval results: `experiment/exp14/eval/`
- Summary: `experiment/exp14/summary.json`

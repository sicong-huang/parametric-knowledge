---
id: E010
run_dir: experiment/exp16
condition: reasoning
model: nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16
datasets: nq, triviaqa, popqa, hotpotqa, 2wikimultihopqa, musique, bamboogle, simpleqa_verified
n_examples: 500
status: Done
---

# E010 — Nemotron-3-Nano-4B, reasoning

**Owner:**
**Why:** Reasoning arm of E010; same rationale as `exp15.md`.
**Setup:** nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16, reasoning condition
(`enable_thinking=True`, the model's default). Decoding per the HF card's
recommended sampling config: `temperature=1.0`, `top_p=0.95`, max_tokens
32768 (room for a full `<think>` trace). Same 8-dataset eval suite, n=500/
dataset, judge + ex_recall enabled.
**Expected result:** Reasoning-ON arm for E010's direct-vs-reasoning delta;
compare against `exp15` to test whether thinking improves parametric
recall at the 4B NVIDIA rung.
**Result:** Mean over the 7 EM-metric datasets (`experiment/exp16/summary.json`):
EM 0.165, cover-EM 0.188, judge_accuracy 0.244, ex_recall 0.183 -- all higher
than the direct arm (`exp15`: 0.103 / 0.116 / 0.150 / 0.113). Extraction
failure rate dropped to 2.4% (vs. 43.8% direct) and mean output length rose
to ~249 chars (vs. ~2.8 direct), i.e. a real `<think>` trace instead of a bare
word. simpleqa_verified: overall_accuracy 0.03 (same as direct; not averaged
into the EM table).
**What we learned:** Reasoning gives Nemotron-3-Nano-4B a clear boost on every
co-primary metric (EM +0.062, judge_accuracy +0.094, ex_recall +0.070) and
fixes the direct arm's extraction-tag problem almost entirely. This is the
strongest EM delta of the 4B-class comparisons run so far.
**Decision:** E010 pair (exp15/exp16) complete. Reasoning clearly helps
parametric recall for this model; fold into the cross-model rollup alongside
E004-E007 in `project_doc.md`.

**Links**
- Settings: `experiment/exp16/settings.json`
- Raw generations: `experiment/exp16/outputs/`
- Eval results: `experiment/exp16/eval/`
- Summary: `experiment/exp16/summary.json`

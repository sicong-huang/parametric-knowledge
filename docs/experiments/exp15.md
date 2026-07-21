---
id: E010
run_dir: experiment/exp15
condition: direct
model: nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16
datasets: nq, triviaqa, popqa, hotpotqa, 2wikimultihopqa, musique, bamboogle, simpleqa_verified
n_examples: 500
status: Done
---

# E010 — Nemotron-3-Nano-4B, direct

**Owner:**
**Why:** Completes the fallback four-model sweep named in `project_doc.md`'s
E008 decision (Nemotron-Nano 4B/9B + Gemma-4 E2B/E4B) -- the smaller NVIDIA
rung, pairing with the already-planned Nemotron-Nano-9B-v2 (E009). There is
no Nemotron-Nano-4B-v2; this uses the Nemotron-3-Nano-4B family instead
(BF16 checkpoint). Verified via the HF card that, unlike the 9B-v2, this
model's `apply_chat_template` *does* honor the `enable_thinking` kwarg
(`enable_thinking=False` disables reasoning; default is `True`) -- no
`toggle_style` override needed, default `kwarg` path applies.
**Setup:** nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16, direct condition
(`enable_thinking=False`). Decoding per the HF card's recommended sampling
config: `temperature=1.0`, `top_p=0.95`, max_tokens 500. Same 8-dataset eval
suite, n=500/dataset, judge + ex_recall enabled.
**Expected result:** Baseline arm for E010's direct-vs-reasoning delta; a
smaller-scale NVIDIA-family counterpart to E009 (9B).
**Result:** Mean over the 7 EM-metric datasets (`experiment/exp15/summary.json`):
EM 0.103, cover-EM 0.116, judge_accuracy 0.150, ex_recall 0.113. Extraction
failure rate averaged 43.8% (as high as 78% on musique) and mean output length
was just 2.8 chars -- the model answers almost bare, often without the
`<answer>` tags the prompt asks for. simpleqa_verified: overall_accuracy 0.03
(not averaged into the EM table per CLAUDE.md convention).
**What we learned:** Direct-mode Nemotron-3-Nano-4B is extremely terse and
frequently skips the `<answer>` tag entirely, which is why EM/cover-EM
undercount its true recall relative to judge_accuracy/ex_recall (both
higher). See `exp16.md` for the reasoning-arm comparison -- reasoning nearly
eliminates this extraction problem (failure rate drops to 2.4%).
**Decision:** Compare against `exp16` for the E010 direct-vs-reasoning delta;
treat judge_accuracy/ex_recall as more trustworthy than raw EM for this arm
given the extraction-failure gap.

**Links**
- Settings: `experiment/exp15/settings.json`
- Raw generations: `experiment/exp15/outputs/`
- Eval results: `experiment/exp15/eval/`
- Summary: `experiment/exp15/summary.json`

---
id: E011
run_dir: experiment/exp17
condition: direct
model: google/gemma-4-E2B-it
datasets: nq, triviaqa, popqa, hotpotqa, 2wikimultihopqa, musique, bamboogle, simpleqa_verified
n_examples: 500
status: Done
---

# E011 — Gemma-4-E2B-it, direct

**Owner:**
**Why:** Completes the fallback four-model sweep named in `project_doc.md`'s
E008 decision (Nemotron-Nano 4B/9B + Gemma-4 E2B/E4B) -- the smaller Gemma
rung, pairing with the already-run Gemma-4-E4B-it (E007). The HF card
describes reasoning as toggled by a `<|think|>` control token, but E2B and
E4B ship the byte-identical `chat_template.jinja` (confirmed: both 18567
chars, both implement `enable_thinking` by inserting `<|think|>`) -- so the
standard `enable_thinking` kwarg drives it exactly as it does for E4B in
exp9/exp10. No `toggle_style` override needed.
**Setup:** google/gemma-4-E2B-it, direct condition (`enable_thinking=False`).
Decoding per the HF card's recommended sampling config: `temperature=1.0`,
`top_p=0.95`, `top_k=64`, max_tokens 500 (mirrors exp9's E4B config). Same
8-dataset eval suite, n=500/dataset, judge + ex_recall enabled.
**Expected result:** Baseline arm for E011's direct-vs-reasoning delta; a
smaller-scale Gemma-4 counterpart to E007 (E4B).
**Result:** Mean over the 7 EM-metric datasets (`experiment/exp17/summary.json`):
EM 0.137, cover-EM 0.241, judge_accuracy 0.211, ex_recall 0.184. Extraction
failure rate averaged 44.8% (up to 78% on musique) despite a much longer
mean output (~212 chars) than Nemotron's direct arm -- Gemma-E2B direct tends
to over-explain without emitting the `<answer>` tag. simpleqa_verified:
overall_accuracy 0.05 (not averaged into the EM table).
**What we learned:** Unlike Nemotron, Gemma-E2B's direct-mode extraction
problem isn't verbosity-driven brevity but a formatting habit -- it writes a
full explanation and still often skips the tag. See `exp18.md` for the
reasoning-arm comparison.
**Decision:** Compare against `exp18` for the E011 direct-vs-reasoning delta.

**Links**
- Settings: `experiment/exp17/settings.json`
- Raw generations: `experiment/exp17/outputs/`
- Eval results: `experiment/exp17/eval/`
- Summary: `experiment/exp17/summary.json`

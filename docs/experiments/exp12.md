---
id: E008
run_dir: experiment/exp12
condition: reasoning
model: HuggingFaceTB/SmolLM3-3B
datasets: nq, triviaqa, popqa, hotpotqa, 2wikimultihopqa, musique, bamboogle, simpleqa_verified
n_examples: 500
status: Planned
---

# E008 — SmolLM3-3B, reasoning

**Owner:**
**Why:** Reasoning-arm counterpart to exp11 (see exp11.md for the model-ladder rationale).
**Setup:** HuggingFaceTB/SmolLM3-3B, reasoning condition (`toggle_style: sysprompt`, appends
`/think` to the system prompt). Decoding: temp 0.6 / top_p 0.95 / max_tokens 32768 (per HF
card's thinking-mode recommendation). Same 8-dataset eval suite, n=500/dataset, judge +
ex_recall enabled.
**Expected result:** Higher EM/judge-accuracy than exp11 if reasoning helps parametric
recall at this size; also expect a much shorter trace than Qwen3.5-4B's reasoning arm
(exp6) given SmolLM3's smaller scale, worth comparing against exp6's verbosity.
**Result:** (fill in from `experiment/exp12/summary.json`)
**What we learned:**
**Decision:**

**Links**
- Settings: `experiment/exp12/settings.json`
- Raw generations: `experiment/exp12/outputs/`
- Eval results: `experiment/exp12/eval/`
- Summary: `experiment/exp12/summary.json`

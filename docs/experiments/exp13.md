---
id: E009
run_dir: experiment/exp13
condition: direct
model: nvidia/NVIDIA-Nemotron-Nano-9B-v2
datasets: nq, triviaqa, popqa, hotpotqa, 2wikimultihopqa, musique, bamboogle, simpleqa_verified
n_examples: 500
status: Done
---

# E009 — Nemotron-Nano-9B-v2, direct

**Owner:**
**Why:** Largest rung of the cross-family model-size ladder (SmolLM3-3B, Qwen3.5-4B,
gemma-4-E4B-it, Nemotron-Nano-9B-v2) and the only NVIDIA-family entry. Verified this
model's `apply_chat_template` does **not** honor the `enable_thinking` kwarg (would
silently no-op) -- reasoning is toggled only via a `/think`/`/no_think` token in the
system/user message, default reasoning-ON. `generate.py` gained `toggle_style: sysprompt`
support for this.
**Setup:** nvidia/NVIDIA-Nemotron-Nano-9B-v2, direct condition (`toggle_style: sysprompt`,
appends `/no_think`). Decoding per NVIDIA's HF-card recommendation for reasoning=False:
greedy (`temperature: 0`), max_tokens 500. Note: greedy contradicts CLAUDE.md's
Qwen-specific "never use greedy" guidance -- that rule doesn't apply to this model family.
Same 8-dataset eval suite, n=500/dataset, judge + ex_recall enabled.
**Expected result:** Baseline arm for E009's direct-vs-reasoning delta; largest model in
the ladder, so expect the strongest direct-condition EM if scale helps parametric recall.
**Result:** Mean over the 7 EM-metric datasets (`experiment/exp13/summary.json`):
EM 0.166, cover-EM 0.190, judge_accuracy 0.253, ex_recall 0.187. Extraction
failure rate averaged just 2.0% (vs. 43.8% for Nemotron-4B's direct arm in
`exp15.md`) -- the 9B model reliably emits the `<answer>` tag even without
reasoning, unlike its 4B sibling. simpleqa_verified: overall_accuracy 0.062
(not averaged into the EM table).
**What we learned:** Nemotron-9B's direct-mode formatting is already solid,
so its EM/judge/ex_recall gap here is small and mostly reflects genuine
recall rather than tag-extraction noise -- a cleaner baseline than
Nemotron-4B's direct arm for isolating the pure reasoning effect.
**Decision:** Compare against `exp14` for the E009 direct-vs-reasoning delta.

**Links**
- Settings: `experiment/exp13/settings.json`
- Raw generations: `experiment/exp13/outputs/`
- Eval results: `experiment/exp13/eval/`
- Summary: `experiment/exp13/summary.json`

---
id: E008
run_dir: experiment/exp11
condition: direct
model: HuggingFaceTB/SmolLM3-3B
datasets: nq, triviaqa, popqa, hotpotqa, 2wikimultihopqa, musique, bamboogle, simpleqa_verified
n_examples: 500
status: Planned
---

# E008 — SmolLM3-3B, direct

**Owner:**
**Why:** Cross-family model-size ladder (SmolLM3-3B, Qwen3.5-4B, gemma-4-E4B-it,
Nemotron-Nano-9B-v2) so the direct-vs-reasoning comparison isn't a Qwen-only artifact.
SmolLM3-3B is the smallest rung and the only HuggingFace-family entry. Toggle mechanism
verified to be genuine (own trained thinking/non-thinking mode), not a separate distill.
**Setup:** HuggingFaceTB/SmolLM3-3B, direct condition (`toggle_style: sysprompt`, appends
`/no_think` to the system prompt per HF card convention — this model does not reliably
honor the `enable_thinking` chat-template kwarg alone when compared to the sysprompt
route). Decoding: temp 0.7 / top_p 0.8 / max_tokens 500. Same 8-dataset eval suite,
n=500/dataset, judge + ex_recall enabled against the local gemma-4-31b-it judge.
**Expected result:** Comparable or somewhat lower EM than Qwen3.5-4B given smaller size;
serves as the baseline arm for E008's direct-vs-reasoning delta.
**Result:** (fill in from `experiment/exp11/summary.json`)
**What we learned:**
**Decision:**

**Links**
- Settings: `experiment/exp11/settings.json`
- Raw generations: `experiment/exp11/outputs/`
- Eval results: `experiment/exp11/eval/`
- Summary: `experiment/exp11/summary.json`

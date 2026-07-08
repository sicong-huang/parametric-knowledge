---
id: E001
run_dir: experiment/exp2
condition: reasoning
model: Qwen/Qwen3-4B
datasets: [nq, triviaqa, popqa, hotpotqa, 2wikimultihopqa, musique, bamboogle]
n_examples: 500
status: Planned
---

# E001 — Direct answer versus reasoning (reasoning arm)

**Owner:** Michelle Sheu
**Why:** RQ1 pair to `exp1`: same model/datasets/sample, `enable_thinking=True`
instead of direct answering. Not yet created -- copy `experiment/exp1/` to
`experiment/exp2/`, set `condition: "reasoning"` in `settings.json`, run via
`bash scripts/run_exp.sh exp2`.
**Setup:** Same as `exp1` except `condition: reasoning`.
**Expected result:** Reasoning should improve EM/cover-EM over `exp1`,
especially on multi-hop datasets (hotpotqa, 2wikimultihopqa, musique,
bamboogle). Per RQ2, average output length will also increase -- E002
(length-controlled comparison) is needed to separate a reasoning effect from
a pure additional-computation effect.
**Result:** _(not yet run)_
**What we learned:**
**Decision:**

**Links**
- Settings: `experiment/exp2/settings.json` (to be created)
- Raw generations: `experiment/exp2/outputs/`
- Eval results: `experiment/exp2/eval/`
- Summary: `experiment/exp2/summary.json`

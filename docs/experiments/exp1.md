---
id: E001
run_dir: experiment/exp1
condition: direct
model: Qwen/Qwen3-4B
datasets: [nq, triviaqa, popqa, hotpotqa, 2wikimultihopqa, musique, bamboogle]
n_examples: 500
status: Done
---

# E001 — Direct answer versus reasoning (direct-answer arm)

**Owner:** Michelle Sheu
**Why:** RQ1 — does reasoning improve factual recall compared with directly
answering? This run is the direct-answer baseline; `exp2` (same E001, condition
`reasoning`) is the paired comparison.
**Setup:** Qwen3-4B, vLLM, `enable_thinking=False`. 7 real QA datasets (single-
and multi-hop), FlashRAG-hub eval splits, 500 examples each (seeded, same
sample reused across conditions for a fair compare). SimpleQA skipped (needs
LLM judge, see `evaluation/simpleqa_judge.py`). Decoding: temp 1.0, top_p 1.0,
max 500 tokens. Metric: normalized EM + cover-EM.
**Expected result:** Baseline accuracy per dataset; comparison point for the
reasoning condition in `exp2`.
**Result:**

| Dataset | Hop | n | EM | Cover-EM | Avg output len |
| :-- | :-- | --: | --: | --: | --: |
| nq | single | 500 | 0.106 | 0.194 | 6.40 |
| triviaqa | single | 500 | 0.270 | 0.338 | 4.33 |
| popqa | single | 500 | 0.124 | 0.158 | 5.17 |
| hotpotqa | multi | 500 | 0.158 | 0.192 | 5.60 |
| 2wikimultihopqa | multi | 500 | 0.240 | 0.264 | 6.93 |
| musique | multi | 500 | 0.018 | 0.036 | 17.96 |
| bamboogle | multi | 125 | 0.032 | 0.056 | 6.49 |

**What we learned:** Direct-answer baseline is weak everywhere, weakest on
musique and bamboogle (EM 0.018 / 0.032) despite musique having by far the
longest outputs (18 words avg vs 4-7 elsewhere) — length alone isn't buying
accuracy here. triviaqa is the strongest single-hop dataset (EM 0.270);
2wikimultihopqa outperforms the other multi-hop sets, which is surprising
given multi-hop is supposed to be harder than single-hop. Cover-EM is
consistently higher than strict EM (~1.3-2x), confirming EM under-counts
correct-but-differently-phrased answers, as expected from the survey notes.
**Decision:** Run `exp2` (condition: reasoning, same seed/sample) to get the
paired RQ1 comparison. If reasoning doesn't help musique/bamboogle
specifically, that's evidence multi-hop failures are a retrieval-chain
problem, not something reasoning alone fixes.

**Links**
- Settings: `experiment/exp1/settings.json`
- Raw generations: `experiment/exp1/outputs/`
- Eval results: `experiment/exp1/eval/`
- Summary: `experiment/exp1/summary.json`

---
id: E004
run_dir: experiment/exp3
condition: direct
model: Qwen/Qwen3.5-2B
datasets: [nq, triviaqa, popqa, hotpotqa, 2wikimultihopqa, musique, bamboogle, simpleqa_verified]
n_examples: 500
status: Done
---

# E004 — Direct answer versus reasoning (Qwen3.5-2B, direct arm)

**Owner:** Michelle Sheu
**Why:** Premise 1 (`project_doc.md` plan 3.1) — reasoning helps factual
recall — across a model size/family sweep, not just the Qwen3-4B pilot
(E001). This is the **pipeline canary**: first run of the sweep, launched
first to confirm the OpenAI-backed judge/ex_recall/simpleqa metrics actually
populate (E001/exp1 silently produced no simpleqa result).
**Setup:** Qwen3.5-2B, vLLM, `enable_thinking=False`. Decoding per Qwen3.5
card's "Instruct (non-thinking) mode, general tasks": temp 1.0, top_p 1.00,
top_k 20, min_p 0, presence_penalty 2.0, rep_penalty 1.0, max 500 tokens.
7 EM datasets (500 examples each, seeded; bamboogle's split is only 125) + simpleqa_verified. Metrics: EM +
cover-EM + judge_accuracy + ex_recall on the EM datasets; native SimpleQA
3-way grade on simpleqa_verified. Requires `OPENAI_API_KEY` exported before
launch (gpt-4.1 judge, gpt-5-mini extractor).
**Expected result:** Baseline accuracy per dataset for the smallest model in
the sweep; comparison point for `exp4` (reasoning arm, same model).
**Result:**

| dataset | n | EM | cover_em | judge_acc | ex_recall | extraction_fail_rate |
|---|---|---|---|---|---|---|
| nq | 500 | 0.074 | 0.130 | 0.170 | 0.116 | 0.266 |
| triviaqa | 500 | 0.152 | 0.246 | 0.248 | 0.206 | 0.254 |
| popqa | 500 | 0.096 | 0.122 | 0.138 | 0.114 | 0.188 |
| hotpotqa | 500 | 0.076 | 0.150 | 0.170 | 0.102 | 0.290 |
| 2wikimultihopqa | 500 | 0.146 | 0.322 | 0.190 | 0.232 | 0.372 |
| musique | 500 | 0.006 | 0.032 | 0.024 | 0.012 | 0.412 |
| bamboogle | 125 | 0.048 | 0.072 | 0.128 | 0.048 | 0.304 |

simpleqa_verified (llm_judge, n=500, separate — not averaged into EM table): overall_accuracy 0.03, accuracy_given_attempted 0.032, f1 0.031, not_attempted 38, extraction_fail_rate 0.228.

**What we learned:** Qwen3.5-2B direct-arm baseline is weak across board — EM 0.6–15.2%, worst on multi-hop musique (0.6% EM), best on triviaqa/2wiki. judge_accuracy and ex_recall both track EM directionally but read higher (e.g. 2wiki EM 14.6% vs judge_acc 19% vs cover_em 32.2%), consistent with EM underestimating true recall on formatting/paraphrase grounds. Extraction failure rates are high (19–41%), worse on longer-answer multi-hop sets — model often doesn't cleanly close `<answer>` tags. Judge/ex_recall/simpleqa pipeline populated correctly (unlike E001/exp1), confirming pipeline canary goal met.
**Decision:** Proceed to exp4 (reasoning arm, same model) for direct comparison at this model size.

**Links**
- Settings: `experiment/exp3/settings.json`
- Raw generations: `experiment/exp3/outputs/`
- Eval results: `experiment/exp3/eval/`
- Summary: `experiment/exp3/summary.json`

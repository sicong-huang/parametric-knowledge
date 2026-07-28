# Premise 2 Execution Plan for Claude Code

Premise 2: hallucinated intermediate facts in reasoning traces cause hallucinated final answers (not just correlate with them).

This plan has three tiers, in dependency order. Tier 1 is observational (uses the existing SAFE-style pipeline). Tier 2 is interventional and is the actual causal proof. Tier 3 replicates the causal result on synthetic KG data. Each tier gets its own experiment ID following the project's convention (experiment/expN/, docs/experiments/expN.md).

Next available experiment IDs: E012, E013, E014 (E001 to E011 are used; confirm E008/SmolLM3 status before running, bump numbers if it gets claimed first).

Budget note: OpenAI spend for this plan should stay under about 150 dollars total, one-shot. Nothing here should call OpenAI or Serper per RL rollout. All Tier 2 and Tier 3 corruption-and-flip experiments use force-decoding against the local model, not external APIs, except for grading trace quality in Tier 1.

---

## Tier 0: Infrastructure (do first)

### Task 0.1: Freeze the trace corpus
- Pull the reasoning-condition generations already produced in E004 to E011 (experiment/exp4, exp6, exp8, exp10, exp14, exp16, exp18). Do not regenerate these.
- Filter to multi-hop datasets for Tier 1 and Tier 2 (HotpotQA, 2WikiMultiHopQA, MuSiQue, Bamboogle) since these have gold supporting-fact annotations that Tier 2 needs. Single-hop datasets can still be used for Tier 1.
- Output: premise2/corpus/traces_manifest.jsonl, one row per trace: model, dataset, question_id, trace_text, final_answer, gold_answer, em_correct.
- Acceptance: manifest covers all 8 models times 4 multi-hop datasets, with per-cell counts logged.

### Task 0.2: Sampling design for Tier 1
- Premise 2's correlational analysis needs multiple samples per question to compare "all facts supported" vs "at least one fact not supported" traces on the same question. Single greedy traces from E004 to E011 do not give this.
- Write premise2/scripts/resample_traces.py: for a stratified subset of questions, regenerate k=8 traces per question at temperature 1.0, reasoning on, using the existing generation scripts in evaluation/.
- Subset: 200 questions per (model, dataset) cell for 4 models (Qwen3.5-2B, Qwen3.5-9B, Nemotron-Nano-9B-v2, Gemma-4-E2B-it) times 4 multi-hop datasets = 3,200 questions times 8 samples = 25,600 traces. This is the trace pool SAFE will grade; do not scale up without re-checking cost first (Task 1.3).
- Acceptance: premise2/corpus/resampled_traces.jsonl populated, with a checked-in log of exact sampling for reproducibility.

---

## Tier 1: Observational within-question correlation

Goal: show that, conditioning on the same question, traces with a hallucinated intermediate fact have lower final-answer accuracy than traces with all facts supported. This controls for question difficulty, unlike a pooled correlation.

### Task 1.1: Atomic fact decomposition
- Write premise2/scripts/decompose_facts.py: for each trace in resampled_traces.jsonl, call gpt-5.4-mini (chat completions, not the built-in web_search tool) to split the think content into atomic, self-contained factual claims, following SAFE's decontextualization step.
- Exclude the final-answer sentence itself; that is graded separately via the existing EM and judge pipeline.
- Acceptance: premise2/corpus/atomic_facts.jsonl, one row per fact: trace_id, fact_id, fact_text.

### Task 1.2: SAFE-literal grading loop
- Write premise2/scripts/safe_grade.py, modeled on the existing evaluation/ex_recall.py threading pattern:
  - Per atomic fact: LLM (gpt-5.4-mini) emits a search query, code calls Serper (google.serper.dev/search, num=10), snippets fed back, LLM emits supported / not_supported / irrelevant or issues another query.
  - Hard cap of 2 search iterations per fact (a documented deviation from SAFE's default up to 5; log this explicitly).
  - Log every query and snippet used per fact to premise2/corpus/safe_evidence_log.jsonl. This is required for auditability.
- Acceptance: premise2/corpus/fact_labels.jsonl: trace_id, fact_id, label, n_queries_used, evidence_ids.

### Task 1.3: Budget check before scaling
- Pilot on 50 traces first (roughly 150 to 300 facts):
  - Compute actual dollars per fact and per trace; compare against the estimated 0.015 to 0.02 dollars per trace. If materially higher, stop and re-scope the sample size in Task 0.2.
  - Human-label a random 50-fact subset; compute agreement with SAFE's automated labels (this replaces the paper's 72 percent, which does not transfer to a different rater LM).
- Acceptance: premise2/reports/safe_pilot_report.md with cost per fact, cost per trace, projected full-run cost, and agreement rate. Get explicit go or no-go before the full Task 1.2 run.

### Task 1.4: Within-question paired analysis
- Write premise2/scripts/tier1_analysis.py:
  - Split traces per question into clean (all facts supported) vs contaminated (at least one not_supported).
  - Compute paired accuracy difference per question; aggregate with a McNemar-style test.
  - Compute dose-response: accuracy vs count of not_supported facts (0, 1, 2, 3+), per model and dataset.
- Acceptance: premise2/reports/tier1_results.md plus a plot of accuracy vs hallucinated-fact count per model, with effect size and significance reported.

---

## Tier 2: Interventional prefix corruption (the causal proof)

Goal: directly manipulate one intermediate fact and force-decode the continuation, to show corruption causes the final answer to flip. Adapted from Lanham et al. 2023's "adding mistakes" protocol, with a paraphrase control to rule out generic distribution-shift damage.

### Task 2.1: Select target traces
- From traces_manifest.jsonl, filter to multi-hop traces that were originally answered correctly and have gold supporting-fact annotations (all four multi-hop datasets ship these; use them directly).
- Align each gold supporting fact to the sentence(s) in the think block that state it (string/entity match; flag unmatched traces rather than silently dropping them).
- Acceptance: premise2/corpus/tier2_candidates.jsonl: trace_id, model, dataset, question_id, hop_facts (list of hop_index, sentence_span, gold_fact_text, on_chain flag). Target at least 500 usable candidates per model, prioritizing Nemotron-Nano-9B-v2 and Gemma-4-E4B-it.

### Task 2.2: Build the three-way corrupted prefix set
For each candidate trace and chosen hop fact, build three prefixes identical up to and including that sentence:
- (a) Original: untouched.
- (b) Corrupted: minimal factual edit (swap entity, date, or number only; keep sentence structure and length otherwise identical). Write premise2/scripts/corrupt_fact.py using gpt-5.4-mini with a strict instruction to change only the target span. Validate every corruption programmatically by diffing the sentence.
- (c) Paraphrased control: same fact, reworded, still correct. This is the critical control ruling out "any off-distribution edit breaks the model."
- QA a 10 percent sample of (b) and (c) before scaling: confirm (b) is factually wrong with nothing else changed, confirm (c) is factually right and meaningfully reworded.
- Acceptance: premise2/corpus/tier2_prefixes.jsonl: trace_id, hop_index, condition, prefix_text.

### Task 2.3: Force-decode continuations
- Write premise2/scripts/force_decode.py: truncate the original trace at the target sentence, splice in each of the three prefix variants, and regenerate the rest of the trace plus final answer from the local model (same model that produced the original trace, via vLLM, not an API call). Sample n=4 continuations per condition at temperature 1.0.
- Acceptance: premise2/corpus/tier2_continuations.jsonl: trace_id, hop_index, condition, sample_id, continuation_text, final_answer.

### Task 2.4: Score flips
- Grade final answers against gold using the existing EM pipeline; no SAFE needed since ground truth is already known.
- Compute per (model, dataset): P(wrong given corrupted) minus P(wrong given paraphrase). Report the corrupted-vs-original gap too as a sanity check.
- Acceptance: premise2/reports/tier2_results.md with flip-rate tables and bootstrap confidence intervals over questions.

### Task 2.5: Placebo check, off-chain corruption
- Repeat 2.2 to 2.4 but corrupt a fact stated in the trace that is not on the gold dependency chain for the question (using the on_chain flag). Expect near-zero flip-rate difference vs the paraphrase control.
- This is the single most important sanity check for the whole causal claim. If the placebo shows a large effect too, do not write up Tier 2 as causal evidence until the corruption methodology is revisited.
- Acceptance: premise2/reports/tier2_placebo_results.md.

### Task 2.6: Position and self-correction analysis
- Break out flip rate by hop position (hop 1 vs hop 2 vs later restatement).
- Classify each continuation for self-correction: did the model notice and contradict the injected error before answering. Report self-correction rate by model size. This is a classification pass over existing Tier 2 outputs, no new generation needed.
- Acceptance: premise2/reports/tier2_position_and_recovery.md.

---

## Tier 3: Synthetic KG replication (after Tier 2 works on real data)

Goal: replicate the causal result where the dependency graph is fully known by construction, removing the alignment guesswork of Task 2.1.

### Task 3.1: Reuse the existing synthetic pipeline
- Use the project's existing synthetic biography and QA generation pipeline; do not build a new one. Confirm whether the SFT'd model is ready; if not, flag this tier as blocked rather than working around it.
- Select 2-hop QA pairs from the synthetic KG.

### Task 3.2: Controlled corruption
- Swap the hop-1 fact for a different, plausible KG entity that exists elsewhere in the KG (stronger and cleaner than Tier 2's free-text edit, since it stays in-distribution by construction).
- Same three-way design and flip-rate measurement as Tasks 2.3 to 2.4.
- Stratify by the fact's pretraining occurrence frequency (already varied in the synthetic data): does corrupting a low-frequency fact flip the answer more or less often than corrupting a high-frequency one. This connects to the planned SFT-vs-RL low-frequency-fact ablation.
- Acceptance: premise2/reports/tier3_results.md, including the frequency-stratified breakdown.

---

## Deliverables checklist
- premise2/corpus/traces_manifest.jsonl
- premise2/corpus/resampled_traces.jsonl
- premise2/corpus/atomic_facts.jsonl
- premise2/corpus/safe_evidence_log.jsonl
- premise2/corpus/fact_labels.jsonl
- premise2/reports/safe_pilot_report.md (go/no-go gate, required before full Tier 1 grading)
- premise2/reports/tier1_results.md
- premise2/corpus/tier2_candidates.jsonl
- premise2/corpus/tier2_prefixes.jsonl
- premise2/corpus/tier2_continuations.jsonl
- premise2/reports/tier2_results.md
- premise2/reports/tier2_placebo_results.md (do not skip)
- premise2/reports/tier2_position_and_recovery.md
- premise2/reports/tier3_results.md (blocked on synthetic SFT pipeline readiness)
- docs/experiments/exp19.md (E012, Tier 1), exp20.md (E013, Tier 2), exp21.md (E014, Tier 3), following the existing experiment template
- Update project_doc.md's Current Understanding and Tasks sections once results land, per the project's existing convention

## Guardrails throughout
- Never call OpenAI's built-in web_search tool for this work; Serper only, per the earlier cost and reproducibility comparison. Log every query and snippet, no exceptions.
- Stop and report before any single task step would exceed about 50 dollars in API spend; get a go/no-go rather than silently burning the monthly budget.
- Keep Tier 2 and Tier 3 generation local via vLLM; zero OpenAI spend there.
- Treat the placebo check (Task 2.5) as load-bearing for the whole causal claim.

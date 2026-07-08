# Task: Real-Dataset Evaluation Pipeline (Search-R1 datasets)

This file is a step-by-step spec for building the real-dataset RL/eval pipeline
described in `Project_doc.md`. Work through the phases in order. Each phase
lists concrete deliverables — treat them as a checklist and don't skip ahead
until a phase's files exist and run without error.

Context docs already in this repo (read these first):
- `Project_doc.md` — overall project goal, plan, experiment log, directory
  structure convention
- `Search-R1_datasets_reference_v2.docx` (or its extracted text) — dataset
  sources, split sizes, and evaluation metric per dataset

## Target directory structure

Build toward this exact layout (already specified in `Project_doc.md`):

```
experiment/
  exp1/
    settings.json
    outputs/
      nq.jsonl
      triviaqa.jsonl
      popqa.jsonl
      simpleqa.jsonl
      musique.jsonl
      hotpotqa.jsonl
      bamboogle.jsonl
      2wiki.jsonl
    eval/
      nq.jsonl
      triviaqa.jsonl
      ...
data/
  {dataset}_test.jsonl        # unified schema, one file per dataset
  {dataset}_train.jsonl       # only for nq, hotpotqa (RL pool), if pulled in
scripts/
  prepare_data.py
  generate.py
  eval_em.py
  eval_llm_judge.py
  run_all.py
```

---

## Phase 1 — Verify dataset splits (Task: "find out real datasets with training split")

**Goal:** confirm actual split sizes by loading the data, not just trusting
headline numbers (the reference doc itself flags that FlashRAG-preprocessed
counts differ from full/original counts).

1. Install deps: `pip install datasets huggingface_hub --break-system-packages`
   (or add to `requirements.txt` if one exists).
2. Write `scripts/check_splits.py` that loads each of the 8 datasets and
   prints train/dev/test counts:
   - 7 via `RUC-NLPIR/FlashRAG_datasets` on HuggingFace (configs: `nq`,
     `triviaqa`, `popqa`, `hotpotqa`, `musique`, `bamboogle`, `2wikimultihopqa`
     — verify exact config names against the dataset card, they may differ
     slightly)
   - SimpleQA separately from `basicv8vc/SimpleQA` (not in FlashRAG)
3. Save the printed table to `docs/dataset_splits_verified.md` — this becomes
   the source of truth instead of the reference doc's headline numbers.
4. Update `Project_doc.md`'s "Discussed" table with any counts that differ
   from what's currently written.

**Decision to record** (append to `Project_doc.md` under "Current understanding"
or "Decided"):
- RL training pool = NQ + HotpotQA train splits merged (Search-R1 protocol),
  subsampled to ~6,000–20,000 examples for first runs.
- PopQA, SimpleQA, Bamboogle, TriviaQA, MuSiQue, 2Wiki = held-out eval-only
  transfer benchmarks for the first pass.
- Metric: EM (normalized string match) as default for 7 datasets; SimpleQA
  uses an LLM grader (correct / incorrect / not attempted), reported
  separately, not averaged in with EM scores.

---

## Phase 2 — Unify data format

**Goal:** one consistent schema across all 8 datasets so generation/eval code
doesn't need per-dataset branching.

1. Write `scripts/prepare_data.py`:
   - Input: raw HF dataset object per dataset.
   - Output: `data/{dataset}_test.jsonl`, one JSON object per line:
     ```json
     {"id": "nq_0001", "dataset": "nq", "question": "...", "golden_answers": ["...", "..."]}
     ```
   - Also emit `data/{dataset}_train.jsonl` for NQ and HotpotQA only.
2. For large test sets (PopQA ~14K, TriviaQA ~11K), also emit a subsampled
   version, e.g. `data/{dataset}_test_sample500.jsonl`, using a fixed seed for
   reproducibility. Keep Bamboogle whole (only 125 examples).
3. Sanity check: print 2 examples from each output file and confirm
   `golden_answers` is always a non-empty list of strings.

---

## Phase 3 — Generation script

**Goal:** produce model outputs for both experimental conditions Task E001
needs (direct answer vs. reasoning).

1. Write `scripts/generate.py` with CLI args:
   `--dataset --model --condition {direct,reasoning} --split {test,train} --sample_size --output_path`
2. Use vLLM for batched inference if available in the environment (much
   faster than plain `transformers` for thousands of prompts); fall back to
   `transformers` if vLLM isn't installed.
3. Two prompt templates:
   - **direct**: ask for the answer only, no reasoning.
   - **reasoning**: ask the model to think step by step, then give a final
     answer in a clearly extractable format (e.g. `Answer: ...`).
4. Output schema, one line per example in `experiment/exp1/outputs/{dataset}.jsonl`:
   ```json
   {"id": "...", "question": "...", "golden_answers": [...], "condition": "direct", "raw_generation": "...", "extracted_answer": "..."}
   ```
5. Always keep `raw_generation` — needed for the length-controlled comparison
   in E002 and for debugging extraction failures.

---

## Phase 4 — Evaluation scripts (Task: "setup evaluation on real datasets")

**Goal:** score the generations from Phase 3.

1. Write `scripts/eval_em.py`:
   - Normalize both prediction and golden answers: lowercase, strip
     punctuation/articles/extra whitespace (standard SQuAD-style
     normalization — check if `experiment/` or elsewhere in the repo already
     has a normalization function before writing a new one).
   - Exact match against any of the golden answers.
   - Also compute a "cover-EM" / substring match as a secondary column (cheap
     to add, more faithful for parametric-recall evaluation per the
     reference doc).
   - Write `experiment/exp1/eval/{dataset}.jsonl` with per-example
     `{"id": ..., "em": 0/1, "cover_em": 0/1}` plus a printed aggregate
     accuracy summary.
2. Write `scripts/eval_llm_judge.py` for SimpleQA specifically:
   - Implement the three-way grader (correct / incorrect / not attempted).
   - Base the grading prompt on the public one from `openai/simple-evals`
     (fetch via web/GitHub, don't reproduce verbatim from memory — check the
     actual repo).
   - Report SimpleQA results separately from the EM-based datasets; do not
     average it into an overall EM number.
3. Keep `outputs/` (raw generations) and `eval/` (scored results) as
   separate files per the directory convention above — never overwrite raw
   generations with scored output.

---

## Phase 5 — One-command runner

1. Write `scripts/run_all.py` that reads `experiment/exp1/settings.json`
   (model name, list of datasets, condition, sample size, decoding params)
   and runs generate → eval for every dataset in sequence.
2. `settings.json` example:
   ```json
   {
     "model": "qwen-4b",
     "datasets": ["nq", "triviaqa", "popqa", "simpleqa", "musique", "hotpotqa", "bamboogle", "2wiki"],
     "condition": "direct",
     "sample_size": 500,
     "max_new_tokens": 256
   }
   ```

---

## Phase 6 — Sanity check before scaling

1. Run the full pipeline end-to-end on 100 NQ examples only.
2. Manually inspect 20 `raw_generation` / `extracted_answer` / `em` triples —
   this is where answer-extraction bugs surface (e.g. "The answer is 1997."
   getting scored as a non-match against "1997").
3. Fix extraction/normalization bugs found, then re-run on the full sample
   sizes for all 8 datasets.

---

## Phase 7 — Baseline numbers

1. Run the chosen small model (e.g. qwen-4b) in the **direct** condition on
   all 8 test sets using `run_all.py`.
2. Record results in `Project_doc.md` under experiment `E001` (or a new
   `E000` baseline row if E001 is reserved for the direct-vs-reasoning
   comparison) — accuracy and average output length per dataset.
3. This baseline is what "Real data and evaluation pipeline done" (7/7
   milestone in `Project_doc.md`'s timeline) refers to.

---

## Notes / things to double check while implementing

- Verify exact FlashRAG config names against the HF dataset card before
  hardcoding them — don't assume `nq`/`triviaqa`/etc. are exactly right.
- SimpleQA is not in FlashRAG — load it separately.
- 2WikiMultiHopQA: FlashRAG ships a 15,000-example train subset, not the
  full 167,454 — confirm which one is actually being loaded.
- Don't average SimpleQA's LLM-judge score together with the EM scores from
  the other 7 datasets in any summary table — report it as its own column.

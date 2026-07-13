# Stratified 2-hop QA

Generate, validate, and repair **stratified 2-hop** bio QA under
`<run_dir>/stratified_2hop/` (default run: `syndata_archive/data/bio/run5`).

## Patterns (10k each → 30k total)

| Pattern | Geometry | Answer |
|---------|----------|--------|
| `P,P->NP` | Join: two people → shared non-person | Shared NP |
| `NP<-P->NP` | One person bridges two NPs | Second NP |
| `P->P->NP` | Person → person → NP | Final NP |

`occupation_is` edges are excluded in `create_qa.py` (`EXCLUDED_RELATIONS`) during
enumeration — new runs never sample those paths.

## Outputs

| File | Contents |
|------|----------|
| `paths_2_hop.jsonl` | Sampled paths |
| `qa_2_hop.jsonl` | Questions + graph edges |
| `qa_2_hop_direct.jsonl` | Direct (answer-only) style |
| `qa_2_hop_reasoning.jsonl` | `<think>…</think>` + answer |
| `validation_report.txt` | `validate_qa.py` report |
| `grounding_report.json` / `grounding_failures.jsonl` | Reasoning checks |
| `repair/<timestamp>/` | Staged reasoning repairs (no merge until `--approve`) |

---

## 0. vLLM (Gemma 4)

```bash
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run vllm serve google/gemma-4-31B-it \
  --served-model-name gemma4 \
  --tensor-parallel-size 4 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --limit-mm-per-prompt '{"image": 0, "audio": 0}' \
  --host 0.0.0.0 \
  --port 8001

curl http://localhost:8001/v1/models
export QA_BASE_URL=http://localhost:8001/v1
# QA_API_KEY optional for local vLLM (defaults to EMPTY)
```

---

## 1. Smoke test (~67 / pattern, ~201 total)

```bash
export QA_BASE_URL=http://localhost:8001/v1

# Paths only (no LLM)
uv run syndata_archive/stratified_2hop/create_qa.py 2_hop \
  --dir syndata_archive/data/bio/run5 --enumerate-only --short

# Questions
uv run syndata_archive/stratified_2hop/create_qa.py 2_hop \
  --dir syndata_archive/data/bio/run5 --short --max-workers 8

# Direct + reasoning
uv run syndata_archive/stratified_2hop/create_qa.py 2_hop_full_answer \
  --dir syndata_archive/data/bio/run5 --style direct

uv run syndata_archive/stratified_2hop/create_qa.py 2_hop_full_answer \
  --dir syndata_archive/data/bio/run5 --style reasoning --max-workers 8

# Validate
uv run syndata_archive/stratified_2hop/validate_qa.py \
  --dir syndata_archive/data/bio/run5 --short
```

---

## 2. Full production (10k / pattern)

```bash
export QA_BASE_URL=http://localhost:8001/v1

# Optional: enumerate paths only
uv run syndata_archive/stratified_2hop/create_qa.py 2_hop \
  --dir syndata_archive/data/bio/run5 \
  --sample-per-pattern 10000 --seed 42 --enumerate-only

# Questions
uv run syndata_archive/stratified_2hop/create_qa.py 2_hop \
  --dir syndata_archive/data/bio/run5 \
  --sample-per-pattern 10000 --seed 42 --max-workers 32

# Direct (no LLM)
uv run syndata_archive/stratified_2hop/create_qa.py 2_hop_full_answer \
  --dir syndata_archive/data/bio/run5 --style direct

# Reasoning
uv run syndata_archive/stratified_2hop/create_qa.py 2_hop_full_answer \
  --dir syndata_archive/data/bio/run5 --style reasoning --max-workers 32

# Fill only missing qaids (e.g. after occupation replace)
uv run syndata_archive/stratified_2hop/create_qa.py 2_hop_full_answer \
  --dir syndata_archive/data/bio/run5 --style reasoning \
  --only-missing --max-workers 32
```

---

## 3. Validate

```bash
# Production sizes
uv run syndata_archive/stratified_2hop/validate_qa.py \
  --dir syndata_archive/data/bio/run5 --expect-per-pattern 10000

# Smoke sizes
uv run syndata_archive/stratified_2hop/validate_qa.py \
  --dir syndata_archive/data/bio/run5 --short
```

Writes `stratified_2hop/validation_report.txt` and
`question_graph_issues.jsonl` when NL vs graph naming fails.

---

## 4. Reasoning grounding / prompt check

```bash
uv run syndata_archive/stratified_2hop/check_reasoning_grounding.py \
  --dir syndata_archive/data/bio/run5

# Fail CI-style on outside entities or prompt issues
uv run syndata_archive/stratified_2hop/check_reasoning_grounding.py \
  --dir syndata_archive/data/bio/run5 --fail-on-outside --fail-on-prompt
```

Writes:

- `grounding_details.jsonl`
- `grounding_failures.jsonl`
- `grounding_report.json`

Tracked prompt issues include:

- `post_think_answer_mismatch`
- `answer_fact_not_in_middle_sentence`
- `source_leak`
- `sentence_count_out_of_band`

---

## 5. Repair failed reasoning (staging → approve)

Does **not** modify `qa_2_hop_reasoning.jsonl` until `apply --approve`.

```bash
export QA_BASE_URL=http://localhost:8001/v1

# Select failing qaids (default: the four issues above)
uv run syndata_archive/stratified_2hop/repair_reasoning.py select \
  --dir syndata_archive/data/bio/run5

# Optional: subset of issues
uv run syndata_archive/stratified_2hop/repair_reasoning.py select \
  --dir syndata_archive/data/bio/run5 \
  --issues post_think_answer_mismatch,source_leak

# Regenerate into repair/<timestamp>/ only
uv run syndata_archive/stratified_2hop/repair_reasoning.py regenerate \
  --run-dir syndata_archive/data/bio/run5/stratified_2hop/repair/<timestamp> \
  --max-workers 32 --max-retries 3

# Recheck regen
uv run syndata_archive/stratified_2hop/repair_reasoning.py recheck \
  --run-dir syndata_archive/data/bio/run5/stratified_2hop/repair/<timestamp>

# ONLY after review — merge passed rows into main reasoning file
uv run syndata_archive/stratified_2hop/repair_reasoning.py apply \
  --run-dir syndata_archive/data/bio/run5/stratified_2hop/repair/<timestamp> \
  --approve --passed-only
```

Staging artifacts: `before.jsonl`, `regen.jsonl`, `passed.jsonl`,
`failed_still.jsonl`, `recheck_report.json`, `manifest.json`.

---

## 6. Train / test split

Default: **global edge-disjoint** 70/30 split. No KG triplet
`(source_id, relation_type, target_id)` appears in both train and test.
The same `qaid` partition is used for base, **direct**, and **reasoning**.

```bash
uv run syndata_archive/stratified_2hop/split_qa.py \
  --dir syndata_archive/data/bio/run5 --seed 42

# Custom train fraction
uv run syndata_archive/stratified_2hop/split_qa.py \
  --dir syndata_archive/data/bio/run5 --train-fraction 0.7 --seed 42

# Smoke
uv run syndata_archive/stratified_2hop/split_qa.py \
  --dir syndata_archive/data/bio/run5 --short
```

**Outputs** (under `stratified_2hop/qa_split/`):

| File | Contents |
|------|----------|
| `qa_2_hop_train.jsonl` / `_test.jsonl` / `_dropped.jsonl` | Base questions |
| `qa_2_hop_direct_train.jsonl` / `_test.jsonl` / `_dropped.jsonl` | Direct style |
| `qa_2_hop_reasoning_train.jsonl` / `_test.jsonl` / `_dropped.jsonl` | Reasoning style |
| `split_report.json` | Counts, per-pattern breakdown, overlap check |

**Why ~5% dropped?** Edge-disjoint is a global constraint: each row
needs both of its edges on the same side. If one edge is already train
and the other is already test (via other paths), that row cannot be
placed without leaking — it is dropped. This is not random data loss;
~1.5k of 30k is typical for this graph.

**Naive split** (allows edge leakage, old 50/50-style):

```bash
uv run syndata_archive/stratified_2hop/split_qa.py \
  --dir syndata_archive/data/bio/run5 --random-split --train-fraction 0.5 --seed 42
```

---

## 7. 1-hop train / test split (same logic)

1-hop lives under `<run_dir>/stratified_1hop/` (not `qa/`). It has **no path
patterns** — questions are single edges — but uses the **same global
edge-disjoint 70/30** rule. Direct / sentence / reasoning share the same qaids.

```bash
uv run syndata_archive/stratified_1hop/split_qa.py \
  --dir syndata_archive/data/bio/run5 --seed 42
```

**Outputs** (under `stratified_1hop/qa_split/`):

| File | Contents |
|------|----------|
| `qa_1_hop_train.jsonl` / `_test.jsonl` / `_dropped.jsonl` | Base |
| `qa_1_hop_direct_*.jsonl` | Direct |
| `qa_1_hop_reasoning_*.jsonl` | Reasoning |
| `split_report.json` | Counts + per-relation breakdown |

**Note:** 1-hop rows are unique `(source, relation)` groups, so each triplet
usually appears once — drops are typically **near zero** (unlike ~4–5% for
2-hop). Ordering is stratified by `relation_type` (round-robin), analogous
to 2-hop’s per-pattern balance.

See also `syndata_archive/stratified_1hop/README.md`.

---

## Scripts

| Script | Role |
|--------|------|
| `create_qa.py` | Enumerate / sample paths, generate questions & styled answers |
| `validate_qa.py` | Graph + NL validation |
| `check_reasoning_grounding.py` | Entity grounding + prompt compliance |
| `repair_reasoning.py` | Stage / regen / recheck / approve-merge reasoning fixes |
| `split_qa.py` | Edge-disjoint train/test split (2-hop) |
| `../stratified_1hop/` | 1-hop pipeline (parallel layout) |
| `common.py` | Shared helpers |

## Common flags

| Flag | Meaning |
|------|---------|
| `--dir` | Graph run dir (default `syndata_archive/data/bio/run5`) |
| `--qa-subdir` | Output folder under run dir (default `stratified_2hop`) |
| `--model` | OpenAI-compatible model id (default `gemma4`) |
| `--max-workers` | Parallel LLM requests |
| `--short` | Smoke-test sizes |
| `--train-fraction` | Target train share for split (default `0.7`) |
| `--random-split` | Naive shuffle split (allows edge leakage) |
| `QA_BASE_URL` | vLLM / API base (e.g. `http://localhost:8001/v1`) |

# parametric-knowledge

Testing whether reasoning and RL improve a language model's ability to reliably access and use factual knowledge already stored in its parameters — closed-book QA, no retrieval. See `project_doc.md` for the full research questions, plan, timeline, and experiment log, and `CLAUDE.md` for repo/agent guidance.

## Requirements

- Python >=3.10, package manager `uv`
- GPU (vLLM inference)
- Deps: `vllm>=0.6.0`, `transformers>=4.46.0`, `datasets>=3.0.0`, `huggingface_hub>=0.25.0`, `openai>=1.0` (see `pyproject.toml`)

```bash
uv sync
```

`OPENAI_API_KEY` (in `.env`) is only needed for the LLM-judge / ex-recall / `simpleqa_verified` paths — not for plain EM scoring.

**Runtime gotcha:** every vLLM invocation needs `VLLM_USE_FLASHINFER_SAMPLER=0`. This machine's PATH resolves `nvcc` to a stray `nvidia-cuda-toolkit` package instead of the real `cuda-nvcc-12-8`, so flashinfer's sampler kernel fails to JIT-compile and crashes the vLLM engine on startup. Disabling the flashinfer sampler skips that path. `scripts/run_exp.sh` sets this for you; set it manually if calling `generate.py` / `run_all.py` directly.

## Quickstart

```bash
# download eval-split data for all enabled datasets (or one with --dataset)
uv run python download_data.py [--dataset nq] [--force]

# generate answers for one dataset into experiment/<exp>/outputs/<dataset>.jsonl
VLLM_USE_FLASHINFER_SAMPLER=0 uv run python scripts/generate.py --dataset nq --exp exp1

# score one dataset's generations into experiment/<exp>/eval/<dataset>.jsonl
uv run python -m evaluation.eval_dataset --dataset nq \
  --outputs experiment/exp1/outputs/nq.jsonl --eval-out experiment/exp1/eval/nq.jsonl \
  [--judge] [--ex-recall]

# one-command generate+eval for every dataset in an experiment's settings.json
VLLM_USE_FLASHINFER_SAMPLER=0 uv run python scripts/run_all.py --exp exp1

# preferred: launch the full run detached in tmux (survives long GPU jobs, sets the env var for you)
bash scripts/run_exp.sh exp1
```

`run_exp.sh` starts tmux session `exp_<exp>`, tees output to `experiment/<exp>/run.log`, and refuses to start a duplicate session for the same exp. The `run-experiment` skill (`.claude/skills/run-experiment/SKILL.md`) wraps starting/monitoring/reporting on this.

## Pipeline

```
datasets_registry.py        switchboard: 8 datasets, each {dir/hf_repo, eval_split, metric, hop, enabled}
        │
download_data.py            pulls eval splits → data/<dataset>/<split>.jsonl
        │
scripts/generate.py          vLLM inference → experiment/<exp>/outputs/<dataset>.jsonl
        │
evaluation/eval_dataset.py   scores → experiment/<exp>/eval/<dataset>.jsonl
        │
scripts/run_all.py           chains download → generate → eval for every dataset,
                              writes experiment/<exp>/summary.json
scripts/run_exp.sh           wraps run_all.py in a detached tmux session
```

**Record schemas at each stage:**

- data (`data/<dataset>/<split>.jsonl`): `{id, question, golden_answers}`
- outputs (`experiment/<exp>/outputs/<dataset>.jsonl`): `{id, question, golden_answers, prompt, raw_output, predicted_answer}`
- eval, EM path (`experiment/<exp>/eval/<dataset>.jsonl`): `{id, predicted_answer, golden_answers, em, cover_em, extraction_failure, raw_output, reasoning_trace}`, plus `judge_grade` if `--judge`, plus `refined_answer`/`recalled`/`skipped` if `--ex-recall`
- `summary.json`: `{exp_id, model, condition, n_examples, results: [...]}`, one result dict per dataset

## Datasets

`datasets_registry.py` is the single source of truth for the 8-dataset eval suite:

| dataset | source | eval_split | metric | hop |
|---|---|---|---|---|
| nq | FlashRAG hub | test | em | single |
| triviaqa | FlashRAG hub | test | em | single |
| popqa | FlashRAG hub | test | em | single |
| hotpotqa | FlashRAG hub | dev | em | multi |
| 2wikimultihopqa | FlashRAG hub | dev | em | multi |
| musique | FlashRAG hub | dev | em | multi |
| bamboogle | FlashRAG hub | test | em | multi |
| simpleqa_verified | `google/simpleqa-verified` (HF hub) | eval | llm_judge | single |

The 7 FlashRAG datasets share a unified `{id, question, golden_answers}` schema and are scored by exact match. `simpleqa_verified` is the exception — not on FlashRAG, loaded directly from the HF hub, and graded by an LLM judge instead of exact match. **Never average its score into the EM-based summary table** — report it separately. (hotpotqa/2wikimultihopqa/musique use `dev` since there are no public test labels.)

## Metrics

- **Primary: normalized exact match** (`evaluation/metrics.py::exact_match`) — lowercase, strip punctuation/articles, collapse whitespace — matching the Search-R1 convention where EM doubles as the RL reward.
- **`cover_em`** (substring match) logged alongside as a looser, more parametric-recall-faithful signal, but EM stays primary.
- **`extraction_failure_rate`** — fraction of generations with no `<answer>...</answer>` tag (see extraction below). exp1 saw 42–73% extraction failures in the direct condition — a format confound, not necessarily a recall failure (see `exp1_extraction_failure.md`).
- **Answer extraction** (`evaluation/parse.py::extract_answer`): model output is expected to end with `<answer>...</answer>` tags (Search-R1 prompt template); takes the last tag match, falling back to the `<think>`-stripped full text if no tag is present.
- **Opt-in co-primary semantic metrics** — run alongside EM (never replacing it), gated by `settings.json`'s `judge`/`ex_recall` booleans (default `false`) and `--judge`/`--ex-recall` flags on `evaluation.eval_dataset`. Rationale in `eval_metrics_research.md`: EM alone can deflate the reasoning-ON condition for formatting/verbosity reasons rather than true recall failure.
  - **`judge_accuracy`** (`evaluation/simpleqa_judge.py`) — reuses the SimpleQA-Verified grader (`JUDGE_MODEL` env var, default `gpt-4.1-2025-04-14`) against the full `golden_answers` alias list, giving paraphrase/verbosity robustness plus a free abstention signal.
  - **`ex_recall`** (`evaluation/ex_recall.py`) — a small LM (`EXTRACTOR_MODEL` env var, default `gpt-5-mini-2025-08-07`) refines `predicted_answer` down to a single committed span, then a word-boundary substring match determines recall. Cheap and not gameable by verbose candidate-listing.
  - Both reuse the `OPENAI_API_KEY` / `OPENAI_BASE_URL` pattern — point `OPENAI_BASE_URL` at a local vLLM server (e.g. an open-weight judge like Qwen2.5-72B-Instruct) to avoid API cost at scale.

## Experiment convention

Each `experiment/expN/` directory has:

- `settings.json` — `exp_id` (links to `project_doc.md`'s Experiment Log, e.g. `E001`), `model`, `condition` (`direct`|`reasoning`), `decoding` (`temperature`, `top_p`, `max_tokens`), `n_examples`, `seed`, `judge`, `ex_recall`, `datasets` (list)
- `outputs/` — raw generations per dataset, never overwritten by scoring
- `eval/` — scored per-example results per dataset
- `run.log` — teed stdout/stderr from the run
- `summary.json` — aggregated metrics across datasets

`condition` (`direct` vs `reasoning`) is controlled by `enable_thinking` passed to the tokenizer's chat template in `scripts/generate.py`, not by a different system prompt — the same answer-then-`<answer>`-tag system prompt is used for both.

`docs/README.md` is a lookup table mapping `experiment/expN/` run dirs to canonical experiment IDs (`E00X`) and per-run docs under `docs/experiments/`. To add a new experiment: copy `experiment/exp1/` → `experiment/expN/`, edit `settings.json`, copy `docs/experiments/template.md` → `docs/experiments/expN.md`, add a row to `docs/README.md`'s index table.

## Repo layout

```
datasets_registry.py           8-dataset switchboard
download_data.py               fetch eval splits
pyproject.toml / uv.lock       uv-managed deps
project_doc.md                 research questions, plan, experiment log (read before new experiment work)
CLAUDE.md                      repo/agent guidance
eval_metrics_research.md       metrics literature survey + recommendations
exp1_extraction_failure.md     findings memo: exp1's answer-tag extraction-failure confound

scripts/
  generate.py                  vLLM inference for one dataset
  run_all.py                   chains download→generate→eval for an experiment
  run_exp.sh                   detached-tmux launcher for run_all.py

evaluation/
  eval_dataset.py              main scoring entrypoint
  metrics.py                   exact_match / cover_em / normalize_answer
  parse.py                     <answer>/<think> tag extraction
  simpleqa_judge.py             GPT-4.1 3-way LLM judge (correct/incorrect/not_attempted)
  ex_recall.py                  refine-then-substring-match recall metric

data/                          downloaded eval-split jsonl, one dir per dataset
experiment/                    one dir per experiment run (expN/)
docs/
  README.md                    run-dir → E00X index
  experiments/                 per-experiment docs (template.md + filled instances)
.claude/skills/run-experiment/ skill for launching/monitoring runs in tmux
```

## Status

Only `experiment/exp1` has been run: E001, direct condition, `Qwen/Qwen3-4B`, 7 EM datasets @ n=500 (`simpleqa_verified` skipped — disabled). exp2 (reasoning condition, same model/data) is planned but not yet run. Known caveat: exp1 saw 42–73% extraction-failure rates in the direct condition, meaning a chunk of low EM scores reflect missing `<answer>` tags rather than failed recall — tracked via `extraction_failure_rate` and detailed in `exp1_extraction_failure.md`.

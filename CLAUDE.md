# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project goal

Test whether reasoning and RL improve a language model's ability to reliably access and use factual knowledge already stored in its parameters (see `project_doc.md` for full research questions, plan, timeline, and the experiment log — read that file before starting new experiment work). `docs/README.md` is a lookup table mapping `experiment/expN/` run dirs to canonical experiment IDs (`E00X`) and per-run docs under `docs/experiments/`.

## Commands

Package manager is `uv` (see `pyproject.toml`, Python >=3.10). No test suite, linter, or formatter is configured in this repo.

```bash
# download eval-split data for all enabled datasets (or one with --dataset)
uv run python download_data.py [--dataset nq] [--force]

# generate answers for one dataset into experiment/<exp>/outputs/<dataset>.jsonl
uv run python scripts/generate.py --dataset nq --exp exp1

# score one dataset's generations into experiment/<exp>/eval/<dataset>.jsonl
uv run python -m evaluation.eval_dataset --dataset nq --outputs experiment/exp1/outputs/nq.jsonl --eval-out experiment/exp1/eval/nq.jsonl

# one-command generate+eval for every dataset in an experiment's settings.json
uv run python scripts/run_all.py --exp exp1

# re-score existing outputs without regenerating (e.g. after changing judge_model/eval_base_url)
uv run python scripts/run_all.py --exp exp1 --eval-only

# preferred: launch the full run detached in tmux (survives long GPU jobs)
bash scripts/run_exp.sh exp1

# start the local judge/ex-recall server (must be running first for exps with judge/ex_recall enabled)
bash scripts/serve_judge.sh
```

**Always run `scripts/generate.py` / `scripts/run_all.py` with `VLLM_USE_FLASHINFER_SAMPLER=0`** (or via `scripts/run_exp.sh`, which sets it). This machine's PATH resolves `nvcc` to a stray `nvidia-cuda-toolkit` package instead of the real `cuda-nvcc-12-8`, so flashinfer's sampler kernel fails to JIT-compile and crashes the vLLM engine on startup. Disabling the flashinfer sampler skips that path.

For long GPU runs, use the `run-experiment` skill (`.claude/skills/run-experiment/SKILL.md`) instead of running `run_all.py` inline — it starts the job in a detached tmux session (`exp_<exp>`), tees output to `experiment/<exp>/run.log`, and refuses to start a duplicate session for the same exp.

## Architecture

**Data flow:** `datasets_registry.py` (single source of truth for the 8-dataset eval suite) → `download_data.py` (pulls eval-split JSONL into `data/<dataset>/<split>.jsonl`) → `scripts/generate.py` (vLLM inference → `experiment/<exp>/outputs/<dataset>.jsonl`) → `evaluation/eval_dataset.py` (scores → `experiment/<exp>/eval/<dataset>.jsonl` + summary). `scripts/run_all.py` chains download→generate→eval for every dataset listed in an experiment's `settings.json` and writes `experiment/<exp>/summary.json`.

**`datasets_registry.py`** is the switchboard: each dataset entry has `dir`/`hf_repo`, `eval_split`, `metric` (`em` or `llm_judge`), `hop`, `enabled`. 7 datasets (nq, triviaqa, popqa, hotpotqa, 2wikimultihopqa, musique, bamboogle) come from the FlashRAG hub (`RUC-NLPIR/FlashRAG_datasets`) with a unified `{id, question, golden_answers}` schema. `simpleqa_verified` is the exception: it's not on FlashRAG, loads from `google/simpleqa-verified` on the HF hub directly, and is graded by an LLM judge instead of exact match. Never average its score into the EM-based summary table — report it separately.

**Metrics** (`evaluation/metrics.py`): default is normalized exact match (`exact_match`) — lowercase, strip punctuation/articles, collapse whitespace — matching the Search-R1 convention where EM doubles as the RL reward. `cover_em` (substring match) is logged alongside it as a looser, more parametric-recall-faithful signal, but EM stays the primary metric.

**Answer extraction** (`evaluation/parse.py`): model output is expected to end with `<answer>...</answer>` tags (Search-R1 prompt template); `extract_answer` takes the last tag match, falling back to the `<think>`-stripped full text if no tag is present.

**SimpleQA judge** (`evaluation/simpleqa_judge.py`): GPT-4.1 autorater, grading prompt copied verbatim from the SimpleQA-Verified convention (a lightly-modified variant of the original `openai/simple-evals` prompt — numeric acceptable-range rules, extra examples — matching the `simpleqa_verified` dataset). 3-way grade (correct/incorrect/not_attempted) → `accuracy_given_attempted`, `overall_accuracy`, `f1`. Needs `OPENAI_API_KEY` (or `OPENAI_BASE_URL` pointed at a local vLLM server to avoid API cost at scale); `JUDGE_MODEL` env var overrides the default `gpt-4.1-2025-04-14`.

**Co-primary semantic metrics on the EM datasets:** EM alone can deflate the reasoning-ON condition for formatting/verbosity reasons rather than true recall failure (see `eval_metrics_research.md`). Two optional, opt-in metrics run *alongside* EM (never replacing it) on the 7 EM-metric datasets, gated by `settings.json`'s `judge`/`ex_recall` booleans (both default `false`) and `--judge`/`--ex-recall` flags on `evaluation.eval_dataset`:
- **`judge_accuracy`** (`evaluation/simpleqa_judge.py:score_judge`/`grade_multi`) — reuses the same SimpleQA-Verified grader against the full `golden_answers` alias list (not just `golden_answers[0]`), giving best paraphrase/verbosity robustness plus a free abstention (not_attempted) signal.
- **`ex_recall`** (`evaluation/ex_recall.py`, ported from Ma & Hewitt's `MelodyHorsee/parametric-knowledge-access`) — a small LM (`EXTRACTOR_MODEL`, default `gpt-5-mini-2025-08-07`) "refines" `predicted_answer` down to a single committed span, then a word-boundary substring match (its own, richer `normalize_answer`, kept local so it never perturbs the strict-EM anchor) determines recall. Cheap, not gameable by verbose candidate-listing, and the fallback co-primary if judge human-validation agreement is ever found to be weak.

Both reuse the `OPENAI_BASE_URL` → local-vLLM override pattern for cost-safe scale (e.g. an open-weight judge like Qwen2.5-72B-Instruct).

**Local judge server:** exp3–exp10 point `eval_base_url` at `http://localhost:8000/v1` (`judge_model`/`extractor_model`: `gemma-4-31b-it`), served locally via `scripts/serve_judge.sh` — `google/gemma-4-31b-it`, fp8 quantization, pinned to GPU1 (`CUDA_VISIBLE_DEVICES=1`), `--max-model-len 40960` (must be large enough to hold a full reasoning-condition `<think>` trace embedded in the grading prompt, or vLLM 400s and crashes `run_all.py`). Must be running (`bash scripts/serve_judge.sh`, wait for `Uvicorn running` in `experiment/judge_server.log`) before launching any exp with `judge`/`ex_recall` enabled. Replaced an earlier remote MLX judge on a separate Mac (`mlx-community/gemma-4-31b-it-nvfp4`) — moved local for speed; fp8 is not a precision regression vs. that 4-bit build.

**Experiment convention:** each `experiment/expN/` has `settings.json` (model, condition `direct`|`reasoning`, decoding params, `n_examples`, `seed`, `judge`/`ex_recall` toggles, dataset list, `exp_id` linking back to `project_doc.md`'s Experiment Log), `outputs/` (raw generations, never overwritten by scoring), `eval/` (scored per-example results), `run.log`, and `summary.json`. To add a new experiment: copy `experiment/exp1/` → `experiment/expN/`, edit `settings.json`, copy `docs/experiments/template.md` → `expN.md`, add a row to `docs/README.md`'s index table.

**When an experiment finishes:** fill in `docs/experiments/expN.md`'s `status` frontmatter and `Result`/`What we learned`/`Decision` fields (pattern: `docs/experiments/exp1.md`) from the run's `summary.json`, and flip its row's `Status` to `Done` in `docs/README.md`. Do this for every finished run, not just the newest — a `Planned`/`Running` doc for a run that already has a populated `summary.json` is stale.

**Prompting condition** (`direct` vs `reasoning`) is controlled by `enable_thinking` passed to the tokenizer's chat template in `scripts/generate.py`, not by a different system prompt — the same `SYSTEM_PROMPT` (answer-then-wrap-in-`<answer>` tags) is used for both conditions.

## Updating project state

`project_doc.md` holds the live experiment log, current understanding, plan, and meeting notes — update it after an experiment finishes (per its own header instruction), not just after code changes.

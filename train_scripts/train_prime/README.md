# SFT on synthetic QA with prime-rl

prime-rl replacement for the TRL script `train_synthetic/sft.py`. Pretraining
(`train_synthetic/pretrain.py`) and eval (`train_synthetic/eval.py`) are unchanged.

```
pretrain ckpt (HF/TRL) ──prepare_model.py──▶ prime-rl-loadable copy (key prefix fixed)
qa/train/*.jsonl ──prepare_data.py──▶ data/sft_prime/<cfg>/train.jsonl + configs/<cfg>.toml
                                              │
           sft @ configs/base.toml @ configs/<cfg>.toml   (prime-rl trainer, DCP ckpt)
                                              │
           tools/convert_dcp_to_bf16.py        (DCP ──▶ HF safetensors in step_N/weights)
                                              │
           train_synthetic/eval.py             (unchanged; vLLM, direct + reasoning)
```

## Environment

Nothing is added to this repo's `pyproject.toml`. Training, export and eval all run in
prime-rl's own locked environment (`~/prime-rl/.venv`: torch 2.13, vLLM 0.32,
transformers 5.19), invoked as `uv run --no-sync --project ~/prime-rl ...`.

- `--no-sync` matters. A plain `uv run` would re-sync that venv to its default deps and
  uninstall the extras and env packages it was installed with.
- eval.py's vLLM imports (`qwen3_5`, `mamba_utils`) resolve in that environment.
- **Local prime-rl patch (required):** `patches/prime-rl-local.patch` (23 lines, on top of
  prime-rl `a566e5d8e`) adds two things stock prime-rl lacks:
  1. tied `lm_head`/`embed_tokens` for Qwen3.5, mirroring prime-rl's own Qwen3 class;
  2. a `[data.loss_mask] generation_prompt` option (default `true` = stock behavior). We set it to
     `false` to mask the thinking prefill, as TRL does.

  Apply it once: `git -C ~/prime-rl apply $PWD/train_scripts/train_prime/patches/prime-rl-local.patch`.
  Without it, `base.toml` fails validation (unknown key) and tied checkpoints train wrong.
- prime-rl doesn't need sudo. Set it up with `git clone`, then
  `git submodule update --init ...`, then `uv sync --all-extras --all-packages`.

## Training configurations

The sampling unit is the **question** (`qaid`). The direct and reasoning files share qaids,
so each sampled question contributes both of its rows, as in the full set. That makes
direct:reasoning exactly 1:1, and stratifying by (hop, pattern) keeps 1-hop:2-hop at
about 47:53. The sets are nested: c2 ⊂ c3 ⊂ c4 ⊂ c1.

| cfg | definition | questions | rows | max_steps (2 ep) |
|---|---|---|---|---|
| `c2_10pct` | 10% | 2,950 | 5,900 | 349 |
| `c3_19pct` | c2 + 10% of remaining 90% (= 19%, not 18%) | 5,604 | 11,208 | 662 |
| `c4_43pct` | c3 + 30% of remaining 81% (= 43.3%) | 12,772 | 25,544 | 1,509 |
| `c1_full` | 100% | 29,497 | 58,994 | 3,481 |

(Counts are for `qa/train`, seed 100. The exact qaids are in `data/sft_prime/<cfg>/qaids.txt`,
and the stats are in `data/sft_prime/manifest.json`.)

Every config trains for the same number of **epochs** (2, as in the TRL recipe), so
smaller sets also get proportionally fewer steps. For a fixed-compute comparison instead,
override `--max-steps` and `--scheduler.warmup-steps`.

## Usage (from repo root)

```bash
# 0. one-time: make the TRL/HF pretrain checkpoint loadable by prime-rl
#    (renames model.language_model.* -> model.*; keeps tied embeddings tied, needs the patch above)
uv run --no-sync --project ~/prime-rl python train_scripts/train_prime/prepare_model.py \
    pretrain-qwen3.5-0.8b checkpoints/pretrain-qwen3.5-0.8b-prime

# 1. build data + overlays (needs the pretrained checkpoint's tokenizer; ~15 s)
uv run --no-sync --project ~/prime-rl python train_scripts/train_prime/prepare_data.py \
    --qa-dir qa/train --tokenizer checkpoints/pretrain-qwen3.5-0.8b-prime
#    (qa_ambiguous: --qa-dir qa_ambiguous/train --out-dir data/sft_prime_ambiguous; also
#     use run_sft.sh --qa-test qa_ambiguous/test)

# 2. train -> export -> eval, one config or all four (c2, c3, c4, c1)
M=checkpoints/pretrain-qwen3.5-0.8b-prime
bash train_scripts/train_prime/run_sft.sh c2_10pct --model $M --gpus 0
bash train_scripts/train_prime/run_sft.sh all --model $M --gpus 0

# smoke test: 5 steps + export + eval on a small test slice
bash train_scripts/train_prime/run_sft.sh c2_10pct --model $M --prefix smoke --gpus 1 \
    --qa-test <dir with a few lines per test file> -- --max-steps 5 --scheduler.warmup-steps 1

# long runs: detach in tmux
tmux new -d -s sft_sweep "bash train_scripts/train_prime/run_sft.sh all --model $M --gpus 0 2>&1 | tee experiment/prime-sft-sweep.log"
```

Outputs:
- `checkpoints/prime-sft/<run>/checkpoints/step_N/weights/` holds the HF export (~1.5 GB). The DCP trainer
  state next to it (`trainer/`, ~9.4 GB with tied embeddings) is deleted once the export is verified,
  because it's only needed to resume a run. Pass `--keep-dcp` to keep it.
- `checkpoints/prime-sft/<run>/logs/` has the trainer logs, and `configs/` has the resolved config.
- `experiment/prime-sft/<run>/{direct,reasoning}.jsonl` and `summary.json` hold the eval results.

Other model families: rerun `prepare_data.py` with that model's `--tokenizer` and
`--run-prefix`, then pass `--model <ckpt> --prefix <name> -- --renderer.name gemma4`
(or `nemotron-3`), and also change `[renderer].name` in `base.toml` before running
`prepare_data.py`.

## Differences from the TRL script

| | TRL `sft.py` | prime-rl |
|---|---|---|
| chat formatting | HF chat template, `chat_template_kwargs.enable_thinking` per row | `qwen3.5` renderer, per-row `enable_thinking` column via `[data.columns.renderer]` |
| batching | 32 padded samples/step | packed rows: 2 × 1024 tokens ≈ 34 samples/step |
| length | `num_train_epochs` | `max_steps`, precomputed from the packed token count |
| parallelism | DeepSpeed ZeRO | FSDP2 (`deployment.num_train_gpus`) |
| checkpoint | HF `save_model` | DCP, exported by `convert_dcp_to_bf16.py` |
| loss tokens | completion only | completion only; thinking prefill masked via the local `generation_prompt = false` |
| weight names | `model.language_model.*` | `model.*` (fixed once by `prepare_model.py`) |
| tied embeddings | supported | supported **with the local patch** (stock prime-rl: not supported) |

**Prefill tokens.** I checked token-for-token: the renderer's full sequence is identical to
the HF chat template, and eval.py's prompt ids are an exact prefix of the training ids in both
modes. Stock prime-rl counts the generation-prompt prefill (`<think>\n` for reasoning,
`<think>\n\n</think>\n\n` for direct) as assistant tokens and trains on it. TRL masks it as
prompt. With `generation_prompt = false`, the patched prime-rl masks every token of
`render(prompt, add_generation_prompt=True)`, so the trained tokens match TRL exactly. For example,
a direct row trains on `Sarvyn Plovik<|im_end|>` (7 tokens) instead of
`<think>\n\n</think>\n\nSarvyn Plovik<|im_end|>` (11). The patch raises an error if that render
isn't a prefix of the training render.

**Tied embeddings.** Qwen3.5-0.8B shares one matrix between `embed_tokens` and `lm_head`, and
TRL keeps them shared during SFT. Stock prime-rl's Qwen3.5 has no tying: on a tied checkpoint it
trains from an uninitialized `lm_head` (step-1 loss ≈ ln(vocab) = 12.4, with no error) and
`convert_dcp_to_bf16.py` crashes on export. The local patch adds tying, and I verified it holds
through training: the DCP checkpoint has optimizer state only for `embed_tokens`, meaning one
shared parameter. The export writes no `lm_head`, keeps `tie_word_embeddings: true`, and eval.py
loads it. `prepare_model.py --untie` is the fallback for an unpatched prime-rl, but it changes
training (two separately updated matrices, about 254M extra trainable parameters).

## Eval

`train_synthetic/eval.py` is the **same eval**: the same test files, both modes, substring
accuracy after `</think>`, reported overall and per hop, after every config. It needed **one
compatibility fix**. prime-rl's vLLM 0.32 renamed `AutoWeightsLoader(skip_prefixes=...)` to
`ignore_unexpected_prefixes`, so eval.py now passes whichever name the installed vLLM
accepts. Behavior on older vLLMs is unchanged. The checkpoint it loads is the HF export
in `step_N/weights` (`Qwen3_5ForCausalLM` with tokenizer and chat template).

Note: eval.py's docstring says greedy decoding, but it samples (temperature 1.0, top_p 0.6,
top_k 60) with no seed, so expect some run-to-run noise in accuracy.

## Verified (smoke test, Qwen3.5-0.8B-Base stand-in, GPU1)

prepare_model → 5 SFT steps (loss 3.68 → 1.95) → DCP→HF export → eval.py on 25
questions per test file ran end to end. Accuracy was 0%, as expected for a base model
that never saw the biographies, but outputs already followed the trained
`<think>…That would be X.</think> X` format.

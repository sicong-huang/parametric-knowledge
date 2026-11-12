# prime-rl SFT commands (qwen3.5-0.8b)

Run all commands from the repo root. Paste **one line at a time**. The launch lines are kept
short on purpose, because a long command that wraps on paste breaks (bash then runs the
model path as a command).

## Status

| config | data | steps | ~time (train + eval) | status | direct | reasoning |
|---|---|---|---|---|---|---|
| `c2_10pct` | 10% · 5,900 rows | 349 | 10 + 10 min | **done** (2026-11-05) | 16.2% | 34.0% |
| `c3_19pct` | 19% · 11,208 rows | 662 | 19 + 10 min | **done** (2026-11-05) | 18.6% | 38.9% |
| `c4_43pct` | 43.3% · 25,544 rows | 1,509 | 43 + 10 min | **done** (2026-11-05) | 21.7% | 44.3% |
| `c1_full` | 100% · 58,994 rows | 3,481 | 1 h 40 min + 10 min | **done** (2026-11-06) | 25.8% | 51.8% |

Accuracy is `eval.py`'s substring accuracy on all 12,543 `qa/test` questions per mode. The
1-hop/2-hop split is in each run's `summary.json`.

## One-time setup (already done on nlp-sv11)

```bash
git -C ~/prime-rl apply $PWD/train_scripts/train_prime/patches/prime-rl-local.patch
uv run --no-sync --project ~/prime-rl python train_scripts/train_prime/prepare_model.py \
    pretrain-qwen3.5-0.8b checkpoints/pretrain-qwen3.5-0.8b-prime
uv run --no-sync --project ~/prime-rl python train_scripts/train_prime/prepare_data.py \
    --qa-dir qa/train --tokenizer checkpoints/pretrain-qwen3.5-0.8b-prime
```

## Shared variables (set these first, in every new shell)

Shell variables don't carry over to a new terminal or an SSH reconnect. If `$M` is empty, the
command becomes `--model --gpus 1`; `run_sft.sh` now catches this and stops with *"option value
'--gpus' looks like a flag; is a shell variable (e.g. $M) unset?"*. (Before that check was added,
prime-rl treated `--gpus` as a Hugging Face model id and failed trying to download it.)

```bash
cd ~/parametric-knowledge
M=checkpoints/pretrain-qwen3.5-0.8b-prime
S=train_scripts/train_prime/run_sft.sh
mkdir -p experiment/prime-sft
nvidia-smi        # pick a free GPU; set --gpus below accordingly
```

Each config section below sets `LOG`, then **echoes all three variables before launching**. If any
of them prints empty, rerun the lines above first.

Each launch below trains, exports to HF, deletes the 9.4 GB DCP once the export is verified,
and then evaluates on `qa/test`. `nohup setsid` detaches the job, so it survives closing the
terminal or losing SSH.

## c2_10pct

```bash
LOG=experiment/prime-sft/c2_10pct.log
echo "M=$M S=$S LOG=$LOG"
nohup setsid bash $S c2_10pct --model $M --gpus 1 >$LOG 2>&1 </dev/null &
```

## c3_19pct

```bash
LOG=experiment/prime-sft/c3_19pct.log
echo "M=$M S=$S LOG=$LOG"
nohup setsid bash $S c3_19pct --model $M --gpus 1 >$LOG 2>&1 </dev/null &
```

## c4_43pct (done)

```bash
LOG=experiment/prime-sft/c4_43pct.log
echo "M=$M S=$S LOG=$LOG"
nohup setsid bash $S c4_43pct --model $M --gpus 1 >$LOG 2>&1 </dev/null &
```

## c1_full

```bash
LOG=experiment/prime-sft/c1_full.log
echo "M=$M S=$S LOG=$LOG"
nohup setsid bash $S c1_full --model $M --gpus 1 >$LOG 2>&1 </dev/null &
```

## All remaining configs, one after another (c2 → c3 → c1, about 3 h)

```bash
LOG=experiment/prime-sft/sweep.log
echo "M=$M S=$S LOG=$LOG"
nohup setsid bash $S c2_10pct c3_19pct c1_full --model $M --gpus 1 >$LOG 2>&1 </dev/null &
```

The two GPUs are separate, so you can also run two configs at once, for example c2 on
`--gpus 0` and c3 on `--gpus 1`. Give each its own `LOG`.

## Monitoring

```bash
tail -f $LOG                                                        # wrapper: train/export/eval stages
grep -a "Step" checkpoints/prime-sft/qwen3.5-0.8b-<cfg>/logs/latest/trainer.log | tail -3   # training steps
nvidia-smi                                                          # ~15-20 GB on the GPU while training
```

`Ctrl-C` on `tail` only stops watching. To stop a run:
```bash
pkill -f "run_sft.sh <cfg>"; pkill -f "sft @"
```

## Re-runs and fixes

```bash
# eval only (model already trained + exported), e.g. after an eval.py change
nohup setsid bash $S <cfg> --gpus 1 --skip-train >experiment/prime-sft/<cfg>_eval.log 2>&1 </dev/null &

# retrain from scratch when a previous attempt died mid-training (prime-rl refuses a run dir
# with artifacts; an attempt that failed before training starts leaves at most an empty dir,
# which is fine to `rmdir` instead)
nohup setsid bash $S <cfg> --model $M --gpus 1 -- --clean >$LOG 2>&1 </dev/null &

# keep the 9.4 GB DCP (to resume training later)
nohup setsid bash $S <cfg> --model $M --gpus 1 --keep-dcp >$LOG 2>&1 </dev/null &

# free eval's staged model copies (cache only; eval re-copies when needed)
rm -rf /tmp/staged-models/weights-*
```

## Where results go

| what | path |
|---|---|
| trained model (HF) | `checkpoints/prime-sft/qwen3.5-0.8b-<cfg>/checkpoints/step_<N>/weights/` |
| trainer log | `checkpoints/prime-sft/qwen3.5-0.8b-<cfg>/logs/latest/trainer.log` |
| resolved config | `checkpoints/prime-sft/qwen3.5-0.8b-<cfg>/configs/latest/resolved/sft.json` |
| eval summary | `experiment/prime-sft/qwen3.5-0.8b-<cfg>/summary.json` |
| per-question outputs | `experiment/prime-sft/qwen3.5-0.8b-<cfg>/{direct,reasoning}.jsonl` |
| dataset summary (all configs) | `data/sft_prime/manifest.json` |

---

# RL (GRPO) on top of c4_43pct

Starts from `checkpoints/prime-sft/qwen3.5-0.8b-c4_43pct/checkpoints/step_1509/weights` and trains
only on the 16,725 training questions c4 never saw. Configs are nested and stratified by (hop,
pattern), using the same 10 / 19 / 43.3 / 100% rule as SFT, but as fractions of that held-out
pool. Every question is used once with thinking off (direct) and once with thinking on
(reasoning), alternating strictly. Reward is binary normalized exact match on the final answer
(text after `</think>`). Each prompt gets 8 rollouts; there are 64 prompts per step and 1 epoch.

| config | pool share | questions | prompts | steps | status | direct | reasoning |
|---|---|---|---|---|---|---|---|
| `r1_10pct` | 10% | 1,672 | 3,344 | 53 | lr sweep pending (below) | | |
| `r2_19pct` | 19% | 3,179 | 6,358 | 100 | not run | | |
| `r3_43pct` | 43.3% | 7,242 | 14,484 | 227 | not run | | |
| `r4_full` | 100% | 16,725 | 33,450 | 523 | not run | | |

"1 epoch" = `2 × questions / 64` steps of trained groups. Groups whose 8 rollouts are all
right or all wrong are dropped and re-sampled, so in practice that is about 3 passes over the
prompts (r1: 10,764 dispatched for 3,344 prompts).

### r1 lr sweep (per-prompt loss weighting)

Direct / reasoning are 1-hop / 2-hop / overall on qa/test. c4 SFT (start point):
33.7 / 10.8 / 21.7 and 60.3 / 29.7 / 44.3.

| run | lr | loss weighting | status | direct | reasoning |
|---|---|---|---|---|---|
| `qwen3.5-0.8b-c4-r1_10pct` (old) | 1e-6 | token | done | 35.0 / 11.3 / 22.7 | 60.7 / 30.4 / 44.9 |
| `qwen3.5-0.8b-c4-r1_10pct-lr3e-6` | 3e-6 | prompt | not run | | |
| `qwen3.5-0.8b-c4-r1_10pct-lr1e-5` | 1e-5 | prompt | not run | | |

Pick the lr from these runs, set it as `[trainer.optim].lr` in `configs/rl/base.toml`, then run
r2–r4. Also re-run r1 without `--lr`, or keep the winning sweep run as r1. Why per-prompt:
`rl_loss_weighting.md`.

**RL needs both GPUs** (GPU 0 runs vLLM rollouts, GPU 1 runs the trainer), so nothing else can
run on the machine at the same time. In a 3-step smoke test, steps took 1–2 min, but that
includes compile warm-up. The real time per step will come from the r1 run.

## RL one-time setup (already done on nlp-sv11)

```bash
git -C ~/prime-rl apply $PWD/train_scripts/train_prime/patches/prime-rl-local.patch   # same patch as SFT, now 3 more changes
python3 train_scripts/train_prime/prepare_rl_data.py     # data/rl_prime/* + configs/rl/r*.toml
```

## RL launch (one config at a time)

```bash
cd ~/parametric-knowledge
R=train_scripts/train_prime/run_rl.sh
mkdir -p experiment/prime-rl
nvidia-smi        # both GPUs must be free

LOG=experiment/prime-rl/r1_10pct.log
echo "R=$R LOG=$LOG"
nohup setsid bash $R r1_10pct --gpus 0,1 >$LOG 2>&1 </dev/null &
```

**lr sweep on r1:** two runs, one after the other (~1 h each, plus ~10 min eval each):

```bash
LOG=experiment/prime-rl/r1_10pct-lrsweep.log
nohup setsid bash -c "bash $R r1_10pct --gpus 0,1 --lr 3e-6 && bash $R r1_10pct --gpus 0,1 --lr 1e-5" \
  >$LOG 2>&1 </dev/null &
```

`--lr X` overrides `[trainer.optim].lr` and appends `-lrX` to the run name, which gives
`qwen3.5-0.8b-c4-r1_10pct-lrX` everywhere (checkpoints, logs, `experiment/prime-rl/`).

For the other configs, replace `r1_10pct` with `r2_19pct`, `r3_43pct` or `r4_full`, in both
`LOG` and the launch line. To run all four in sequence:
`nohup setsid bash $R all --gpus 0,1 >experiment/prime-rl/sweep.log 2>&1 </dev/null &`

## RL monitoring

```bash
tail -f $LOG                                                                  # wrapper stages
grep -a "Step " checkpoints/prime-rl/qwen3.5-0.8b-c4-<cfg>/logs/latest/orchestrator.log | tail -3   # step, time, reward
```

The orchestrator line shows the step number, its time and the mean reward over both modes.
Per-mode reward (`train/direct/...`, `train/reasoning/...`) is in
`checkpoints/prime-rl/<run>/monitors/file/metrics.jsonl`.

To stop a run: `pkill -f "run_rl.sh"; pkill -f "rl @ "`. Then check `nvidia-smi` and confirm
both GPUs are empty.

## RL re-runs

Re-runs use the same flags as SFT:
- `--skip-train`: eval only.
- `-- --clean`: retrain after a run that died mid-training.
- `--keep-dcp`: keep the resumable checkpoint.
- `--model PATH`: start from a different SFT checkpoint.

## RL outputs

| what | path |
|---|---|
| trained model (HF) | `checkpoints/prime-rl/qwen3.5-0.8b-c4-<cfg>/checkpoints/step_<N>/weights/` |
| logs (orchestrator / trainer / inference / envs) | `checkpoints/prime-rl/qwen3.5-0.8b-c4-<cfg>/logs/latest/` |
| per-step metrics | `checkpoints/prime-rl/qwen3.5-0.8b-c4-<cfg>/monitors/file/metrics.jsonl` |
| eval summary | `experiment/prime-rl/qwen3.5-0.8b-c4-<cfg>/summary.json` |
| task sets + manifest | `data/rl_prime/<cfg>/tasks.jsonl`, `data/rl_prime/manifest.json` |

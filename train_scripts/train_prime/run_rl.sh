#!/usr/bin/env bash
# RL -> export -> eval for one or more prime-rl RL configs (same shape as run_sft.sh).
#
#   bash train_scripts/train_prime/run_rl.sh r1_10pct                      # one config
#   bash train_scripts/train_prime/run_rl.sh all                           # r1, r2, r3, r4 in order
#   bash train_scripts/train_prime/run_rl.sh r1_10pct -- --max-steps 2     # extra args after -- go to `rl`
#
# Options:
#   --model PATH     starting checkpoint (default: [model].name in configs/rl/base.toml = c4_43pct SFT)
#   --prefix NAME    run.name = <prefix>-<cfg>  (default: from the overlay, qwen3.5-0.8b-c4-<cfg>)
#   --lr LR          trainer learning rate (default: [trainer.optim].lr in configs/rl/base.toml);
#                    also appends -lr<LR> to run.name so sweep runs get their own dirs
#   --gpus IDS       CUDA_VISIBLE_DEVICES; needs 2 GPUs (1 vLLM inference + 1 trainer) (default: 0,1)
#   --qa-test DIR    test split for eval.py (default: qa/test)
#   --output-dir DIR RL runs root (default: output_dir in configs/rl/base.toml)
#   --eval-dir DIR   eval results root (default: experiment/prime-rl)
#   --skip-train / --skip-eval
#   --keep-dcp       keep the DCP trainer state after export (deleted by default once the export is verified)
#
# Needs the local prime-rl patch (patches/prime-rl-local.patch): tied Qwen3.5 embeddings in the
# trainer, and text-only Qwen3.5 in prime-rl's vLLM server. The kg_qa taskset (envs/kg_qa) is put
# on PYTHONPATH here, so nothing is installed into prime-rl's environment.
#
# Outputs:
#   checkpoints/prime-rl/<run>/checkpoints/step_<N>/weights/  HF safetensors (what eval loads)
#   experiment/prime-rl/<run>/{direct,reasoning}.jsonl, summary.json
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HERE="$ROOT/train_scripts/train_prime"
CONF="$HERE/configs/rl"
PRIME_RL_DIR="${PRIME_RL_DIR:-$HOME/prime-rl}"
cd "$ROOT"

MODEL="" PREFIX="" LR="" QA_TEST="qa/test" SKIP_TRAIN=0 SKIP_EVAL=0 KEEP_DCP=0 OUTPUT_DIR="" EVAL_DIR="experiment/prime-rl"
GPUS="${CUDA_VISIBLE_DEVICES:-0,1}"
CFGS=() EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --prefix) PREFIX="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --qa-test) QA_TEST="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --eval-dir) EVAL_DIR="$2"; shift 2 ;;
    --skip-train) SKIP_TRAIN=1; shift ;;
    --skip-eval) SKIP_EVAL=1; shift ;;
    --keep-dcp) KEEP_DCP=1; shift ;;
    --) shift; EXTRA=("$@"); break ;;
    all) CFGS+=(r1_10pct r2_19pct r3_43pct r4_full); shift ;;
    -*) echo "unknown option $1" >&2; exit 2 ;;
    *) CFGS+=("$1"); shift ;;
  esac
done
[[ ${#CFGS[@]} -gt 0 ]] || { sed -n '2,26p' "$0"; exit 2; }
for v in "$MODEL" "$GPUS" "$PREFIX" "$LR" "$QA_TEST" "$OUTPUT_DIR" "$EVAL_DIR"; do
  [[ "$v" != -* ]] || { echo "option value '$v' looks like a flag; is a shell variable unset?" >&2; exit 2; }
done
[[ -z "$MODEL" || -d "$MODEL" ]] || { echo "--model '$MODEL' is not a directory (cwd: $ROOT)" >&2; exit 2; }
[[ "$GPUS" == *,* ]] || { echo "--gpus '$GPUS': RL needs two GPUs (inference + trainer), e.g. --gpus 0,1" >&2; exit 2; }

export CUDA_VISIBLE_DEVICES="$GPUS"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export PATH="$CUDA_HOME/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONPATH="$HERE/envs${PYTHONPATH:+:$PYTHONPATH}"
# Alternate the direct/reasoning sources strictly (local patch) instead of prime-rl's weighted
# random draw, so one epoch dispatches every question exactly once per mode.
export PRL_ROUND_ROBIN_SOURCES=1

prl() { uv run --no-sync --project "$PRIME_RL_DIR" "$@"; }
toml_get() { python3 -c "import sys,tomllib; d=tomllib.load(open(sys.argv[1],'rb'))
for k in sys.argv[2].split('.'): d=d[k]
print(d)" "$1" "$2"; }

OUTPUT_DIR="${OUTPUT_DIR:-$(toml_get "$CONF/base.toml" output_dir)}"

for cfg in "${CFGS[@]}"; do
  overlay="$CONF/$cfg.toml"
  [[ -f "$overlay" ]] || { echo "missing $overlay — run prepare_rl_data.py first" >&2; exit 1; }
  run="$(toml_get "$overlay" run.name)"
  [[ -n "$PREFIX" ]] && run="$PREFIX-$cfg"
  [[ -n "$LR" ]] && run="$run-lr$LR"
  args=(@ "$CONF/base.toml" @ "$overlay" --run.name "$run" --output-dir "$OUTPUT_DIR")
  [[ -n "$MODEL" ]] && args+=(--model.name "$MODEL")
  [[ -n "$LR" ]] && args+=(--trainer.optim.lr "$LR")

  echo "=== [$cfg] run=$run gpus=$CUDA_VISIBLE_DEVICES${LR:+ lr=$LR}"
  if [[ $SKIP_TRAIN -eq 0 ]]; then
    prl rl "${args[@]}" "${EXTRA[@]}"
  fi

  ckpt="$(ls -d "$OUTPUT_DIR/$run"/checkpoints/step_* 2>/dev/null | sort -V | tail -1 || true)"
  [[ -n "$ckpt" ]] || { echo "no checkpoint under $OUTPUT_DIR/$run/checkpoints" >&2; exit 1; }
  if [[ ! -f "$ckpt/weights/config.json" ]]; then
    echo "=== [$cfg] exporting $ckpt -> $ckpt/weights"
    prl python "$PRIME_RL_DIR/tools/convert_dcp_to_bf16.py" "$ckpt"
  fi

  # Trainer->vLLM weight-sync files (filesystem broadcast); only needed while training runs.
  if [[ -d "$OUTPUT_DIR/$run/broadcasts" ]]; then
    echo "=== [$cfg] removing weight-sync dir $OUTPUT_DIR/$run/broadcasts ($(du -sh "$OUTPUT_DIR/$run/broadcasts" | cut -f1))"
    rm -rf "$OUTPUT_DIR/$run/broadcasts"
  fi

  if [[ $KEEP_DCP -eq 0 && -d "$ckpt/trainer" ]]; then
    if [[ -f "$ckpt/weights/config.json" ]] && compgen -G "$ckpt/weights/*.safetensors" >/dev/null \
       && [[ -z "$(find "$ckpt/weights" -name '*.safetensors' -size 0)" ]]; then
      echo "=== [$cfg] export verified; deleting DCP $ckpt/trainer ($(du -sh "$ckpt/trainer" | cut -f1))"
      rm -rf "$ckpt/trainer"
    else
      echo "=== [$cfg] WARNING: export incomplete in $ckpt/weights; keeping DCP" >&2
    fi
  fi

  if [[ $SKIP_EVAL -eq 0 ]]; then
    echo "=== [$cfg] eval on $QA_TEST"
    CUDA_VISIBLE_DEVICES="${GPUS%%,*}" prl python train_scripts/train_synthetic/eval.py \
      --model "$ckpt/weights" --qa-dir "$QA_TEST" \
      --output "$EVAL_DIR/$run" --max-new-tokens 256
  fi
done

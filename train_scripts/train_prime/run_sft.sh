#!/usr/bin/env bash
# Train -> export -> eval for one or more prime-rl SFT configs.
#
#   bash train_scripts/train_prime/run_sft.sh c2_10pct                     # one config
#   bash train_scripts/train_prime/run_sft.sh all                          # c2, c3, c4, c1 in order
#   bash train_scripts/train_prime/run_sft.sh c1_full --model checkpoints/pretrain-gemma4-e2b \
#        --prefix gemma4-e2b -- --renderer.name gemma4                     # extra args after -- go to `sft`
#
# Options:
#   --model PATH     pretrained checkpoint (default: [model].name in configs/base.toml)
#   --prefix NAME    run.name = <prefix>-<cfg>  (default: from the overlay, qwen3.5-0.8b-<cfg>)
#   --gpus IDS       CUDA_VISIBLE_DEVICES for train/export/eval (default: $CUDA_VISIBLE_DEVICES or 0)
#   --qa-test DIR    test split for eval.py (default: qa/test)
#   --output-dir DIR training runs root (default: output_dir in configs/base.toml)
#   --eval-dir DIR   eval results root (default: experiment/prime-sft)
#   --skip-train / --skip-eval
#   --keep-dcp       keep the DCP trainer state (step_<N>/trainer, ~9.4 GB for 0.8B) after export;
#                    by default it is deleted once the HF export is verified (it's only needed to resume)
#
# Everything runs in prime-rl's own environment (`uv run --project $PRIME_RL_DIR`), never this
# repo's: prime-rl pins torch/vLLM/transformers via its lockfile, and its vLLM (0.32) is also
# what eval.py's Qwen3.5 registration imports.
#
# Outputs:
#   checkpoints/prime-sft/<run>/checkpoints/step_<N>/trainer/  DCP checkpoint (deleted after export unless --keep-dcp)
#   checkpoints/prime-sft/<run>/checkpoints/step_<N>/weights/  HF safetensors (what eval loads)
#   experiment/prime-sft/<run>/{direct,reasoning}.jsonl, summary.json
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HERE="$ROOT/train_scripts/train_prime"
PRIME_RL_DIR="${PRIME_RL_DIR:-$HOME/prime-rl}"
cd "$ROOT"

MODEL="" PREFIX="" QA_TEST="qa/test" SKIP_TRAIN=0 SKIP_EVAL=0 KEEP_DCP=0 OUTPUT_DIR="" EVAL_DIR="experiment/prime-sft"
GPUS="${CUDA_VISIBLE_DEVICES:-0}"
CFGS=() EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --prefix) PREFIX="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --qa-test) QA_TEST="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --eval-dir) EVAL_DIR="$2"; shift 2 ;;
    --skip-train) SKIP_TRAIN=1; shift ;;
    --skip-eval) SKIP_EVAL=1; shift ;;
    --keep-dcp) KEEP_DCP=1; shift ;;
    --) shift; EXTRA=("$@"); break ;;
    all) CFGS+=(c2_10pct c3_19pct c4_43pct c1_full); shift ;;
    -*) echo "unknown option $1" >&2; exit 2 ;;
    *) CFGS+=("$1"); shift ;;
  esac
done
[[ ${#CFGS[@]} -gt 0 ]] || { sed -n '2,19p' "$0"; exit 2; }
# An unset shell variable (e.g. `--model $M` in a fresh shell) makes the next flag the value:
# `--model --gpus` would be sent to prime-rl as an HF Hub repo id and fail far from the cause.
for v in "$MODEL" "$GPUS" "$PREFIX" "$QA_TEST" "$OUTPUT_DIR" "$EVAL_DIR"; do
  [[ "$v" != -* ]] || { echo "option value '$v' looks like a flag; is a shell variable (e.g. \$M) unset?" >&2; exit 2; }
done
[[ -z "$MODEL" || -d "$MODEL" ]] || { echo "--model '$MODEL' is not a directory (cwd: $ROOT)" >&2; exit 2; }
[[ " ${CFGS[*]} " != *" 1 "* && " ${CFGS[*]} " != *" 0 "* ]] || { echo "config list contains a GPU id (${CFGS[*]}); check --gpus" >&2; exit 2; }

export CUDA_VISIBLE_DEVICES="$GPUS"
# This machine's PATH resolves nvcc to a stray toolkit (see CLAUDE.md); point JIT
# compilers at the real CUDA 12.8 and keep vLLM off the flashinfer sampler.
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export PATH="$CUDA_HOME/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0

# --no-sync: a plain `uv run` would re-sync prime-rl's venv to its default deps and strip the
# extras / env packages it was installed with (uv sync --all-extras --all-packages).
prl() { uv run --no-sync --project "$PRIME_RL_DIR" "$@"; }
toml_get() { python3 -c "import sys,tomllib; d=tomllib.load(open(sys.argv[1],'rb'))
for k in sys.argv[2].split('.'): d=d[k]
print(d)" "$1" "$2"; }

OUTPUT_DIR="${OUTPUT_DIR:-$(toml_get "$HERE/configs/base.toml" output_dir)}"

for cfg in "${CFGS[@]}"; do
  overlay="$HERE/configs/$cfg.toml"
  [[ -f "$overlay" ]] || { echo "missing $overlay — run prepare_data.py first" >&2; exit 1; }
  run="$(toml_get "$overlay" run.name)"
  [[ -n "$PREFIX" ]] && run="$PREFIX-$cfg"
  args=(@ "$HERE/configs/base.toml" @ "$overlay" --run.name "$run" --output-dir "$OUTPUT_DIR")
  [[ -n "$MODEL" ]] && args+=(--model.name "$MODEL")

  echo "=== [$cfg] run=$run gpus=$CUDA_VISIBLE_DEVICES"
  if [[ $SKIP_TRAIN -eq 0 ]]; then
    prl sft "${args[@]}" "${EXTRA[@]}"
  fi

  ckpt="$(ls -d "$OUTPUT_DIR/$run"/checkpoints/step_* 2>/dev/null | sort -V | tail -1 || true)"
  [[ -n "$ckpt" ]] || { echo "no checkpoint under $OUTPUT_DIR/$run/checkpoints" >&2; exit 1; }
  if [[ ! -f "$ckpt/weights/config.json" ]]; then
    echo "=== [$cfg] exporting $ckpt -> $ckpt/weights"
    prl python "$PRIME_RL_DIR/tools/convert_dcp_to_bf16.py" "$ckpt"
  fi

  # Free the DCP trainer state once the export is verified (config + non-empty safetensors).
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
    prl python train_scripts/train_synthetic/eval.py \
      --model "$ckpt/weights" --qa-dir "$QA_TEST" \
      --output "$EVAL_DIR/$run" --max-new-tokens 256
  fi
done

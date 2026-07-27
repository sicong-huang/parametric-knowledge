#!/usr/bin/env bash
# ── pretrain job submission wrapper ─────────────────────────────────────────
# Usage:
#   ./scripts/pretrain.sbatch                                    # sbatch, defaults below
#   ./scripts/pretrain.sbatch --epochs 5 --batch-size 64 --output-dir checkpoints/exp-2
#   ./scripts/pretrain.sbatch --gpus 4 --mem 128G --time 6:00:00 --epochs 10
#   ./scripts/pretrain.sbatch --srun                             # run via srun in this terminal
#   ./scripts/pretrain.sbatch --srun --gpus 1 --epochs 1         # quick interactive test
#
# All flags are optional; unrecognized flags are forwarded to pretrain.py.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PARTITION=gpuA40x4
ACCOUNT=bgnf-delta-gpu
GPUS=2
MEM=128G
CPUS=24
TIME=4:00:00
JOB_NAME=""
MODE=sbatch

EPOCHS=3
BATCH_SIZE=128
GRAD_ACCUM=1
OUTPUT_DIR=checkpoints/default
NOTE=""

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --partition)  PARTITION="$2";  shift 2 ;;
        --account)    ACCOUNT="$2";    shift 2 ;;
        --gpus)       GPUS="$2";       shift 2 ;;
        --mem)        MEM="$2";        shift 2 ;;
        --cpus)       CPUS="$2";       shift 2 ;;
        --time)       TIME="$2";       shift 2 ;;
        --job-name)   JOB_NAME="$2";   shift 2 ;;
        --epochs)     EPOCHS="$2";     shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --grad-accum) GRAD_ACCUM="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --note)       NOTE="$2";       shift 2 ;;
        --srun)       MODE=srun;       shift ;;
        *)            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

JOB_NAME="${JOB_NAME:-$(basename "$OUTPUT_DIR")}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"

# Flags shared by both srun and sbatch — same names, one source of truth.
SLURM_FLAGS=(--partition="${PARTITION}" --account="${ACCOUNT}" --gpus-per-node="${GPUS}"
    --mem="${MEM}" --cpus-per-task="${CPUS}" --time="${TIME}" --job-name="${JOB_NAME}")

TRAIN_ARGS=(
    train_synthetic/pretrain.py
    --epochs "${EPOCHS}"
    --batch-size "${BATCH_SIZE}"
    --grad-accum "${GRAD_ACCUM}"
    --output-dir "${OUTPUT_DIR}"
)
[[ -n "${NOTE}" ]] && TRAIN_ARGS+=(--note "${NOTE}")
TRAIN_ARGS+=(${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"})

JOB_INNER="cd $(printf '%q' "${REPO_ROOT}") && export UV_PROJECT_ENVIRONMENT=/tmp/venv && uv sync --frozen && uv run torchrun --nproc-per-node=${GPUS} $(printf '%q ' "${TRAIN_ARGS[@]}")"

echo "[${MODE}] ${JOB_NAME}: ${JOB_INNER}"

if [[ "${MODE}" == "srun" ]]; then
    srun "${SLURM_FLAGS[@]}" --pty bash -c "${JOB_INNER}"
else
    sbatch "${SLURM_FLAGS[@]}" --output="${OUTPUT_DIR}/%x_%j.log" \
        --wrap="${JOB_INNER}"
fi

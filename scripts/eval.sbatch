#!/usr/bin/env bash
# ── eval job submission wrapper ─────────────────────────────────────────────
# Usage:
#   ./scripts/eval.sbatch --model checkpoints/sft-gemma4-e2b --output experiment/sft-gemma4-e2b
#   ./scripts/eval.sbatch --model checkpoints/sft-gemma4-e2b --output experiment/sft-gemma4-e2b --no-vllm
#   ./scripts/eval.sbatch --model checkpoints/sft-gemma4-e2b --output experiment/sft-gemma4-e2b --gpus 1 --time 1:00:00
#   ./scripts/eval.sbatch --srun --model checkpoints/sft-gemma4-e2b --output experiment/sft-gemma4-e2b  # run via srun in this terminal
#
# --model and --output are required; unrecognized flags are forwarded to eval.py
# (e.g. --qa-dir, --max-new-tokens, --no-vllm).
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PARTITION=gpuA40x4
ACCOUNT=bgnf-delta-gpu
GPUS=1
MEM=64G
CPUS=8
TIME=2:00:00
JOB_NAME=""
MODE=sbatch

MODEL=""
OUTPUT=""

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --partition) PARTITION="$2"; shift 2 ;;
        --account)   ACCOUNT="$2";   shift 2 ;;
        --gpus)      GPUS="$2";      shift 2 ;;
        --mem)       MEM="$2";       shift 2 ;;
        --cpus)      CPUS="$2";      shift 2 ;;
        --time)      TIME="$2";      shift 2 ;;
        --job-name)  JOB_NAME="$2";  shift 2 ;;
        --model)     MODEL="$2";     shift 2 ;;
        --output)    OUTPUT="$2";    shift 2 ;;
        --srun)      MODE=srun;      shift ;;
        *)           EXTRA_ARGS+=("$1"); shift ;;
    esac
done

[[ -n "${MODEL}" ]]  || { echo "error: --model is required" >&2; exit 1; }
[[ -n "${OUTPUT}" ]] || { echo "error: --output is required" >&2; exit 1; }

JOB_NAME="${JOB_NAME:-$(basename "$OUTPUT")}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p "${OUTPUT}"

# Flags shared by both srun and sbatch — same names, one source of truth.
SLURM_FLAGS=(--partition="${PARTITION}" --account="${ACCOUNT}" --gpus-per-node="${GPUS}"
    --mem="${MEM}" --cpus-per-task="${CPUS}" --time="${TIME}" --job-name="${JOB_NAME}")

EVAL_ARGS=(
    train_synthetic/eval.py
    --model "${MODEL}"
    --output "${OUTPUT}"
)
EVAL_ARGS+=(${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"})

JOB_INNER="cd $(printf '%q' "${REPO_ROOT}") && export UV_PROJECT_ENVIRONMENT=/tmp/venv && uv sync --frozen && uv run $(printf '%q ' "${EVAL_ARGS[@]}")"

echo "[${MODE}] ${JOB_NAME}: ${JOB_INNER}"

if [[ "${MODE}" == "srun" ]]; then
    srun "${SLURM_FLAGS[@]}" --pty bash -c "${JOB_INNER}"
else
    sbatch "${SLURM_FLAGS[@]}" --output="${OUTPUT}/%x_%j.log" \
        --wrap="${JOB_INNER}"
fi

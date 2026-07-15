#!/usr/bin/env bash
# Launch an experiment run detached in tmux so it survives beyond one turn
# and can be monitored. Usage: scripts/run_exp.sh <exp>  (e.g. exp1)
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "usage: $0 <exp>" >&2
    exit 1
fi

EXP="$1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="exp_${EXP}"
EXP_DIR="${REPO_ROOT}/experiment/${EXP}"
LOG="${EXP_DIR}/run.log"

if [ ! -f "${EXP_DIR}/settings.json" ]; then
    echo "error: ${EXP_DIR}/settings.json not found" >&2
    exit 1
fi

if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "error: tmux session '${SESSION}' already running -- refusing to start a duplicate GPU job" >&2
    echo "       monitor it with: tmux capture-pane -t ${SESSION} -p | tail -n 40" >&2
    exit 1
fi

mkdir -p "${EXP_DIR}"
# load .env (HF_TOKEN, OPENAI_API_KEY, ...) into the tmux session's env
if [ -f "${REPO_ROOT}/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.env"
    set +a
fi
# VLLM_USE_FLASHINFER_SAMPLER=0: this machine's PATH resolves nvcc to a stray
# nvidia-cuda-toolkit package (CUDA 12.0) instead of the real cuda-nvcc-12-8
# install. flashinfer's top-p/top-k sampler kernel JIT-compiles against CUB
# and needs BlockAdjacentDifference::FlagHeads, which that old toolkit's CUB
# lacks -> vLLM engine crashes on startup. Disabling the flashinfer sampler
# skips that JIT path entirely. If calling scripts/generate.py or
# scripts/run_all.py directly (outside this wrapper) for debugging, prefix
# the same env var or you'll hit the same crash.
# JUDGE_MODEL=gpt-4.1-mini: default JUDGE_MODEL (gpt-4.1) is ~5x more expensive
# for the same SimpleQA-Verified grading convention; at full-eval-split scale
# (n_examples=0) across 8 experiments, full gpt-4.1 would blow a $1500 API
# budget (~$1900-2000) where gpt-4.1-mini lands around $550-650. Override by
# exporting JUDGE_MODEL before calling this script if a run needs the default.
tmux new-session -d -s "${SESSION}" \
    "cd '${REPO_ROOT}' && set -o pipefail; VLLM_USE_FLASHINFER_SAMPLER=0 JUDGE_MODEL=\${JUDGE_MODEL:-gpt-4.1-mini} HF_TOKEN='${HF_TOKEN:-}' OPENAI_API_KEY='${OPENAI_API_KEY:-}' uv run python scripts/run_all.py --exp '${EXP}' 2>&1 | tee '${LOG}'"

echo "started tmux session '${SESSION}', logging to ${LOG}"
echo "monitor: tmux has-session -t ${SESSION} && tail -n 40 ${LOG}"
echo "stop:    tmux kill-session -t ${SESSION}"

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
tmux new-session -d -s "${SESSION}" \
    "cd '${REPO_ROOT}' && set -o pipefail; VLLM_USE_FLASHINFER_SAMPLER=0 uv run python scripts/run_all.py --exp '${EXP}' 2>&1 | tee '${LOG}'"

echo "started tmux session '${SESSION}', logging to ${LOG}"
echo "monitor: tmux has-session -t ${SESSION} && tail -n 40 ${LOG}"
echo "stop:    tmux kill-session -t ${SESSION}"

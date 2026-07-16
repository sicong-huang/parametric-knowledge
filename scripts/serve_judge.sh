#!/usr/bin/env bash
# Launch the local judge/ex-recall model (gemma-4-31b-it, fp8, GPU1) as an
# OpenAI-compatible vLLM server, detached in tmux so it survives beyond one
# turn. Point experiment settings.json's eval_base_url at
# http://localhost:8000/v1 (judge_model / extractor_model: "gemma-4-31b-it")
# to use it -- see scripts/run_all.py, which forwards those settings.json
# keys into OPENAI_BASE_URL / JUDGE_MODEL / EXTRACTOR_MODEL for
# evaluation/simpleqa_judge.py and evaluation/ex_recall.py.
#
# Usage: scripts/serve_judge.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="judge"
LOG="${REPO_ROOT}/experiment/judge_server.log"
MODEL="google/gemma-4-31b-it"
SERVED_NAME="gemma-4-31b-it"
PORT=8000

if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "error: tmux session '${SESSION}' already running -- refusing to start a duplicate judge server" >&2
    echo "       monitor it with: tmux capture-pane -t ${SESSION} -p | tail -n 40" >&2
    exit 1
fi

# load .env (HF_TOKEN, ...) into the tmux session's env
if [ -f "${REPO_ROOT}/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.env"
    set +a
fi

# CUDA_VISIBLE_DEVICES=1: pin the judge to GPU1 only, leaving GPU0 free for
# scripts/generate.py (which loads its model on the default/first visible
# GPU) so generation and judging don't fight over GPU memory.
# --quantization fp8: the 31B judge is bf16 (~62GB) and won't fit on one
# 48GB GPU; fp8 shrinks it to ~31GB. This is no precision regression versus
# the previous remote judge, which served an even more aggressively
# quantized 4-bit (nvfp4) build.
# --max-model-len 40960: reasoning-condition experiments emit long
# <think>...</think> traces (max_tokens up to 32768) that get embedded
# verbatim in the judge/ex-recall grading prompt as predicted_answer/
# raw_output; a tight cap (4096 was tried first) makes vLLM hard-fail
# those requests with a 400 instead of truncating, which crashes the
# whole run_all.py process. 40960 covers a 32768-token trace plus rubric
# overhead; KV cache budget has ample room for it (observed ~37GiB KV
# cache available at gpu_memory_utilization=0.9 with a 4096 cap).
# VLLM_USE_FLASHINFER_SAMPLER=0: same nvcc/flashinfer JIT-compile crash as
# scripts/run_exp.sh -- this machine's PATH resolves nvcc to a stray
# nvidia-cuda-toolkit package instead of the real cuda-nvcc-12-8 install.
tmux new-session -d -s "${SESSION}" \
    "cd '${REPO_ROOT}' && set -o pipefail; CUDA_VISIBLE_DEVICES=1 VLLM_USE_FLASHINFER_SAMPLER=0 HF_TOKEN='${HF_TOKEN:-}' uv run vllm serve '${MODEL}' --served-model-name '${SERVED_NAME}' --quantization fp8 --port ${PORT} --max-model-len 40960 --gpu-memory-utilization 0.9 2>&1 | tee '${LOG}'"

echo "started tmux session '${SESSION}', logging to ${LOG}"
echo "wait for readiness: grep -q 'Uvicorn running' <(tail -f ${LOG})"
echo "check:   curl -s http://localhost:${PORT}/v1/models"
echo "monitor: tmux has-session -t ${SESSION} && tail -n 40 ${LOG}"
echo "stop:    tmux kill-session -t ${SESSION}"

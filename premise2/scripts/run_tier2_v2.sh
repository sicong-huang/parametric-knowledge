#!/bin/bash
# Tier 2 placebo-fix re-run: corrupt_fact -> placebo select -> force_decode
# (main + placebo, per model) -> score_flips -> placebo score -> compare_arms
# -> position_recovery. select_targets.py already ran (candidates regenerated
# with the splitter fix + purity filter + position data).
set -euo pipefail
cd /home/mjsheu/parametric-knowledge

export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=EMPTY
export EXTRACTOR_MODEL=gemma-4-31b-it
export VLLM_USE_FLASHINFER_SAMPLER=0

echo "=== [1/8] corrupt_fact.py (main arm prefixes) ==="
uv run python premise2/scripts/corrupt_fact.py

echo "=== [2/8] placebo.py --select (placebo arm prefixes, position-matched) ==="
uv run python premise2/scripts/placebo.py --select

declare -A MODEL_EXP=(
  ["google/gemma-4-E4B-it"]=exp10
  ["nvidia/NVIDIA-Nemotron-Nano-9B-v2"]=exp14
  ["nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"]=exp16
  ["google/gemma-4-E2B-it"]=exp18
)

for model in "${!MODEL_EXP[@]}"; do
  exp="${MODEL_EXP[$model]}"
  echo "=== [3/8] force_decode.py --model $model --exp $exp ==="
  uv run python premise2/scripts/force_decode.py --model "$model" --exp "$exp"
  echo "=== [4/8] placebo.py --force-decode --model $model --exp $exp ==="
  uv run python premise2/scripts/placebo.py --force-decode --model "$model" --exp "$exp"
done

echo "=== [5/8] score_flips.py (main arm) ==="
uv run python premise2/scripts/score_flips.py

echo "=== [6/8] placebo.py --score (placebo arm) ==="
uv run python premise2/scripts/placebo.py --score

echo "=== [7/8] compare_arms.py (DiD + no-op instability) ==="
uv run python premise2/scripts/compare_arms.py

echo "=== [8/8] position_recovery.py ==="
uv run python premise2/scripts/position_recovery.py

echo "=== DONE ==="

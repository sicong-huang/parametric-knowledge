#!/usr/bin/env bash
#SBATCH --partition=gpuA100x4
#SBATCH --account=bful-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --job-name=build-mamba-ssm
#SBATCH --output=build_mamba_%j.log

set -euo pipefail

# A100 (8.0), A40 (8.6), H200 (9.0) — matches the GPUs these kernels need to run on.
export TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0"
export MAX_JOBS="${SLURM_CPUS_PER_TASK}"

echo current directory: $(pwd)

uv sync

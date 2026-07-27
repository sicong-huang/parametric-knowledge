# Working commands

## Interactive GPU session

```bash
salloc -p gpuA100x4-interactive -A bful-delta-gpu -t 1:00:00 -c 24 --gres=gpu:4 --mem=128G
srun -p gpuA100x4-interactive -A bful-delta-gpu -t 1:00:00 -c 24 --gres=gpu:4 --mem=128G --pty bash
srun -p gpuA100x4-interactive -A bful-delta-gpu -t 1:00:00 -c 12 --gres=gpu:1 --mem=64G --pty bash
srun -p gpuA40x4-interactive -A bful-delta-gpu -t 30:00 -c 8 --gres=gpu:1 --mem=32G --pty bash
srun -p gpuA40x4-interactive -A bful-delta-gpu -t 1:00:00 -c 12 --gres=gpu:2 --mem=64G --pty bash
srun -p gpuA40x4-interactive -A bful-delta-gpu -t 1:00:00 -c 32 --gres=gpu:4 --mem=256G --pty bash
srun -p gpuH200x8-interactive -A bful-delta-gpu -t 1:00:00 -c 16 --gres=gpu:2 --mem=128G --pty bash
```

## Make venv local to compute node
```bash
export UV_PROJECT_ENVIRONMENT=/tmp/venv
uv sync --frozen
```

## Quick test in interactive allocation

Inside `salloc` (or attach with `srun --jobid <JOBID> ...`). Swap model/args as needed.

```bash
srun uv run accelerate launch --num_processes=4 --mixed_precision=bf16 --gradient_accumulation_steps=32 \
    train_synthetic/pretrain.py \
    --model nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 \
    --data data/bio/biographies_uneven.jsonl \
    --epochs 0.1 \
    --batch-size 1 \
    --grad-accum 32 \
    --lr 3e-4 \
    --ds-stage 1 \
    --output-dir checkpoints/test-nemo-4b \
    --save-steps 2000 \
    --save-strategy steps \
    --note "test nemotron 4b on uneven biographies"
```

## qwen3.5-0.8b

### pretrain

```bash
args=(
    --partition gpuA100x4
    --account bful-delta-gpu
    --gpus 4
    --mem 128G
    --cpus 24
    --time 8:00:00
    --job-name pretrain-qwen3.5-0.8b

    --model Qwen/Qwen3.5-0.8B-Base
    --data data/bio/biographies_uneven.jsonl
    --epochs 3
    --batch-size 8
    --grad-accum 4
    --lr 3e-4
    --ds-stage 0
    --output-dir checkpoints/pretrain-qwen3.5-0.8b
    --save-steps 2000
    --save-strategy steps
    --note "pretrain qwen3.5-0.8b on uneven biographies"
)
scripts/pretrain.sbatch "${args[@]}"
```

interactive test:
```bash
uv run torchrun --nproc-per-node=2 \
    train_synthetic/pretrain.py \
    --model Qwen/Qwen3.5-0.8B-Base \
    --data data/bio/biographies_uneven.jsonl \
    --epochs 0.1 \
    --batch-size 4 \
    --grad-accum 8 \
    --lr 3e-4 \
    --ds-stage 0 \
    --output-dir checkpoints/test-qwen3.5-0.8b \
    --save-steps 2000 \
    --save-strategy steps \
    --note "test qwen3.5-0.8b on uneven biographies"
```

### sft

```bash
args=(
    --partition gpuA100x4-interactive
    --account bful-delta-gpu
    --gpus 2
    --mem 64G
    --cpus 12
    --time 1:00:00
    --job-name sft-qwen3.5-0.8b

    --model checkpoints/pretrain-qwen3.5-0.8b
    --qa-dir data/qa/train
    --epochs 2
    --batch-size 8
    --grad-accum 2
    --lr 2e-5
    --ds-stage 0
    --output-dir checkpoints/sft-qwen3.5-0.8b
    --note "sft qwen3.5-0.8b on mixed direct+reasoning QA"
)
scripts/sft.sbatch "${args[@]}"
```

interactive test:
```bash
uv run torchrun --nproc-per-node=2 \
    train_synthetic/sft.py \
    --model checkpoints/pretrain-qwen3.5-0.8b \
    --qa-dir data/qa/train \
    --epochs 2 \
    --batch-size 8 \
    --grad-accum 2 \
    --lr 2e-5 \
    --ds-stage 0 \
    --output-dir checkpoints/test-sft-qwen3.5-0.8b \
    --note "sft qwen3.5-0.8b on mixed direct+reasoning QA"
```

## gemma4-e2b

### pretrain

```bash
args=(
    --partition gpuH200x8
    --account bful-delta-gpu
    --gpus 4
    --mem 192G
    --cpus 24
    --time 6:00:00
    --job-name pt-gemma4-e2b

    --model google/gemma-4-E2B
    --data data/bio/biographies_uneven.jsonl
    --epochs 3
    --batch-size 4
    --grad-accum 8
    --lr 3e-4
    --ds-stage 2
    --output-dir checkpoints/pretrain-gemma4-e2b
    --save-steps 3000
    --save-strategy steps
    --note "pretrain gemma4-e2b on uneven biographies"
)
scripts/pretrain.sbatch "${args[@]}"
```

On A100 instead of H200: add `--gradient-checkpointing`, `--ds-stage 3`, `--batch-size 1 --grad-accum 32`.

Interactive test:
```bash
srun uv run torchrun --nproc-per-node=4 \
    train_synthetic/pretrain.py \
    --model google/gemma-4-E2B \
    --data data/bio/biographies_uneven.jsonl \
    --epochs 0.1 \
    --batch-size 1 \
    --grad-accum 32 \
    --lr 3e-4 \
    --ds-stage 2 \
    --output-dir checkpoints/test-gemma4-e2b \
    --save-steps 2000 \
    --save-strategy steps \
    --note "test gemma4-e2b on uneven biographies"
```

### sft

```bash
args=(
    --partition gpuH200x8
    --account bful-delta-gpu
    --gpus 2
    --mem 96G
    --cpus 12
    --time 1:00:00
    --job-name sft-gemma4-e2b

    --model checkpoints/pretrain-gemma4-e2b
    --qa-dir data/qa/train
    --chat-template google/gemma-4-E2B-it
    --epochs 2
    --batch-size 16
    --grad-accum 1
    --lr 2e-5
    --ds-stage 0
    --output-dir checkpoints/sft-gemma4-e2b
    --note "sft gemma4-e2b on mixed direct+reasoning QA"
)
scripts/sft.sbatch "${args[@]}"
```
interactive test:
```bash
uv run torchrun --nproc-per-node=2 \
    train_synthetic/sft.py \
    --model checkpoints/pretrain-gemma4-e2b \
    --qa-dir data/qa/train \
    --chat-template google/gemma-4-E2B-it \
    --epochs 0.1 \
    --batch-size 16 \
    --grad-accum 1 \
    --lr 2e-5 \
    --ds-stage 0 \
    --output-dir checkpoints/test-sft-gemma4-e2b \
    --note "sft gemma4-e2b on mixed direct+reasoning QA"
```

## nemotron 3 nano 4b

### pretrain

```bash
args=(
    --partition gpuA100x4
    --account bful-delta-gpu
    --gpus 4
    --mem 128G
    --cpus 24
    --time 16:00:00
    --job-name pt-nemo-4b

    --model nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16
    --data data/bio/biographies_uneven.jsonl
    --epochs 3
    --batch-size 1
    --grad-accum 32
    --lr 3e-4
    --ds-stage 1
    --output-dir checkpoints/pretrain-nemotron3nano-4b
    --save-steps 2000
    --save-strategy steps
    --note "pretrain nemotron 3 nano 4b on uneven biographies"
)
scripts/pretrain.sbatch "${args[@]}"
```

### sft

```bash
args=(
    --partition gpuA40x4
    --account bful-delta-gpu
    --gpus 4
    --mem 128G
    --cpus 24
    --time 4:00:00
    --job-name sft-nemotron3nano-4b

    --model checkpoints/pretrain-nemotron3nano-4b
    --qa-dir data/qa/train
    --epochs 2
    --batch-size 2
    --grad-accum 4
    --lr 2e-5
    --ds-stage 1
    --output-dir checkpoints/sft-nemotron3nano-4b
    --note "sft nemotron 3 nano 4b on mixed direct+reasoning QA"
)
scripts/sft.sbatch "${args[@]}"
```

interactive test:
```bash
uv run torchrun --nproc-per-node=4 \
    train_synthetic/sft.py \
    --model checkpoints/pretrain-nemotron3nano-4b \
    --qa-dir data/qa/train \
    --epochs 0.1 \
    --batch-size 1 \
    --grad-accum 8 \
    --lr 2e-5 \
    --ds-stage 1 \
    --output-dir checkpoints/test2-sft-nemotron3nano-4b \
    --note "sft nemotron 3 nano 4b on mixed direct+reasoning QA"
```

## nemotron 2 nano 9b

### pretrain (test only so far)

```bash
args=(
    --partition gpuA100x4-interactive
    --account bful-delta-gpu
    --gpus 4
    --mem 128G
    --cpus 24
    --time 00:30:00
    --job-name default-test
    --srun

    --model nvidia/NVIDIA-Nemotron-Nano-9B-v2
    --data data/bio/biographies_uneven.jsonl
    --epochs 0.1
    --batch-size 1
    --grad-accum 32
    --gradient-checkpointing
    --lr 3e-4
    --ds-stage 3
    --output-dir checkpoints/default-test
    --save-steps 1000
    --save-strategy steps
    --note "quick srun test on uneven biographies"
)
scripts/pretrain.sbatch "${args[@]}"
```

## Quick test in interactive allocation

Inside `salloc` (or attach with `srun --jobid <JOBID> ...`). Swap model/args as needed.

```bash
srun uv run accelerate launch --num_processes=4 --mixed_precision=bf16 --gradient_accumulation_steps=32 \
    train_synthetic/pretrain.py \
    --model nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 \
    --data data/bio/biographies_uneven.jsonl \
    --epochs 0.1 \
    --batch-size 1 \
    --grad-accum 32 \
    --lr 3e-4 \
    --ds-stage 1 \
    --output-dir checkpoints/test-nemo-4b \
    --save-steps 2000 \
    --save-strategy steps \
    --note "test nemotron 4b on uneven biographies"
```


# Evaluation

```bash
uv run train_synthetic/eval.py \
    --model checkpoints/sft-qwen3.5-0.8b \
    --qa-dir data/qa/test \
    --output experiment/sft-qwen3.5-0.8b \
    --max-new-tokens 256
```

```bash
uv run train_synthetic/eval.py \
    --model checkpoints/sft-gemma4-e2b \
    --qa-dir data/qa/test \
    --output experiment/sft-gemma4-e2b \
    --max-new-tokens 256 \
    --no-vllm
```

```bash
uv run train_synthetic/eval.py \
    --model checkpoints/sft-nemotron3nano-4b \
    --qa-dir data/qa/test \
    --output experiment/sft-nemotron3nano-4b \
    --max-new-tokens 256
```

## gemma4-e2b

```bash
args=(
    --partition gpuA100x4
    --account bful-delta-gpu
    --gpus 1
    --mem 64G
    --cpus 12
    --time 2:00:00
    --job-name eval-gemma4-e2b
    --no-vllm

    --model checkpoints/sft-gemma4-e2b
    --qa-dir data/qa/test
    --output experiment/sft-gemma4-e2b
    --max-new-tokens 256
)
scripts/eval.sbatch "${args[@]}"
```

## nemotron 3 nano 4b
```bash
args=(
    --partition gpuA100x4
    --account bful-delta-gpu
    --gpus 1
    --mem 64G
    --cpus 12
    --time 2:00:00
    --job-name eval-nemotron3nano-4b

    --model checkpoints/sft-nemotron3nano-4b
    --qa-dir data/qa/test
    --output experiment/sft-nemotron3nano-4b
    --max-new-tokens 256
)
scripts/eval.sbatch "${args[@]}"
```

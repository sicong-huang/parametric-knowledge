# Commands used to train and evaluate the checkpoints

Training arguments come from each checkpoint's `run_config.json`. Slurm resources (partition, account,
GPU count, memory, time) are not recorded there, so they are copied from the original launch notes
and are not verified against the logs. Run everything from the repo root.

Pretrain and SFT use `torchrun --nproc-per-node=<GPUS> train_synthetic/<script>.py ...`.
Evaluation uses `train_synthetic/eval.py`.

## qwen3.5-0.8b

### Pretrain  (checkpoints/pretrain-qwen3.5-0.8b)
```bash
torchrun --nproc-per-node=4 train_synthetic/pretrain.py \
    --model Qwen/Qwen3.5-0.8B-Base \
    --data data/bio/biographies_uneven.jsonl \
    --max-seq-length 2048 \
    --epochs 3 --batch-size 8 --grad-accum 4 --lr 3e-4 \
    --backend deepspeed --ds-stage 0 \
    --save-steps 1000 --save-strategy steps \
    --output-dir checkpoints/pretrain-qwen3.5-0.8b \
    --note "pretrain qwen3.5-0.8b on uneven biographies"
```

### SFT  (checkpoints/sft-qwen3.5-0.8b)
```bash
torchrun --nproc-per-node=2 train_synthetic/sft.py \
    --model checkpoints/pretrain-qwen3.5-0.8b \
    --qa-dir data/qa/train \
    --epochs 2 --batch-size 8 --grad-accum 2 --lr 2e-5 \
    --ds-stage 0 --seed 100 \
    --output-dir checkpoints/sft-qwen3.5-0.8b \
    --note "sft qwen3.5-0.8b on mixed direct+reasoning QA"
```

### Eval
```bash
python train_synthetic/eval.py \
    --model checkpoints/sft-qwen3.5-0.8b \
    --qa-dir data/qa/test \
    --output experiment/sft-qwen3.5-0.8b \
    --max-new-tokens 256
```

## gemma4-e2b

### Pretrain  (checkpoints/pretrain-gemma4-e2b)
```bash
torchrun --nproc-per-node=4 train_synthetic/pretrain.py \
    --model google/gemma-4-E2B \
    --data data/bio/biographies_uneven.jsonl \
    --max-seq-length 2048 \
    --epochs 3 --batch-size 4 --grad-accum 8 --lr 3e-4 \
    --backend deepspeed --ds-stage 2 \
    --save-steps 2000 --save-strategy steps \
    --output-dir checkpoints/pretrain-gemma4-e2b \
    --note "pretrain gemma4-e2b on uneven biographies"
```
(ran on H200s; on A100s add `--gradient-checkpointing --ds-stage 3 --batch-size 1 --grad-accum 32`)

### SFT  (checkpoints/sft-gemma4-e2b)
```bash
torchrun --nproc-per-node=2 train_synthetic/sft.py \
    --model checkpoints/pretrain-gemma4-e2b \
    --qa-dir data/qa/train \
    --chat-template google/gemma-4-E2B-it \
    --epochs 2 --batch-size 16 --grad-accum 1 --lr 2e-5 \
    --ds-stage 0 --seed 100 \
    --output-dir checkpoints/sft-gemma4-e2b \
    --note "sft gemma4-e2b on mixed direct+reasoning QA"
```

### Eval  (--no-vllm)
```bash
python train_synthetic/eval.py \
    --model checkpoints/sft-gemma4-e2b \
    --qa-dir data/qa/test \
    --output experiment/sft-gemma4-e2b \
    --max-new-tokens 256 --no-vllm
```

## nemotron3nano-4b

### Pretrain  (checkpoints/pretrain-nemotron3nano-4b)
```bash
torchrun --nproc-per-node=4 train_synthetic/pretrain.py \
    --model nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 \
    --data data/bio/biographies_uneven.jsonl \
    --max-seq-length 2048 \
    --epochs 3 --batch-size 1 --grad-accum 32 --lr 3e-4 \
    --ds-stage 1 \
    --save-steps 2000 --save-strategy steps \
    --output-dir checkpoints/pretrain-nemotron3nano-4b \
    --note "pretrain nemotron 3 nano 4b on uneven biographies"
```

### SFT  (checkpoints/sft-nemotron3nano-4b)
```bash
torchrun --nproc-per-node=4 train_synthetic/sft.py \
    --model checkpoints/pretrain-nemotron3nano-4b \
    --qa-dir data/qa/train \
    --epochs 2 --batch-size 2 --grad-accum 4 --lr 2e-5 \
    --ds-stage 1 --seed 100 \
    --output-dir checkpoints/sft-nemotron3nano-4b \
    --note "sft nemotron 3 nano 4b on mixed direct+reasoning QA"
```

### Eval
```bash
python train_synthetic/eval.py \
    --model checkpoints/sft-nemotron3nano-4b \
    --qa-dir data/qa/test \
    --output experiment/sft-nemotron3nano-4b \
    --max-new-tokens 256
```

#!/usr/bin/env python3
"""Continued pretraining on synthetic biography data.

Usage:
    uv run train_synthetic/pretrain.py
    uv run train_synthetic/pretrain.py --data data/bio/biographies_uneven.jsonl
    uv run train_synthetic/pretrain.py --epochs 5 --lr 3e-5
"""

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from datasets import Dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def load_biographies(*jsonl_paths: str) -> Dataset:
    """Load biography texts from one or more biographies.jsonl files.

    Extracts only the 'biography' field and wraps it in a 'text' field
    suitable for causal LM continued pretraining.
    """
    texts = []
    for path in jsonl_paths:
        with open(path) as f:
            for line in f:
                record = json.loads(line)
                texts.append({"text": record["biography"]})
    return Dataset.from_list(texts)


def dataset_cache_path(*data_paths: str) -> Path:
    """Derive a cache path from input file paths and their modification times.

    The cache is invalidated automatically when any source file is modified.
    """
    key = "\n".join(f"{p}:{os.path.getmtime(p)}" for p in sorted(data_paths))
    digest = hashlib.md5(key.encode()).hexdigest()[:12]
    return Path("data/processed") / digest


def main():
    parser = argparse.ArgumentParser(description="Continued pretraining on biography data")
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M")
    parser.add_argument("--output-dir", default="checkpoints/exp-0")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--data", nargs="+", default=["data/bio/biographies.jsonl"])
    parser.add_argument("--epochs", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=False)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ds-stage", type=int, default=2, choices=[0, 1, 2, 3])
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--save-strategy", default="steps", choices=["no", "steps", "epoch"])
    parser.add_argument("--note", type=str, default="")
    args = parser.parse_args()

    print(f"DeepSpeed ZeRO stage: {args.ds_stage}")

    ds_config = {
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": args.ds_stage,
            "overlap_comm": args.ds_stage >= 1,
            "contiguous_gradients": args.ds_stage >= 1,
            "reduce_scatter": args.ds_stage >= 2,
        },
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "wall_clock_breakdown": False,
    }

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_length=args.max_seq_length,
        dataset_text_field="text",
        dataset_num_proc=8,
        optim="adamw_torch_fused",
        warmup_steps=0.05,
        weight_decay=0.01,
        lr_scheduler_type="constant_with_warmup",
        seed=100,
        bf16=True,
        gradient_checkpointing=args.gradient_checkpointing,
        logging_steps=50,
        save_steps=args.save_steps,
        save_strategy=args.save_strategy,
        report_to="none",
        deepspeed=ds_config,
    )

    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,} (all trainable)")

    if training_args.process_index == 0:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        run_config = {
            "args": vars(args),
            "training_arguments": {
                k: v
                for k, v in dataclasses.asdict(training_args).items()
                if isinstance(v, (bool, int, float, str, list, type(None)))
            },
        }
        config_path = output_dir / "run_config.json"
        config_path.write_text(json.dumps(run_config, indent=2))
        print(f"Run config saved to {config_path}")

    dataset_cache = dataset_cache_path(*args.data)
    with training_args.main_process_first(desc="loading data"):
        if not dataset_cache.exists():
            print(f"Loading data from: {args.data}")
            dataset = load_biographies(*args.data)
            print(f"Dataset size: {len(dataset)} biographies — saving to {dataset_cache}")
            dataset_cache.parent.mkdir(parents=True, exist_ok=True)
            dataset.save_to_disk(str(dataset_cache))
        else:
            print(f"Using cached dataset at {dataset_cache}")
    dataset = load_from_disk(str(dataset_cache))

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    trainer.train()

    print(f"Saving to {args.output_dir}")
    trainer.save_model(args.output_dir)
    print("Done.")

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

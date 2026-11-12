#!/usr/bin/env python3
"""SFT on synthetic QA (1/2-hop × direct/reasoning) with completion-only loss.

Usage:
    uv run train_synthetic/sft.py --model checkpoints/pretrain-qwen3.5-0.8b
    uv run train_synthetic/sft.py --model checkpoints/pretrain-qwen3.5-0.8b \
        --output-dir checkpoints/sft-0 --epochs 2
"""

import argparse
import dataclasses
import hashlib
import json
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from accelerate import PartialState
from datasets import Dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from trl.chat_template_utils import clone_chat_template, get_training_chat_template, has_generation_markers

HOPS = ("1_hop", "2_hop")
STYLES = ("direct", "reasoning")
CACHE_READY = ".ready"


def cache_ready(cache_dir: Path) -> bool:
    return (cache_dir / CACHE_READY).is_file()


def load_qa_sft(qa_dir: Path, seed: int) -> Dataset:
    """Concatenate all QA train splits into shuffled prompt/completion examples."""
    examples = []
    for n_hop in HOPS:
        for style in STYLES:
            src = qa_dir / f"qa_{n_hop}_{style}_train.jsonl"
            if not src.is_file():
                raise SystemExit(f"Missing QA file: {src}")
            with open(src) as f:
                for line in f:
                    record = json.loads(line)
                    examples.append(
                        {
                            "prompt": [{"role": "user", "content": record["question"]}],
                            "completion": [{"role": "assistant", "content": record["full_answer"]}],
                            # Match generation prompt to completion (thinking vs empty <think>).
                            "chat_template_kwargs": {"enable_thinking": style == "reasoning"},
                        }
                    )
    return Dataset.from_list(examples).shuffle(seed=seed)


def qa_source_paths(qa_dir: Path) -> list[str]:
    return [str(qa_dir / f"qa_{n_hop}_{style}_train.jsonl") for n_hop in HOPS for style in STYLES]


def sft_dataset_cache_path(qa_dir: Path, seed: int, model_path: str, chat_template, training_args) -> Path:
    """Derive a cache path from QA files, tokenizer settings, and SFT prep options."""
    key = "\n".join(f"{p}:{os.path.getmtime(p)}" for p in sorted(qa_source_paths(qa_dir)))
    key += (
        f"\nseed:{seed}\nmodel:{model_path}\nchat_template:{chat_template}"
        f"\nmax_length:{training_args.max_length}\ncompletion_only:{training_args.completion_only_loss}"
    )
    digest = hashlib.md5(key.encode()).hexdigest()[:12]
    return Path("data/processed") / f"qa-sft-{digest}"


def apply_chat_template(model, tokenizer, chat_template_path: str | None):
    if chat_template_path is None:
        return model, tokenizer
    if os.path.isfile(chat_template_path) and chat_template_path.endswith((".jinja", ".j2")):
        with open(chat_template_path, encoding="utf-8") as chat_template_file:
            tokenizer.chat_template = chat_template_file.read()
        return model, tokenizer
    model, tokenizer, _ = clone_chat_template(model, tokenizer, chat_template_path)
    return model, tokenizer


def prepare_sft_dataset(dataset: Dataset, processing_class, args: SFTConfig) -> Dataset:
    """Run SFTTrainer's dataset prep once so tokenized data can be cached to disk."""
    if args.assistant_only_loss and not has_generation_markers(processing_class.chat_template):
        resolved_chat_template = get_training_chat_template(processing_class)
    else:
        resolved_chat_template = None

    if args.completion_only_loss is None:
        first = dataset[0]
        use_completion_only_loss = "prompt" in first and "completion" in first
    else:
        use_completion_only_loss = args.completion_only_loss

    tokenizer = getattr(processing_class, "tokenizer", processing_class)
    prep = SimpleNamespace(
        _tokenizer=tokenizer,
        chat_template=resolved_chat_template,
        completion_only_loss=use_completion_only_loss,
    )
    # Rank 0 builds alone while other ranks wait on an outer barrier; skip inner sync.
    orig_main_process_first = PartialState.main_process_first

    @contextmanager
    def _no_distributed_first(_self):
        yield

    PartialState.main_process_first = _no_distributed_first
    try:
        return SFTTrainer._prepare_dataset(prep, dataset, processing_class, args, args.packing, None, "train")
    finally:
        PartialState.main_process_first = orig_main_process_first


def stage_model_local(model_path: str, training_args) -> str:
    """Copy a local checkpoint to $TMPDIR once per node (avoids Lustre mmap stalls)."""
    src = Path(model_path)
    if not src.is_dir():
        return model_path
    dst = Path(os.environ.get("TMPDIR", "/tmp")) / "staged-models" / src.resolve().name
    ignore = shutil.ignore_patterns("checkpoint-*", "*.log", "training_args.bin")
    wanted = [f for f in src.iterdir() if f.is_file() and f.name not in ignore(str(src), os.listdir(src))]
    staged = all(
        (dst / f.name).is_file() and (dst / f.name).stat().st_size == f.stat().st_size for f in wanted
    )
    with training_args.main_process_first(local=True, desc="staging model to node-local disk"):
        if training_args.local_process_index == 0 and not staged:
            print(f"Staging {src} -> {dst}")
            shutil.copytree(src, dst, dirs_exist_ok=True, ignore=ignore)
    return str(dst)


def main():
    parser = argparse.ArgumentParser(description="SFT on synthetic QA data")
    parser.add_argument("--model", default="checkpoints/exp-0")
    parser.add_argument("--output-dir", default="checkpoints/sft-0")
    parser.add_argument("--qa-dir", type=Path, default=Path("data/qa/train"))
    parser.add_argument(
        "--chat-template",
        default=None,
        help="HF id, local dir, or .jinja when --model's tokenizer has no template",
    )
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=False)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--ds-stage", type=int, default=2, choices=[0, 1, 2, 3])
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--note", type=str, default="")
    args = parser.parse_args()

    print(f"DeepSpeed ZeRO stage: {args.ds_stage}")
    s = args.ds_stage
    ds_config = {
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": s,
            "overlap_comm": s >= 1,
            "contiguous_gradients": s >= 1,
            "reduce_scatter": s >= 2,
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
        completion_only_loss=True,
        chat_template_path=args.chat_template,
        dataset_num_proc=8,
        optim="adamw_torch_fused",
        warmup_steps=0.05,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=args.seed,
        bf16=True,
        gradient_checkpointing=args.gradient_checkpointing,
        logging_steps=50,
        save_strategy="no",
        report_to="none",
        deepspeed=ds_config,
    )

    model_path = stage_model_local(args.model, training_args)
    print(f"Loading model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model, tokenizer = apply_chat_template(model, tokenizer, args.chat_template)
    training_args.chat_template_path = None
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,} (all trainable)")

    if training_args.process_index == 0:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        config_path = out / "run_config.json"
        config_path.write_text(
            json.dumps(
                {
                    "args": {**vars(args), "qa_dir": str(args.qa_dir)},
                    "training_arguments": {
                        k: v
                        for k, v in dataclasses.asdict(training_args).items()
                        if isinstance(v, (bool, int, float, str, list, type(None)))
                    },
                },
                indent=2,
            )
        )
        print(f"Run config saved to {config_path}")

    dataset_cache = sft_dataset_cache_path(args.qa_dir, args.seed, model_path, args.chat_template, training_args)
    with training_args.main_process_first(desc="loading data"):
        if training_args.process_index == 0:
            if cache_ready(dataset_cache):
                print(f"Using cached dataset at {dataset_cache}")
            else:
                if dataset_cache.exists():
                    shutil.rmtree(dataset_cache)
                print(f"Loading QA data from {args.qa_dir}")
                raw = load_qa_sft(args.qa_dir, args.seed)
                print(f"Dataset size: {len(raw)} examples — tokenizing and saving to {dataset_cache}")
                dataset_cache.parent.mkdir(parents=True, exist_ok=True)
                prepared = prepare_sft_dataset(raw, tokenizer, training_args)
                prepared.save_to_disk(str(dataset_cache))
                (dataset_cache / CACHE_READY).write_text("ok")
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    if not cache_ready(dataset_cache):
        raise RuntimeError(f"Tokenized dataset cache not ready: {dataset_cache}")
    dataset = load_from_disk(str(dataset_cache))
    print(f"Dataset size: {len(dataset)} examples")

    training_args.dataset_kwargs = {"skip_prepare_dataset": True}
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )
    trainer.train()
    print(f"Saving to {args.output_dir}")
    trainer.save_model(args.output_dir)
    # Base model may be multimodal (e.g. gemma4); vLLM needs processor_config.json
    # even for text-only inference, and save_model only writes the tokenizer.
    try:
        AutoProcessor.from_pretrained(model_path).save_pretrained(args.output_dir)
    except Exception as e:
        print(f"Warning: could not save processor config: {e}")
    print("Done.")

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

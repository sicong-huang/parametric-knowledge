#!/usr/bin/env python3
"""Make an HF/TRL Qwen3.5 text-only checkpoint loadable by prime-rl's trainer.

transformers saves `Qwen3_5ForCausalLM` (what train_synthetic/pretrain.py writes) with the
VL-style key prefix `model.language_model.*`, but prime-rl's text-only Qwen3.5 model loads
`model.*` and fails with "Missing key in checkpoint state_dict: model.embed_tokens.weight".
This writes a copy with the prefix stripped (and any `mtp.*` heads dropped, which prime-rl
drops anyway). Config, tokenizer and chat template are copied unchanged. The weights are
the same; only the tensor names change.

Tied embeddings: stock prime-rl's Qwen3.5 has no weight tying (it would train from an
UNINITIALIZED lm_head, loss ~= ln(vocab) = 12.4, and crash on export). Our local prime-rl
patch (patches/prime-rl-local.patch) adds tying, so tied checkpoints are
kept tied by default, matching TRL. --untie instead writes lm_head = a copy of the embeddings
and sets tie_word_embeddings=false (only needed on an unpatched prime-rl).

    uv run --no-sync --project ~/prime-rl python train_scripts/train_prime/prepare_model.py \
        pretrain-qwen3.5-0.8b checkpoints/pretrain-qwen3.5-0.8b-prime
"""

import argparse
import json
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

OLD, NEW = "model.language_model.", "model."
SKIP = {"training_args.bin"}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src", type=Path)
    p.add_argument("dst", type=Path)
    p.add_argument("--untie", action="store_true",
                   help="untie lm_head from embeddings; only for an unpatched prime-rl (see docstring)")
    args = p.parse_args()

    if not (args.src / "config.json").is_file():
        raise SystemExit(f"{args.src}/config.json missing; is the checkpoint fully copied?")
    shards = sorted(args.src.glob("*.safetensors"))
    if not shards:
        raise SystemExit(f"no *.safetensors in {args.src}")
    args.dst.mkdir(parents=True, exist_ok=True)

    for f in args.src.iterdir():
        if f.is_file() and f.suffix != ".safetensors" and f.name not in SKIP and not f.name.endswith(".index.json"):
            shutil.copy2(f, args.dst / f.name)

    config = json.loads((args.src / "config.json").read_text())
    text_cfg = config.get("text_config", config)
    tied = config.get("tie_word_embeddings", text_cfg.get("tie_word_embeddings", False))
    if tied and not args.untie:
        print("tie_word_embeddings=true: kept tied (requires the patched prime-rl Qwen3.5, see docstring)")
    untie = tied and args.untie

    renamed = dropped = 0
    weight_map = {}
    for shard in shards:
        tensors = {}
        with safe_open(shard, "pt") as h:
            metadata = h.metadata() or {}
            for k in h.keys():
                if k.startswith("mtp."):
                    dropped += 1
                    continue
                new = NEW + k[len(OLD):] if k.startswith(OLD) else k
                renamed += new != k
                tensors[new] = h.get_tensor(k)
                weight_map[new] = shard.name
        if untie and "model.embed_tokens.weight" in tensors and not any("lm_head.weight" in w for w in weight_map):
            tensors["lm_head.weight"] = tensors["model.embed_tokens.weight"].clone()
            weight_map["lm_head.weight"] = shard.name
        save_file(tensors, args.dst / shard.name, metadata={**metadata, "format": "pt"})

    if untie:
        config["tie_word_embeddings"] = False
        if "text_config" in config:
            config["text_config"]["tie_word_embeddings"] = False
        (args.dst / "config.json").write_text(json.dumps(config, indent=2))
        print("untied: lm_head.weight = copy of embed_tokens, tie_word_embeddings=false")
    if len(shards) > 1:
        (args.dst / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {}, "weight_map": weight_map}, indent=2))
    print(f"{args.src} -> {args.dst}: {len(weight_map)} tensors, {renamed} renamed, {dropped} mtp dropped")


if __name__ == "__main__":
    main()

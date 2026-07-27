#!/usr/bin/env python3
"""Evaluate an SFT checkpoint on the QA test set in direct and reasoning modes.

For each style, all ``<qa-dir>/qa_{n_hop}_{style}_test.jsonl`` files are
evaluated with greedy decoding. Accuracy is lower-cased substring match of the
ground-truth answer in the model output; for reasoning, text inside
``<think>...</think>`` is excluded from matching.

Usage:
    uv run train_synthetic/eval.py --model checkpoints/sft-qwen3.5-0.8b \
        --output experiment/sft-qwen3.5-0.8b
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, ModelRegistry, SamplingParams
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.models.interfaces import IsHybrid, SupportsMRoPE
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLM as _VllmQwen3_5ForCausalLM
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper


class Qwen3_5ForCausalLMHF(_VllmQwen3_5ForCausalLM, IsHybrid, SupportsMRoPE):
    """vllm's registry only maps "Qwen3_5ForConditionalGeneration" (the VL
    variant); a text-only "Qwen3_5ForCausalLM" checkpoint falls back to it
    and crashes on the missing vision_config, so we register this class under
    that name explicitly. It also strips the "language_model." infix HF's
    checkpoint uses (e.g. "model.language_model.layers...") that vllm's
    unwrapped "model.layers..." doesn't expect.

    The base text-only class doesn't mix in `IsHybrid` (only the VL wrapper
    does), so vllm never reconciles the GDN (mamba) state page size against
    the full-attention page size for it, and KV cache init crashes with
    "page size of the layer is not divisible by the maximum page size". Mix
    it in here with the same state-shape/dtype/copy-func classmethods the VL
    class uses (they only read hf_text_config, nothing VL-specific).
    """

    hf_to_vllm_mapper = WeightsMapper(orig_to_new_substr={"language_model.": ""})

    def load_weights(self, weights):
        loader = AutoWeightsLoader(self, skip_prefixes=["mtp."])
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config):
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config):
        hf_config = vllm_config.model_config.hf_text_config
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            vllm_config.parallel_config.tensor_parallel_size,
            hf_config.linear_num_key_heads,
            hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim,
            num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(cls):
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()

    def get_mrope_input_positions(self, input_tokens, mm_features):
        """Text-only reduction of Qwen3VL's mrope position calc: with no
        mm_features the T/H/W position ids are all just the sequential text
        position, so this is equivalent to plain 1D rope broadcast 3x.
        """
        assert not mm_features, "text-only reduction; not valid for multimodal input"
        text_len = len(input_tokens)
        positions = np.broadcast_to(np.arange(text_len), (3, text_len))
        return torch.from_numpy(positions.copy()), 0


ModelRegistry.register_model("Qwen3_5ForCausalLM", Qwen3_5ForCausalLMHF)

HOPS = ("1_hop", "2_hop")
STYLES = ("direct", "reasoning")


def strip_think(text: str) -> str:
    """Return the answer: everything after the last ``</think>``.

    The chat template emits the opening ``<think>`` as part of the generation
    prompt, so a trace usually arrives untagged and can only be located by its
    terminator. An unterminated ``<think>`` leaves no answer to match against;
    output with neither tag is a plain answer.
    """
    _, sep, answer = text.rpartition("</think>")
    if sep:
        return answer
    return "" if "<think>" in text else text


def stage_model_local(model_path: str) -> str:
    """Copy a local checkpoint to node-local disk before loading.

    Single-process version of sft.py's stage_model_local: mmap'ing safetensors
    off Lustre is slow; one sequential copy is fast. Skips if already staged.
    """
    src = Path(model_path)
    if not src.is_dir():
        return model_path
    dst = Path(os.environ.get("TMPDIR", "/tmp")) / "staged-models" / src.resolve().name
    ignore = shutil.ignore_patterns("checkpoint-*", "*.log", "training_args.bin")
    wanted = [f for f in src.iterdir() if f.is_file() and f.name not in ignore(str(src), os.listdir(src))]
    if not all((dst / f.name).is_file() and (dst / f.name).stat().st_size == f.stat().st_size for f in wanted):
        print(f"Staging {src} -> {dst}")
        shutil.copytree(src, dst, dirs_exist_ok=True, ignore=ignore)
    return str(dst)


def load_examples(qa_dir: Path, style: str) -> list[dict]:
    examples = []
    for n_hop in HOPS:
        src = qa_dir / f"qa_{n_hop}_{style}_test.jsonl"
        if not src.is_file():
            raise SystemExit(f"Missing QA file: {src}")
        with open(src) as f:
            for line in f:
                record = json.loads(line)
                examples.append(
                    {
                        "qaid": record["qaid"],
                        "n_hop": n_hop,
                        "question": record["question"],
                        "answer": record["answer"],
                    }
                )
    return examples


def stop_token_ids(tokenizer) -> list[int]:
    """EOS plus the turn terminator the chat template appends after assistant.

    Works across ChatML/Llama-3/etc. without hardcoding token names: render a
    dummy assistant message and take leading added-vocab ids from the suffix
    (e.g. ``<|im_end|>`` after the answer). ``all_special_ids`` is not enough —
    Qwen leaves ``<|im_end|>`` out of it.
    """
    ids: set[int] = set()
    eos = tokenizer.eos_token_id
    if isinstance(eos, int):
        ids.add(eos)
    elif eos:
        ids.update(int(x) for x in eos)

    marker = "<<<END>>>"
    try:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": "?"}, {"role": "assistant", "content": marker}],
            tokenize=False,
        )
    except Exception:
        return sorted(ids)
    if marker not in text:
        return sorted(ids)

    added = set(tokenizer.get_added_vocab().values())
    for tid in tokenizer.encode(text.split(marker, 1)[1], add_special_tokens=False):
        if tid in added or tid in ids:
            ids.add(tid)
        else:
            break
    return sorted(ids)


def generate(llm, tokenizer, examples, style, max_new_tokens, stop_ids):
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": ex["question"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=style == "reasoning",
        )
        for ex in examples
    ]
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=1.0,
        top_p=0.6,
        top_k=60,
        stop_token_ids=stop_ids,
    )
    results = llm.generate(prompts, sampling_params)
    return [r.outputs[0].text for r in results]


def generate_hf(model, tokenizer, examples, style, max_new_tokens, stop_ids, batch_size=8):
    """HF `model.generate` equivalent of `generate`, for --no-vllm."""
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": ex["question"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=style == "reasoning",
        )
        for ex in examples
    ]
    outputs = []
    for i in tqdm(range(0, len(prompts), batch_size), desc=style, unit="batch"):
        batch = prompts[i : i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=1.0,
                top_p=0.6,
                top_k=60,
                eos_token_id=stop_ids,
                pad_token_id=tokenizer.pad_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        outputs.extend(tokenizer.batch_decode(out[:, prompt_len:], skip_special_tokens=True))
    return outputs


def main():
    parser = argparse.ArgumentParser(description="Evaluate SFT checkpoint on QA test set")
    parser.add_argument("--model", required=True)
    parser.add_argument("--qa-dir", type=Path, default=Path("data/qa/test"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--no-vllm", action="store_true", help="Use HF `model.generate` instead of vLLM"
    )
    args = parser.parse_args()

    model_path = stage_model_local(args.model)
    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    stop_ids = stop_token_ids(tokenizer)
    print(f"stop_token_ids: {stop_ids} {[tokenizer.decode([i]) for i in stop_ids]}")

    if args.no_vllm:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, attn_implementation="sdpa"
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
    else:
        llm = LLM(model=model_path, dtype="bfloat16")

    args.output.mkdir(parents=True, exist_ok=True)
    summary = {"model": args.model, "qa_dir": str(args.qa_dir), "stop_token_ids": stop_ids}

    for style in STYLES:
        examples = load_examples(args.qa_dir, style)
        if args.no_vllm:
            outputs = generate_hf(model, tokenizer, examples, style, args.max_new_tokens, stop_ids)
        else:
            outputs = generate(llm, tokenizer, examples, style, args.max_new_tokens, stop_ids)
        for ex, out in zip(examples, outputs):
            matchable = strip_think(out) if style == "reasoning" else out
            if style == "reasoning" and "<think>" not in out:
                # the generation prompt supplied the opening tag, so put it back
                out = "<think>\n" + out
            ex["output"] = out
            ex["correct"] = ex["answer"].lower() in matchable.lower()

        with open(args.output / f"{style}.jsonl", "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")

        stats = {"accuracy": sum(ex["correct"] for ex in examples) / len(examples), "n": len(examples)}
        for n_hop in HOPS:
            subset = [ex for ex in examples if ex["n_hop"] == n_hop]
            stats[n_hop] = {"accuracy": sum(ex["correct"] for ex in subset) / len(subset), "n": len(subset)}
        summary[style] = stats
        print(f"[{style}] accuracy: {stats['accuracy']:.4f} ({stats['n']} examples)")

    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()


def test_strip_think():
    assert strip_think("reasoning about X\n</think>\n\nY") == "\n\nY"
    assert strip_think("<think>the answer is X</think>The answer is Y.") == "The answer is Y."
    assert strip_think("<think>unterminated reasoning about X") == ""
    assert strip_think("no tags at all") == "no tags at all"
    assert strip_think("first</think>second</think>third") == "third"


def test_stop_token_ids():
    tok = AutoTokenizer.from_pretrained("checkpoints/sft-qwen3.5-0.8b")
    ids = stop_token_ids(tok)
    assert tok.eos_token_id in ids
    assert tok.convert_tokens_to_ids("<|im_end|>") in ids

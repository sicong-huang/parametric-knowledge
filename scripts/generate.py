"""Generate answers for ONE dataset with vLLM -> experiment/<exp>/outputs/<dataset>.jsonl.

Usage:
    uv run python scripts/generate.py --dataset nq --exp exp1

If running this directly (not via scripts/run_exp.sh), prefix with
VLLM_USE_FLASHINFER_SAMPLER=0 -- see the comment in run_exp.sh for why.
"""
import argparse
import json
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from datasets_registry import DATASETS
from evaluation.parse import extract_answer

REPO_ROOT = pathlib.Path(__file__).parent.parent

SYSTEM_PROMPT = (
    "Answer the question. Think step by step if needed, then give your final "
    "answer wrapped in <answer>...</answer> tags. Keep the answer itself short "
    "(a word or short phrase)."
)


def load_examples(dataset: str, n_examples: int, seed: int) -> list[dict]:
    cfg = DATASETS[dataset]
    split_path = REPO_ROOT / "data" / dataset / f"{cfg['eval_split']}.jsonl"
    if not split_path.exists():
        raise FileNotFoundError(
            f"{split_path} missing -- run download_data.py --dataset {dataset} first"
        )
    examples = [json.loads(line) for line in open(split_path)]
    if n_examples and n_examples < len(examples):
        rng = random.Random(seed)
        examples = rng.sample(examples, n_examples)
    return examples


def build_prompts(examples: list[dict]) -> list[str]:
    return [f"Question: {ex['question']}" for ex in examples]


def run_vllm(model: str, prompts: list[str], enable_thinking: bool, decoding: dict) -> list[str]:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(model)
    chat_prompts = []
    for p in prompts:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": p},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        chat_prompts.append(text)

    llm = LLM(model=model)
    sampling_params = SamplingParams(**decoding)
    outputs = llm.generate(chat_prompts, sampling_params)
    # vLLM preserves input order
    return [o.outputs[0].text for o in outputs], chat_prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--exp", required=True, help="experiment name, e.g. exp1")
    args = ap.parse_args()

    exp_dir = REPO_ROOT / "experiment" / args.exp
    settings = json.loads((exp_dir / "settings.json").read_text())

    examples = load_examples(args.dataset, settings["n_examples"], settings["seed"])
    prompts = build_prompts(examples)
    enable_thinking = settings["condition"] == "reasoning"

    raw_outputs, chat_prompts = run_vllm(
        settings["model"], prompts, enable_thinking, settings["decoding"]
    )

    out_path = exp_dir / "outputs" / f"{args.dataset}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for ex, prompt, raw in zip(examples, chat_prompts, raw_outputs):
            f.write(json.dumps({
                "id": ex["id"],
                "question": ex["question"],
                "golden_answers": ex["golden_answers"],
                "prompt": prompt,
                "raw_output": raw,
                "predicted_answer": extract_answer(raw),
            }) + "\n")

    print(f"[done] {args.dataset}: {len(examples)} examples -> {out_path}")


if __name__ == "__main__":
    main()

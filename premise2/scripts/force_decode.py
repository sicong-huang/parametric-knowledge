"""Tier 2 Task 2.3: force-decode continuations from the three-way corrupted
prefix set.

For each prefix row in premise2/corpus/tier2_prefixes.jsonl, splice the prefix
text after the original chat-template question prompt, and regenerate the rest
of the trace + final answer from the SAME local model that produced the
original trace (never an API call) -- reusing scripts/generate.py's exact
tokenizer/vLLM invocation pattern so the prompt rendering matches training-time
formatting. n=4 continuations per condition at temperature 1.0.

vLLM here is the offline in-process LLM(), same as scripts/generate.py:
run_vllm -- there is no separate "continuation" API; force-decoding is done by
appending the prefix string directly to the rendered chat prompt (which already
ends right after the assistant turn opens, so appending is equivalent to what
the model would have generated itself) and calling llm.generate() again.

Usage (one model at a time -- avoid re-loading vLLM per-row):
    VLLM_USE_FLASHINFER_SAMPLER=0 uv run python premise2/scripts/force_decode.py \\
        --model "nvidia/NVIDIA-Nemotron-Nano-9B-v2" --exp exp14
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from evaluation.parse import extract_answer

REPO_ROOT = pathlib.Path(__file__).parent.parent.parent
PREFIXES_PATH = REPO_ROOT / "premise2" / "corpus" / "tier2_prefixes.jsonl"
MANIFEST_PATH = REPO_ROOT / "premise2" / "corpus" / "traces_manifest.jsonl"
CONTINUATIONS_OUT = REPO_ROOT / "premise2" / "corpus" / "tier2_continuations.jsonl"

N_SAMPLES = 4
DECODING = {"temperature": 1.0, "top_p": 0.95, "max_tokens": 4096}


def build_chat_prefix(model: str, exp_name: str, question: str, corrupted_reasoning_prefix: str) -> str:
    """Render the same system+user chat template scripts/generate.py used for
    this exp (condition=reasoning), then append the corrupted reasoning prefix
    text so the model continues from exactly that point."""
    from transformers import AutoTokenizer
    import importlib

    generate_mod = importlib.import_module("scripts.generate")
    settings = json.loads((REPO_ROOT / "experiment" / exp_name / "settings.json").read_text())
    toggle_style = settings.get("toggle_style", "kwarg")

    tokenizer = AutoTokenizer.from_pretrained(model)
    if toggle_style == "sysprompt":
        system_content = generate_mod.SYSTEM_PROMPT + "\n/think"
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"Question: {question}"},
        ]
        chat_prefix = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        messages = [
            {"role": "system", "content": generate_mod.SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}"},
        ]
        chat_prefix = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
    return chat_prefix + corrupted_reasoning_prefix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id, e.g. nvidia/NVIDIA-Nemotron-Nano-9B-v2")
    ap.add_argument("--exp", required=True, help="exp dir this model's traces came from, e.g. exp14")
    args = ap.parse_args()

    # Load question text per trace_id from the manifest.
    question_by_trace = {}
    for line in open(MANIFEST_PATH):
        r = json.loads(line)
        if r["model"] == args.model:
            question_by_trace[r["trace_id"]] = r["question"]

    rows = []
    with open(PREFIXES_PATH) as f:
        for line in f:
            row = json.loads(line)
            if row["prefix_text"] is None:
                continue  # rejected corruption
            if row["trace_id"] not in question_by_trace:
                continue  # different model's trace
            rows.append(row)

    print(f"[force_decode] {len(rows)} prefix rows for model={args.model}")
    if not rows:
        print("nothing to do")
        return

    full_prompts = []
    for row in rows:
        question = question_by_trace[row["trace_id"]]
        full_prompts.append(build_chat_prefix(args.model, args.exp, question, row["prefix_text"]))

    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model)
    sampling_params = SamplingParams(n=N_SAMPLES, **DECODING)
    outputs = llm.generate(full_prompts, sampling_params)

    out_rows = []
    for row, output in zip(rows, outputs):
        for sample_id, sample in enumerate(output.outputs):
            continuation_text = sample.text
            out_rows.append({
                "trace_id": row["trace_id"],
                "hop_index": row["hop_index"],
                "condition": row["condition"],
                "sample_id": sample_id,
                "continuation_text": continuation_text,
                "final_answer": extract_answer(continuation_text),
            })

    CONTINUATIONS_OUT.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if CONTINUATIONS_OUT.exists() else "w"
    with open(CONTINUATIONS_OUT, mode) as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")

    print(f"[done] {len(out_rows)} continuations -> {CONTINUATIONS_OUT} (mode={mode})")


if __name__ == "__main__":
    main()

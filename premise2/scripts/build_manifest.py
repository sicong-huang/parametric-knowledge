"""Freeze the reasoning-condition trace corpus for Premise 2 (Tier 0, Task 0.1).

Reads the existing reasoning-condition generations from experiment/exp{10,14,16,18}
(the 4 models locked for this work -- Qwen models excluded) and writes one row per
trace to premise2/corpus/traces_manifest.jsonl.

Do NOT regenerate anything here -- this only reads outputs/*.jsonl that already exist.

Usage:
    uv run python premise2/scripts/build_manifest.py
"""
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from evaluation.metrics import exact_match
from evaluation.parse import extract_think

REPO_ROOT = pathlib.Path(__file__).parent.parent.parent
MANIFEST_OUT = REPO_ROOT / "premise2" / "corpus" / "traces_manifest.jsonl"

# The 4 locked exp dirs (reasoning condition), Qwen dirs excluded per user decision.
REASONING_EXP_DIRS = ["exp10", "exp14", "exp16", "exp18"]

_ANSWER_TAG_RE = re.compile(r"<answer>.*?</answer>", re.DOTALL)


def extract_reasoning(raw: str) -> str:
    """extract_think only catches <think>...</think>; Nemotron models use that
    format, but the gemma-4 family emits a plain "thought\\n..." prefix with no
    closing tag. Fall back to the raw text with any <answer> block stripped out
    so we still get a usable reasoning span for every model."""
    think = extract_think(raw)
    if think.strip():
        return think
    return _ANSWER_TAG_RE.sub("", raw).strip()


def build_manifest():
    rows = []
    counts = {}

    for exp_name in REASONING_EXP_DIRS:
        exp_dir = REPO_ROOT / "experiment" / exp_name
        settings = json.loads((exp_dir / "settings.json").read_text())
        model = settings["model"]
        exp_id = settings["exp_id"]
        assert settings["condition"] == "reasoning", (
            f"{exp_name} is not a reasoning-condition run: {settings['condition']}"
        )

        outputs_dir = exp_dir / "outputs"
        for outputs_path in sorted(outputs_dir.glob("*.jsonl")):
            dataset = outputs_path.stem
            n = 0
            for line in open(outputs_path):
                rec = json.loads(line)
                raw_output = rec["raw_output"]
                predicted_answer = rec["predicted_answer"]
                golden_answers = rec["golden_answers"]
                rows.append({
                    "trace_id": f"{exp_name}:{dataset}:{rec['id']}",
                    "model": model,
                    "exp_id": exp_id,
                    "exp_name": exp_name,
                    "dataset": dataset,
                    "question_id": rec["id"],
                    "question": rec["question"],
                    "trace_text": raw_output,
                    "reasoning": extract_reasoning(raw_output),
                    "final_answer": predicted_answer,
                    "gold_answer": golden_answers,
                    "em_correct": exact_match(predicted_answer, golden_answers),
                })
                n += 1
            counts[(model, dataset)] = n

    MANIFEST_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_OUT, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"[done] {len(rows)} traces -> {MANIFEST_OUT}")
    print("\nPer (model, dataset) counts:")
    for (model, dataset), n in sorted(counts.items()):
        print(f"  {model:45s} {dataset:20s} {n}")


if __name__ == "__main__":
    build_manifest()

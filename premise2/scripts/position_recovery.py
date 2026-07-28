"""Tier 2 Task 2.6: position and self-correction analysis.

Classification pass over the existing tier2_continuations.jsonl (no new
generation): breaks out flip rate by hop position (hop 1 vs hop 2 vs later),
and classifies each corrupted-condition continuation for self-correction --
did the model notice/contradict the injected error before answering.

Self-correction detection is a lexical heuristic (checks for contradiction/
correction markers -- "actually", "wait", "correction", "mistake", "let me
reconsider", "no,", "I made an error" -- appearing after the corrupted
sentence's position in the continuation). This is a coarse proxy, documented
as such; a manual spot-check of a sample is recommended before citing the
self-correction rate as precise.

Usage:
    uv run python premise2/scripts/position_recovery.py
"""
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from evaluation.metrics import exact_match

REPO_ROOT = pathlib.Path(__file__).parent.parent.parent
CONTINUATIONS_PATH = REPO_ROOT / "premise2" / "corpus" / "tier2_continuations.jsonl"
CANDIDATES_PATH = REPO_ROOT / "premise2" / "corpus" / "tier2_candidates.jsonl"
MANIFEST_PATH = REPO_ROOT / "premise2" / "corpus" / "traces_manifest.jsonl"
OUT_PATH = REPO_ROOT / "premise2" / "reports" / "tier2_position_and_recovery.md"

SELF_CORRECTION_MARKERS = re.compile(
    r"\b(actually|wait[,.]|correction|mistake|let me reconsider|"
    r"i made an error|that'?s (?:not right|incorrect|wrong)|"
    r"on second thought|let me re-?check|i need to correct)\b",
    re.IGNORECASE,
)


def classify_self_correction(continuation_text: str) -> bool:
    return bool(SELF_CORRECTION_MARKERS.search(continuation_text))


def main():
    gold_by_trace = {}
    model_by_trace = {}
    for line in open(MANIFEST_PATH):
        r = json.loads(line)
        gold_by_trace[r["trace_id"]] = r["gold_answer"]
        model_by_trace[r["trace_id"]] = r["model"]

    # trace_id -> hop_index -> total hop count (for position bucketing: first/mid/last)
    hop_count_by_trace = {}
    for line in open(CANDIDATES_PATH):
        row = json.loads(line)
        hop_count_by_trace[row["trace_id"]] = sum(1 for h in row["hop_facts"] if h["matched"])

    # (model, position_bucket) -> [is_wrong,...] for corrupted condition
    flip_by_position = {}
    # model -> [n_self_corrected, n_total] for corrupted condition
    recovery_by_model = {}

    for line in open(CONTINUATIONS_PATH):
        row = json.loads(line)
        if row["condition"] != "corrupted":
            continue
        trace_id = row["trace_id"]
        if trace_id not in gold_by_trace:
            continue
        model = model_by_trace[trace_id]
        n_hops = hop_count_by_trace.get(trace_id, 1)
        hop_index = row["hop_index"]
        if n_hops <= 1:
            bucket = "hop1(only)"
        elif hop_index == 0:
            bucket = "hop1"
        elif hop_index == n_hops - 1:
            bucket = "last"
        else:
            bucket = "mid"

        is_wrong = 1 - exact_match(row["final_answer"], gold_by_trace[trace_id])
        flip_by_position.setdefault((model, bucket), []).append(is_wrong)

        self_corrected = classify_self_correction(row["continuation_text"])
        rec = recovery_by_model.setdefault(model, [0, 0])
        rec[0] += int(self_corrected)
        rec[1] += 1

    lines = ["# Tier 2 — position and self-correction analysis\n"]
    lines.append(
        "Classification pass over existing `tier2_continuations.jsonl` (corrupted "
        "condition only). No new generation. Self-correction is a lexical-marker "
        "heuristic — treat as a coarse proxy, not a precise rate; spot-check before citing.\n"
    )

    lines.append("## Flip rate by hop position\n")
    lines.append("| Model | Position | n | P(wrong) |")
    lines.append("| :-- | :-- | --: | --: |")
    for (model, bucket), vals in sorted(flip_by_position.items()):
        lines.append(f"| {model} | {bucket} | {len(vals)} | {sum(vals)/len(vals):.3f} |")

    lines.append("\n## Self-correction rate by model (corrupted condition)\n")
    lines.append("| Model | n | n_self_corrected | self_correction_rate |")
    lines.append("| :-- | --: | --: | --: |")
    for model, (n_corrected, n_total) in sorted(recovery_by_model.items()):
        rate = n_corrected / n_total if n_total else 0.0
        lines.append(f"| {model} | {n_total} | {n_corrected} | {rate:.3f} |")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines) + "\n")
    print(f"[done] -> {OUT_PATH}")


if __name__ == "__main__":
    main()

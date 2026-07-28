"""Tier 2 Task 2.4: score flips from force-decoded continuations.

Grades tier2_continuations.jsonl final_answer against the trace's gold answer
via the existing EM pipeline (evaluation/metrics.exact_match) -- no SAFE/judge
needed since ground truth is already known. Computes, per (model, dataset):
  P(wrong | corrupted) - P(wrong | paraphrase)   [the causal flip-rate gap]
  P(wrong | corrupted) - P(wrong | original)     [sanity-check gap]
with bootstrap CIs over questions (trace_id + hop_index unit).

Usage:
    uv run python premise2/scripts/score_flips.py [--out premise2/reports/tier2_results.md]
"""
import argparse
import collections
import json
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from evaluation.metrics import exact_match

REPO_ROOT = pathlib.Path(__file__).parent.parent.parent
CONTINUATIONS_PATH = REPO_ROOT / "premise2" / "corpus" / "tier2_continuations.jsonl"
MANIFEST_PATH = REPO_ROOT / "premise2" / "corpus" / "traces_manifest.jsonl"
DEFAULT_OUT = REPO_ROOT / "premise2" / "reports" / "tier2_results.md"

N_BOOTSTRAP = 2000


def bootstrap_ci(values: list[float], n_boot=N_BOOTSTRAP, seed=0):
    if not values:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot)]
    return (sum(values) / n, lo, hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--continuations", default=str(CONTINUATIONS_PATH))
    args = ap.parse_args()

    gold_by_trace = {}
    model_dataset_by_trace = {}
    for line in open(MANIFEST_PATH):
        r = json.loads(line)
        gold_by_trace[r["trace_id"]] = r["gold_answer"]
        model_dataset_by_trace[r["trace_id"]] = (r["model"], r["dataset"])

    # per (model, dataset, condition) -> list of per-question wrong-fraction
    # (average over the n=4 samples for that trace_id+hop_index, so the unit
    # of the bootstrap is the question, not the individual sample)
    per_question_wrong = collections.defaultdict(lambda: collections.defaultdict(list))
    # (model, dataset) -> condition -> question_key -> [is_wrong,...]
    raw = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))

    n_rows = 0
    for line in open(args.continuations):
        row = json.loads(line)
        n_rows += 1
        trace_id = row["trace_id"]
        if trace_id not in gold_by_trace:
            continue
        model, dataset = model_dataset_by_trace[trace_id]
        gold = gold_by_trace[trace_id]
        is_wrong = 1 - exact_match(row["final_answer"], gold)
        key = (model, dataset)
        qkey = f"{trace_id}:{row['hop_index']}"
        raw[key][row["condition"]][qkey].append(is_wrong)

    lines = ["# Tier 2 results — force-decode flip rates\n"]
    lines.append(f"Source: `{args.continuations}` ({n_rows} continuation rows).\n")
    lines.append(
        "Per question (trace_id + hop_index), wrong-rate = mean(is_wrong) over the "
        f"n={{samples}} force-decoded continuations for that condition. "
        "Bootstrap 95% CI over questions.\n"
    )
    lines.append("| Model | Dataset | n_q | P(wrong\\|orig) | P(wrong\\|corrupted) | P(wrong\\|paraphrase) | corrupted-paraphrase gap | corrupted-original gap |")
    lines.append("| :-- | :-- | --: | --: | --: | --: | --: | --: |")

    for (model, dataset), by_cond in sorted(raw.items()):
        q_keys = set()
        for cond in ("original", "corrupted", "paraphrase"):
            q_keys |= set(by_cond.get(cond, {}).keys())
        # only keep questions present in all 3 conditions for a fair paired comparison
        q_keys = [
            q for q in q_keys
            if q in by_cond.get("original", {}) and q in by_cond.get("corrupted", {}) and q in by_cond.get("paraphrase", {})
        ]
        if not q_keys:
            continue

        def wrong_frac(cond):
            return [sum(by_cond[cond][q]) / len(by_cond[cond][q]) for q in q_keys]

        orig_vals = wrong_frac("original")
        corr_vals = wrong_frac("corrupted")
        para_vals = wrong_frac("paraphrase")

        orig_mean, _, _ = bootstrap_ci(orig_vals)
        corr_mean, corr_lo, corr_hi = bootstrap_ci(corr_vals)
        para_mean, para_lo, para_hi = bootstrap_ci(para_vals)

        gap_corr_para = [c - p for c, p in zip(corr_vals, para_vals)]
        gap_mean, gap_lo, gap_hi = bootstrap_ci(gap_corr_para)
        gap_corr_orig = [c - o for c, o in zip(corr_vals, orig_vals)]
        gap2_mean, gap2_lo, gap2_hi = bootstrap_ci(gap_corr_orig)

        lines.append(
            f"| {model} | {dataset} | {len(q_keys)} | {orig_mean:.3f} | "
            f"{corr_mean:.3f} [{corr_lo:.3f},{corr_hi:.3f}] | "
            f"{para_mean:.3f} [{para_lo:.3f},{para_hi:.3f}] | "
            f"{gap_mean:+.3f} [{gap_lo:+.3f},{gap_hi:+.3f}] | "
            f"{gap2_mean:+.3f} [{gap2_lo:+.3f},{gap2_hi:+.3f}] |"
        )

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"[done] -> {out_path}")


if __name__ == "__main__":
    main()

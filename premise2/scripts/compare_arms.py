"""Tier 2 placebo-fix: difference-in-differences + no-op instability reporting.

exp20.md's original placebo check compared cell means by eye
(tier2_results.md vs tier2_placebo_results.md) and never computed a paired
statistic, so it couldn't tell "the placebo gap is real" from "the placebo
gap is noise from a smaller/harder trace population." This script:

  1. Restricts to the (trace_id, hop_index) pairs present in BOTH the main and
     placebo continuation files (after placebo.py's position-matching, this
     should be most of the main-arm's population) and computes a paired
     difference-in-differences: (main's corrupted-paraphrase gap) minus
     (placebo's corrupted-paraphrase gap), bootstrapped over questions.
     If DiD excludes zero with the main gap positive, the on-chain effect is
     distinguishable from generic force-decode instability. If it doesn't,
     the placebo still tracks the main arm and Tier 2 is not yet citable as
     causal (per the plan's Guardrails).

  2. Reports P(wrong | original) per arm explicitly. Every candidate trace was
     em_correct==1 before any splicing, so any nonzero value here is pure
     temperature-1.0 resampling noise, not an effect of the edit. This is
     exp20.md's Decision item (c) ("add a no-op condition") -- already
     satisfied by the existing `original` condition in both arms; it only
     needed to be reported, not re-run.

Usage:
    uv run python premise2/scripts/compare_arms.py
"""
import collections
import json
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from evaluation.metrics import exact_match

REPO_ROOT = pathlib.Path(__file__).parent.parent.parent
MAIN_CONTINUATIONS_PATH = REPO_ROOT / "premise2" / "corpus" / "tier2_continuations.jsonl"
PLACEBO_CONTINUATIONS_PATH = REPO_ROOT / "premise2" / "corpus" / "tier2_placebo_continuations.jsonl"
MANIFEST_PATH = REPO_ROOT / "premise2" / "corpus" / "traces_manifest.jsonl"
OUT_PATH = REPO_ROOT / "premise2" / "reports" / "tier2_arm_comparison.md"

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


def load_per_question_wrong(path, gold_by_trace, model_dataset_by_trace):
    """(model, dataset) -> condition -> qkey -> [is_wrong,...], qkey='trace_id:hop_index'."""
    raw = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
    if not path.exists():
        return raw
    for line in open(path):
        row = json.loads(line)
        trace_id = row["trace_id"]
        if trace_id not in gold_by_trace:
            continue
        model, dataset = model_dataset_by_trace[trace_id]
        is_wrong = 1 - exact_match(row["final_answer"], gold_by_trace[trace_id])
        qkey = f"{trace_id}:{row['hop_index']}"
        raw[(model, dataset)][row["condition"]][qkey].append(is_wrong)
    return raw


def wrong_frac_by_q(by_cond, cond, q_keys):
    return {q: sum(by_cond[cond][q]) / len(by_cond[cond][q]) for q in q_keys if q in by_cond.get(cond, {})}


def main():
    gold_by_trace = {}
    model_dataset_by_trace = {}
    for line in open(MANIFEST_PATH):
        r = json.loads(line)
        gold_by_trace[r["trace_id"]] = r["gold_answer"]
        model_dataset_by_trace[r["trace_id"]] = (r["model"], r["dataset"])

    main_raw = load_per_question_wrong(MAIN_CONTINUATIONS_PATH, gold_by_trace, model_dataset_by_trace)
    placebo_raw = load_per_question_wrong(PLACEBO_CONTINUATIONS_PATH, gold_by_trace, model_dataset_by_trace)

    lines = ["# Tier 2 — main vs placebo arm comparison (DiD + no-op instability)\n"]
    lines.append(
        f"Sources: `{MAIN_CONTINUATIONS_PATH}`, `{PLACEBO_CONTINUATIONS_PATH}`.\n"
    )
    lines.append(
        "Paired on (trace_id, hop_index) -- only questions present in both the main "
        "and placebo arms, in all 3 conditions, are used. DiD = main's "
        "(corrupted-paraphrase) gap minus placebo's, per question; bootstrap 95% CI "
        "over questions. If DiD excludes zero and the main gap is positive, the "
        "on-chain effect is distinguishable from generic force-decode instability.\n"
    )
    lines.append(
        "| Model | Dataset | n_paired | main gap | placebo gap | DiD (main-placebo) |"
        " P(wrong\\|orig) main | P(wrong\\|orig) placebo |"
    )
    lines.append("| :-- | :-- | --: | --: | --: | --: | --: | --: |")

    all_keys = sorted(set(main_raw) | set(placebo_raw))
    did_values_pooled = []
    for key in all_keys:
        m_bc = main_raw.get(key, {})
        p_bc = placebo_raw.get(key, {})

        m_q = set(m_bc.get("original", {})) & set(m_bc.get("corrupted", {})) & set(m_bc.get("paraphrase", {}))
        p_q = set(p_bc.get("original", {})) & set(p_bc.get("corrupted", {})) & set(p_bc.get("paraphrase", {}))
        paired_q = sorted(m_q & p_q)
        if not paired_q:
            continue

        m_orig = wrong_frac_by_q(m_bc, "original", paired_q)
        m_corr = wrong_frac_by_q(m_bc, "corrupted", paired_q)
        m_para = wrong_frac_by_q(m_bc, "paraphrase", paired_q)
        p_orig = wrong_frac_by_q(p_bc, "original", paired_q)
        p_corr = wrong_frac_by_q(p_bc, "corrupted", paired_q)
        p_para = wrong_frac_by_q(p_bc, "paraphrase", paired_q)

        main_gaps = [m_corr[q] - m_para[q] for q in paired_q]
        placebo_gaps = [p_corr[q] - p_para[q] for q in paired_q]
        did_values = [mg - pg for mg, pg in zip(main_gaps, placebo_gaps)]
        did_values_pooled.extend(did_values)

        main_gap_mean, _, _ = bootstrap_ci(main_gaps)
        placebo_gap_mean, _, _ = bootstrap_ci(placebo_gaps)
        did_mean, did_lo, did_hi = bootstrap_ci(did_values)
        orig_main_mean = sum(m_orig[q] for q in paired_q) / len(paired_q)
        orig_placebo_mean = sum(p_orig[q] for q in paired_q) / len(paired_q)

        lines.append(
            f"| {key[0]} | {key[1]} | {len(paired_q)} | {main_gap_mean:+.3f} | "
            f"{placebo_gap_mean:+.3f} | {did_mean:+.3f} [{did_lo:+.3f},{did_hi:+.3f}] | "
            f"{orig_main_mean:.3f} | {orig_placebo_mean:.3f} |"
        )

    if did_values_pooled:
        pooled_mean, pooled_lo, pooled_hi = bootstrap_ci(did_values_pooled)
        lines.append("")
        lines.append(
            f"**Pooled DiD across all paired questions (n={len(did_values_pooled)}): "
            f"{pooled_mean:+.3f} [{pooled_lo:+.3f},{pooled_hi:+.3f}]**"
        )

    lines.append("\n## No-op instability (P(wrong | original), all candidates were em_correct==1)\n")
    lines.append(
        "All of these traces were originally answered correctly with no edit at all. "
        "Any nonzero value below is pure temperature-1.0 force-decode resampling noise, "
        "not an effect of corruption -- this is the no-op control (exp20.md Decision "
        "item (c)), already present as the `original` condition in both arms.\n"
    )
    lines.append("| Model | Dataset | n_q main | P(wrong\\|orig) main | n_q placebo | P(wrong\\|orig) placebo |")
    lines.append("| :-- | :-- | --: | --: | --: | --: |")
    for key in all_keys:
        m_bc = main_raw.get(key, {})
        p_bc = placebo_raw.get(key, {})
        m_orig_all = m_bc.get("original", {})
        p_orig_all = p_bc.get("original", {})
        if not m_orig_all and not p_orig_all:
            continue
        m_vals = [sum(v) / len(v) for v in m_orig_all.values()]
        p_vals = [sum(v) / len(v) for v in p_orig_all.values()]
        m_mean = sum(m_vals) / len(m_vals) if m_vals else float("nan")
        p_mean = sum(p_vals) / len(p_vals) if p_vals else float("nan")
        lines.append(f"| {key[0]} | {key[1]} | {len(m_vals)} | {m_mean:.3f} | {len(p_vals)} | {p_mean:.3f} |")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines) + "\n")
    print(f"[done] -> {OUT_PATH}")


if __name__ == "__main__":
    main()

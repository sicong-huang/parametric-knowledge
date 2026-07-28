"""Tier 2 Task 2.5: off-chain placebo check (LOAD-BEARING).

Repeats corrupt_fact.py + force_decode.py + score_flips.py but corrupts a fact
stated in the trace that is NOT on the gold dependency chain, using the
off_chain_sentences already recorded per candidate in tier2_candidates.jsonl
(now purity-filtered, see select_targets.offchain_purity_score). Expect
near-zero flip-rate difference vs the paraphrase control. If the placebo shows
a large effect too, DO NOT write up Tier 2 as causal evidence until the
corruption methodology is revisited (see plan Guardrails).

Position-matched selection (fix for the first run's confound): the original
version picked the first off-chain sentence >=5 words, which lands near the
top of the trace (median prefix 19 words vs the main arm's 70.5) -- so the
placebo cut far more of the trace than the main condition did, and on the
SAME traces with no edit at all, that shallow cut point was ~10pp more
unstable than the main arm's deeper cut (see exp20.md's Decision). This
version pairs each placebo row to a specific on-chain hop and picks the
off-chain sentence whose prefix_words is closest to that hop's, within a
tolerance -- so main and placebo are truncated at comparable depth and the
comparison is paired at the (trace_id, hop_index) level for an exact
difference-in-differences (see compare_arms.py).

This reuses corrupt_fact.build_one_prefix_set / validate_corruption and
force_decode's chat-prefix builder rather than duplicating them -- only the
candidate-selection step differs (off-chain sentence instead of a matched hop).

Usage:
    uv run python premise2/scripts/placebo.py --select    # build off-chain prefixes
    VLLM_USE_FLASHINFER_SAMPLER=0 uv run python premise2/scripts/placebo.py \\
        --force-decode --model "nvidia/NVIDIA-Nemotron-Nano-9B-v2" --exp exp14
    uv run python premise2/scripts/placebo.py --score
"""
import argparse
import concurrent.futures
import json
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import corrupt_fact
import force_decode
from evaluation.metrics import exact_match
from evaluation.parse import extract_answer

REPO_ROOT = pathlib.Path(__file__).parent.parent.parent
CANDIDATES_PATH = REPO_ROOT / "premise2" / "corpus" / "tier2_candidates.jsonl"
MANIFEST_PATH = REPO_ROOT / "premise2" / "corpus" / "traces_manifest.jsonl"
PLACEBO_PREFIXES_OUT = REPO_ROOT / "premise2" / "corpus" / "tier2_placebo_prefixes.jsonl"
PLACEBO_CONTINUATIONS_OUT = REPO_ROOT / "premise2" / "corpus" / "tier2_placebo_continuations.jsonl"
PLACEBO_REPORT_OUT = REPO_ROOT / "premise2" / "reports" / "tier2_placebo_results.md"

# How far (in words) an off-chain sentence's cut position may sit from its
# paired on-chain hop's cut position and still count as "position-matched".
# Chosen loosely -- the failure mode being fixed was a ~50-word median gap
# (70.5 vs 19.0), so a candidate within 15 words is a large improvement even
# though it isn't exact.
POSITION_TOLERANCE_WORDS = 15


def select_offchain_prefixes():
    client = corrupt_fact._client()
    model = __import__("os").environ.get("EXTRACTOR_MODEL", corrupt_fact.DEFAULT_MODEL)

    trace_text_by_id = {}
    with open(MANIFEST_PATH) as f:
        for line in f:
            r = json.loads(line)
            trace_text_by_id[r["trace_id"]] = r["reasoning"]

    jobs = []  # (trace_id, hop_index, sentence, prefix_before)
    n_hops_considered = 0
    n_hops_no_match = 0
    with open(CANDIDATES_PATH) as f:
        for line in f:
            cand = json.loads(line)
            trace_text = trace_text_by_id.get(cand["trace_id"])
            offchain = cand["off_chain_sentences"]
            if trace_text is None or not offchain:
                continue

            used = set()  # off-chain sentences already claimed by an earlier hop in this trace
            for hop in cand["hop_facts"]:
                if not hop["matched"] or hop["prefix_words"] is None:
                    continue
                n_hops_considered += 1
                target_pw = hop["prefix_words"]

                best = None
                best_dist = None
                for oc in offchain:
                    if oc["sentence"] in used or oc["prefix_words"] is None:
                        continue
                    dist = abs(oc["prefix_words"] - target_pw)
                    if dist <= POSITION_TOLERANCE_WORDS and (best_dist is None or dist < best_dist):
                        best, best_dist = oc, dist
                if best is None:
                    n_hops_no_match += 1
                    continue

                used.add(best["sentence"])
                idx = trace_text.find(best["sentence"])
                if idx == -1:
                    continue
                jobs.append((cand["trace_id"], hop["hop_index"], best["sentence"], trace_text[:idx]))

    print(
        f"[placebo select] {len(jobs)} position-matched off-chain jobs "
        f"({n_hops_no_match}/{n_hops_considered} on-chain hops had no off-chain "
        f"sentence within {POSITION_TOLERANCE_WORDS} words, skipped)"
    )

    out_rows = []

    def worker(job):
        trace_id, hop_index, sent, prefix_before = job
        rows = [{"condition": "original", "prefix_text": prefix_before + sent, "sentence": sent}]
        corrupted = corrupt_fact._call(client, model, corrupt_fact.CORRUPT_PROMPT.format(sentence=sent))
        if corrupt_fact.validate_corruption(sent, corrupted):
            rows.append({"condition": "corrupted", "prefix_text": prefix_before + corrupted, "sentence": corrupted})
        else:
            rows.append({"condition": "corrupted", "prefix_text": None, "sentence": corrupted, "rejected": True})
        paraphrase = corrupt_fact._call(client, model, corrupt_fact.PARAPHRASE_PROMPT.format(sentence=sent))
        rows.append({"condition": "paraphrase", "prefix_text": prefix_before + paraphrase, "sentence": paraphrase})
        return trace_id, hop_index, rows

    with concurrent.futures.ThreadPoolExecutor(max_workers=corrupt_fact.CONCURRENCY) as pool:
        futures = [pool.submit(worker, job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            trace_id, hop_index, rows = future.result()
            for r in rows:
                out_rows.append({"trace_id": trace_id, "hop_index": hop_index, "arm": "placebo", **r})

    PLACEBO_PREFIXES_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(PLACEBO_PREFIXES_OUT, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    print(f"[done] {len(out_rows)} placebo prefix rows -> {PLACEBO_PREFIXES_OUT}")


def force_decode_offchain(model: str, exp_name: str):
    question_by_trace = {}
    for line in open(MANIFEST_PATH):
        r = json.loads(line)
        if r["model"] == model:
            question_by_trace[r["trace_id"]] = r["question"]

    rows = []
    with open(PLACEBO_PREFIXES_OUT) as f:
        for line in f:
            row = json.loads(line)
            if row["prefix_text"] is None or row["trace_id"] not in question_by_trace:
                continue
            rows.append(row)

    print(f"[placebo force_decode] {len(rows)} rows for model={model}")
    if not rows:
        return

    full_prompts = [
        force_decode.build_chat_prefix(model, exp_name, question_by_trace[row["trace_id"]], row["prefix_text"])
        for row in rows
    ]

    from vllm import LLM, SamplingParams

    llm = LLM(model=model)
    sampling_params = SamplingParams(n=force_decode.N_SAMPLES, **force_decode.DECODING)
    outputs = llm.generate(full_prompts, sampling_params)

    out_rows = []
    for row, output in zip(rows, outputs):
        for sample_id, sample in enumerate(output.outputs):
            out_rows.append({
                "trace_id": row["trace_id"],
                "hop_index": row["hop_index"],
                "condition": row["condition"],
                "sample_id": sample_id,
                "continuation_text": sample.text,
                "final_answer": extract_answer(sample.text),
            })

    mode = "a" if PLACEBO_CONTINUATIONS_OUT.exists() else "w"
    with open(PLACEBO_CONTINUATIONS_OUT, mode) as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    print(f"[done] {len(out_rows)} placebo continuations -> {PLACEBO_CONTINUATIONS_OUT} (mode={mode})")


def score_placebo():
    # Reuse score_flips.py as a subprocess, pointed at the placebo corpus files.
    import subprocess

    subprocess.run([
        sys.executable, str(REPO_ROOT / "premise2" / "scripts" / "score_flips.py"),
        "--continuations", str(PLACEBO_CONTINUATIONS_OUT),
        "--out", str(PLACEBO_REPORT_OUT),
    ], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--force-decode", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--exp")
    args = ap.parse_args()

    if args.select:
        select_offchain_prefixes()
    if args.force_decode:
        if not args.model or not args.exp:
            raise SystemExit("--force-decode requires --model and --exp")
        force_decode_offchain(args.model, args.exp)
    if args.score:
        score_placebo()


if __name__ == "__main__":
    main()

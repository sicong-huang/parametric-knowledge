"""
Stratified 2-hop QA generation — 20k per pattern (or smoke-test with --short).

Outputs go to <run_dir>/stratified_2hop/ by default.

Usage:
    # Smoke test (~67 per pattern, ~201 total) — enumerate only, no LLM
    uv run syndata_archive/stratified_2hop/create_qa.py 2_hop \\
        --dir syndata_archive/data/bio/run5 --enumerate-only --short

    # Full stratified question generation
    uv run syndata_archive/stratified_2hop/create_qa.py 2_hop \\
        --dir syndata_archive/data/bio/run5 --sample-per-pattern 20000 \\
        --seed 42 --max-workers 100

    # Styled full answers
    uv run syndata_archive/stratified_2hop/create_qa.py 2_hop_full_answer \\
        --dir syndata_archive/data/bio/run5 --style direct
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

# Support running as `uv run syndata_archive/stratified_2hop/create_qa.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    DEFAULT_QA_SUBDIR,
    SHORT_PER_PATTERN,
    _generate_2hop_question,
    _load_wiki_bio_map,
    build_enumerators,
    effective_sample_per_pattern,
    iter_jsonl,
    load_graph,
    make_openai_client,
    qa_dir_for,
    sample_stratified,
    style_direct_2hop,
    style_reasoning_2hop,
    style_sentence_2hop,
)

load_dotenv()

ALL_STYLES = ("direct", "sentence", "reasoning")


def cmd_2_hop(args):
    run_dir = Path(args.dir)
    qa_dir = qa_dir_for(run_dir, args.qa_subdir)
    qa_dir.mkdir(parents=True, exist_ok=True)

    id_to_name, id_to_type, edges, outgoing, incoming, is_person = load_graph(run_dir)
    print(f"Edges: {len(edges):,} total")

    enumerators = build_enumerators(outgoing, incoming, is_person, id_to_name)
    sample_n = effective_sample_per_pattern(args)
    if args.short:
        print(f"Short mode: {sample_n} paths per pattern (~{sample_n * 3} total)")

    records = sample_stratified(enumerators, sample_n, args.seed)

    paths_path = qa_dir / "paths_2_hop.jsonl"
    with paths_path.open("w") as out:
        for rec in records:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records):,} path records to {paths_path}")

    if args.enumerate_only:
        return

    for i, rec in enumerate(records, start=1):
        rec["qaid"] = f"2hop_{i:05d}"

    client = make_openai_client(args)

    output_path = qa_dir / "qa_2_hop.jsonl"
    results: list[dict | None] = [None] * len(records)
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_idx = {
            executor.submit(_generate_2hop_question, client, args.model, rec): i
            for i, rec in enumerate(records)
        }
        for future in tqdm(as_completed(future_to_idx), total=len(records), desc="Generating 2-hop QA"):
            idx = future_to_idx[future]
            results[idx] = future.result()

    with output_path.open("w") as out:
        for rec in results:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(results):,} records to {output_path}")


def cmd_2_hop_full_answer(args):
    run_dir = Path(args.dir)
    qa_dir = qa_dir_for(run_dir, args.qa_subdir)
    input_path = qa_dir / "qa_2_hop.jsonl"
    output_path = qa_dir / f"qa_2_hop_{args.style}.jsonl"

    if not input_path.exists():
        raise FileNotFoundError(
            f"{input_path} not found — run the '2_hop' subcommand first."
        )

    records = list(iter_jsonl(input_path))
    if args.short:
        records = records[:200]
        print(f"Short mode: processing first {len(records)} records")

    existing_by_qaid: dict[str, dict] = {}
    if args.only_missing and output_path.exists():
        for rec in iter_jsonl(output_path):
            qaid = rec.get("qaid")
            if qaid:
                existing_by_qaid[qaid] = rec
        print(f"Only-missing: loaded {len(existing_by_qaid):,} existing {args.style} records")

    to_generate = [
        (i, rec)
        for i, rec in enumerate(records)
        if not (args.only_missing and rec.get("qaid") in existing_by_qaid)
    ]
    if args.only_missing:
        print(
            f"Only-missing: generating {len(to_generate):,} / {len(records):,} "
            f"(skipping {len(records) - len(to_generate):,})"
        )
        if not to_generate:
            print(f"Nothing to do — {output_path} already complete.")
            return

    if args.style == "direct":
        results = [None] * len(records)
        for i, rec in enumerate(records):
            if args.only_missing and rec.get("qaid") in existing_by_qaid:
                results[i] = existing_by_qaid[rec["qaid"]]
            else:
                results[i] = style_direct_2hop(rec)
    elif args.style in ("sentence", "reasoning"):
        print("Loading wiki biographies…")
        bio_map = _load_wiki_bio_map(run_dir)
        print(f"Loaded {len(bio_map):,} wiki biographies")

        id_to_name: dict[str, str] = {}
        for entity in iter_jsonl(run_dir / "entities.jsonl"):
            id_to_name[entity["id"]] = entity["name"]
        print(f"Loaded {len(id_to_name):,} entity names")

        client = make_openai_client(args)

        generate_fn = {
            "sentence": style_sentence_2hop,
            "reasoning": style_reasoning_2hop,
        }[args.style]

        results = [None] * len(records)
        for i, rec in enumerate(records):
            if args.only_missing and rec.get("qaid") in existing_by_qaid:
                results[i] = existing_by_qaid[rec["qaid"]]

        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_idx = {
                executor.submit(
                    generate_fn, client, args.model, rec, bio_map, id_to_name,
                ): i
                for i, rec in to_generate
            }
            for future in tqdm(
                as_completed(future_to_idx),
                total=len(to_generate),
                desc=f"Generating {args.style} answers",
            ):
                idx = future_to_idx[future]
                results[idx] = future.result()
    else:
        raise ValueError(f"Unknown style: {args.style}")

    with output_path.open("w") as out:
        for rec in results:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(results):,} records to {output_path}")


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dir",
        default="syndata_archive/data/bio/run5",
        help="Run directory (default: syndata_archive/data/bio/run5).",
    )
    parser.add_argument(
        "--qa-subdir",
        default=DEFAULT_QA_SUBDIR,
        help=f"QA output subfolder under run dir (default: {DEFAULT_QA_SUBDIR}).",
    )
    parser.add_argument("--model", default="gemma4", help="OpenAI-compatible model id.")
    parser.add_argument("--qa-base-url", default=None, help="OpenAI API base URL override.")
    parser.add_argument("--qa-api-key", default=None, help="API key (optional for local vLLM; defaults to EMPTY).")
    parser.add_argument("--max-workers", type=int, default=8, help="Parallel LLM workers.")
    parser.add_argument(
        "--short",
        action="store_true",
        help=f"Dev mode: {SHORT_PER_PATTERN} paths per pattern for 2_hop; first 200 for full_answer.",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Stratified 2-hop QA generation from a knowledge graph run directory.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    hp = subparsers.add_parser("2_hop", help="Generate stratified 2-hop QA pairs.")
    _add_common_args(hp)
    hp.add_argument(
        "--sample-per-pattern",
        type=int,
        default=None,
        help="Sample this many paths per pattern (ignored when --short is set).",
    )
    hp.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    hp.add_argument(
        "--enumerate-only",
        action="store_true",
        help="Write paths_2_hop.jsonl only; skip LLM question generation.",
    )

    fa = subparsers.add_parser(
        "2_hop_full_answer",
        help="Create styled full-answer files from qa_2_hop.jsonl.",
    )
    _add_common_args(fa)
    fa.add_argument("--style", choices=ALL_STYLES, required=True)
    fa.add_argument(
        "--only-missing",
        action="store_true",
        help=(
            "Reuse existing qa_2_hop_{style}.jsonl rows by qaid; "
            "only generate for qaids present in qa_2_hop.jsonl but missing from the styled file."
        ),
    )

    args = parser.parse_args()

    if args.command == "2_hop":
        cmd_2_hop(args)
    elif args.command == "2_hop_full_answer":
        cmd_2_hop_full_answer(args)


if __name__ == "__main__":
    main()

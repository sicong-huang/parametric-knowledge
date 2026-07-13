"""
Stratified train/test split for 2-hop QA.

Default: global edge-disjoint split (no KG triplet in both train and test),
target 70% train / 30% test. Same qaids for base, direct, and reasoning.

Usage:
    uv run syndata_archive/stratified_2hop/split_qa.py \\
        --dir syndata_archive/data/bio/run5 --seed 42

    # Naive per-pattern shuffle (50/50, allows edge leakage)
    uv run syndata_archive/stratified_2hop/split_qa.py \\
        --dir syndata_archive/data/bio/run5 --random-split --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import DEFAULT_QA_SUBDIR, PATTERNS, iter_jsonl, pattern_seed, qa_dir_for  # noqa: E402

EdgeTriplet = tuple[str, str, str]


def edge_triplets(record: dict) -> list[EdgeTriplet]:
    """Return both (source_id, relation_type, target_id) triplets for a 2-hop row."""
    return [
        (e["source_id"], e["relation_type"], e["target_id"])
        for e in record["edges"]
    ]


def round_robin_by_pattern(records: list[dict], seed: int) -> list[dict]:
    """Shuffle within each pattern, then interleave for balanced processing."""
    by_pattern: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_pattern[rec["pattern"]].append(rec)
    for pattern in PATTERNS:
        batch = by_pattern.get(pattern, [])
        rng = random.Random(pattern_seed(seed, pattern))
        rng.shuffle(batch)
        by_pattern[pattern] = batch

    ordered: list[dict] = []
    indices = {p: 0 for p in PATTERNS}
    while True:
        progressed = False
        for pattern in PATTERNS:
            batch = by_pattern.get(pattern, [])
            idx = indices[pattern]
            if idx < len(batch):
                ordered.append(batch[idx])
                indices[pattern] = idx + 1
                progressed = True
        if not progressed:
            break
    return ordered


def split_records_random(
    records: list[dict], seed: int, train_fraction: float = 0.5,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Naive stratified shuffle split (may leak edges across train/test)."""
    by_pattern: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_pattern[rec["pattern"]].append(rec)

    train: list[dict] = []
    test: list[dict] = []
    for pattern in PATTERNS:
        batch = by_pattern[pattern]
        rng = random.Random(pattern_seed(seed, pattern))
        shuffled = batch.copy()
        rng.shuffle(shuffled)
        n_train = int(round(len(shuffled) * train_fraction))
        train.extend(shuffled[:n_train])
        test.extend(shuffled[n_train:])
        print(
            f"  {pattern}: {n_train:,} train + {len(shuffled) - n_train:,} test"
        )

    return train, test, []


def split_records_edge_disjoint(
    records: list[dict],
    seed: int,
    train_fraction: float = 0.7,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Global edge-disjoint greedy split.

    Each triplet (source_id, relation_type, target_id) may appear in train
    OR test, never both. Rows whose edges are already split across sides are
    dropped unless a row's edges are already split across sides.
    """
    ordered = round_robin_by_pattern(records, seed)
    edge_side: dict[EdgeTriplet, str] = {}
    train: list[dict] = []
    test: list[dict] = []
    dropped: list[dict] = []

    for rec in ordered:
        edges = edge_triplets(rec)
        train_ok = all(
            edge_side.get(t) in (None, "train") for t in edges
        )
        test_ok = all(
            edge_side.get(t) in (None, "test") for t in edges
        )

        if not train_ok and not test_ok:
            dropped.append(rec)
            continue
        if train_ok and test_ok:
            assigned = len(train) + len(test)
            current_train_frac = len(train) / assigned if assigned else 0.0
            side = "train" if current_train_frac < train_fraction else "test"
        elif train_ok:
            side = "train"
        else:
            side = "test"

        if side == "train":
            train.append(rec)
        else:
            test.append(rec)
        for t in edges:
            edge_side[t] = side

    return train, test, dropped


def verify_edge_disjoint(train: list[dict], test: list[dict]) -> int:
    """Return count of triplets shared between train and test (expect 0)."""
    train_edges: set[EdgeTriplet] = set()
    test_edges: set[EdgeTriplet] = set()
    for rec in train:
        train_edges.update(edge_triplets(rec))
    for rec in test:
        test_edges.update(edge_triplets(rec))
    return len(train_edges & test_edges)


def pattern_counts(records: list[dict]) -> dict[str, int]:
    c: Counter[str] = Counter()
    for rec in records:
        c[rec["pattern"]] += 1
    return dict(c)


def write_split(
    split_dir: Path,
    train: list[dict],
    test: list[dict],
    dropped: list[dict],
    suffix: str,
) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    fname = f"qa_2_hop{suffix}.jsonl"
    for subset, name in (
        (train, "train"),
        (test, "test"),
        (dropped, "dropped"),
    ):
        if not subset and name == "dropped":
            continue
        out_path = split_dir / fname.replace(".jsonl", f"_{name}.jsonl")
        with out_path.open("w") as out:
            for rec in subset:
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Wrote {len(subset):,} records to {out_path}")


def apply_split_to_styled(
    split_dir: Path,
    qa_dir: Path,
    train: list[dict],
    test: list[dict],
    dropped: list[dict],
    style: str,
    short: bool,
) -> None:
    styled_path = qa_dir / f"qa_2_hop_{style}.jsonl"
    if not styled_path.exists():
        print(f"Skipping {styled_path.name} (not found)")
        return
    styled_records = list(iter_jsonl(styled_path))
    if short:
        styled_records = styled_records[:200]
    by_qaid = {rec["qaid"]: rec for rec in styled_records}
    train_s = [by_qaid[r["qaid"]] for r in train if r["qaid"] in by_qaid]
    test_s = [by_qaid[r["qaid"]] for r in test if r["qaid"] in by_qaid]
    dropped_s = [by_qaid[r["qaid"]] for r in dropped if r["qaid"] in by_qaid]
    print(f"Splitting qa_2_hop_{style}.jsonl:")
    write_split(split_dir, train_s, test_s, dropped_s, suffix=f"_{style}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stratified train/test split for 2-hop QA.",
    )
    parser.add_argument("--dir", default="syndata_archive/data/bio/run5")
    parser.add_argument("--qa-subdir", default=DEFAULT_QA_SUBDIR)
    parser.add_argument(
        "--split-subdir",
        default="qa_split",
        help="Subfolder under qa-subdir for split outputs (default: qa_split).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.7,
        help="Target train fraction (default: 0.7).",
    )
    parser.add_argument(
        "--random-split",
        action="store_true",
        help="Naive shuffle split (allows edge leakage across train/test).",
    )
    parser.add_argument(
        "--short",
        action="store_true",
        help="Split only the first 200 base records (for smoke-test datasets).",
    )
    args = parser.parse_args()

    run_dir = Path(args.dir)
    qa_dir = qa_dir_for(run_dir, args.qa_subdir)
    split_dir = qa_dir / args.split_subdir

    base_path = qa_dir / "qa_2_hop.jsonl"
    if not base_path.exists():
        raise FileNotFoundError(f"{base_path} not found")

    records = list(iter_jsonl(base_path))
    if args.short:
        records = records[:200]
        print(f"Short mode: splitting first {len(records)} records")

    mode = "random" if args.random_split else "edge_disjoint"
    print(
        f"Splitting {len(records):,} records "
        f"(mode={mode}, train_fraction={args.train_fraction})"
    )

    if args.random_split:
        train, test, dropped = split_records_random(
            records, args.seed, args.train_fraction,
        )
    else:
        train, test, dropped = split_records_edge_disjoint(
            records,
            args.seed,
            args.train_fraction,
        )

    overlap = verify_edge_disjoint(train, test)
    kept = len(train) + len(test)
    report = {
        "mode": mode,
        "seed": args.seed,
        "train_fraction_target": args.train_fraction,
        "total_input": len(records),
        "train": len(train),
        "test": len(test),
        "dropped": len(dropped),
        "kept": kept,
        "actual_train_fraction": len(train) / kept if kept else 0.0,
        "edge_overlap_train_test": overlap,
        "by_pattern": {
            "train": pattern_counts(train),
            "test": pattern_counts(test),
            "dropped": pattern_counts(dropped),
        },
    }

    print(f"\nSummary: {len(train):,} train + {len(test):,} test + {len(dropped):,} dropped")
    print(f"  actual train fraction: {report['actual_train_fraction']:.1%}")
    print(f"  edge overlap train∩test: {overlap} (expect 0 for edge_disjoint)")
    for pat in PATTERNS:
        print(
            f"  {pat}: train={report['by_pattern']['train'].get(pat, 0):,} "
            f"test={report['by_pattern']['test'].get(pat, 0):,} "
            f"dropped={report['by_pattern']['dropped'].get(pat, 0):,}"
        )

    if overlap:
        raise RuntimeError(
            f"Edge overlap between train and test: {overlap} triplets"
        )

    write_split(split_dir, train, test, dropped, suffix="")
    for style in ("direct", "sentence", "reasoning"):
        apply_split_to_styled(
            split_dir, qa_dir, train, test, dropped, style, args.short,
        )

    report_path = split_dir / "split_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nReport → {report_path}")


if __name__ == "__main__":
    main()

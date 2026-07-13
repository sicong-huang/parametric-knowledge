"""
Edge-disjoint train/test split for 1-hop QA (mirrors stratified 2-hop logic).

Default: global edge-disjoint, 70% train / 30% test. Same qaids for base,
direct, sentence, and reasoning. Outputs under <run_dir>/stratified_1hop/qa_split/.

Usage:
    uv run syndata_archive/stratified_1hop/split_qa.py \\
        --dir syndata_archive/data/bio/run5 --seed 42

    uv run syndata_archive/stratified_1hop/split_qa.py \\
        --dir syndata_archive/data/bio/run5 --random-split --train-fraction 0.7
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stratified_2hop"))

from split_qa import (  # noqa: E402
    edge_triplets,
    verify_edge_disjoint,
)

DEFAULT_QA_SUBDIR = "stratified_1hop"

def _relation_seed(seed: int, relation: str) -> int:
    digest = hashlib.sha256(f"{seed}:1hop:{relation}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def order_by_relation(records: list[dict], seed: int) -> list[dict]:
    """Shuffle within each relation_type, then round-robin (like 2-hop patterns)."""
    by_rel: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        rel = rec["edges"][0]["relation_type"]
        by_rel[rel].append(rec)

    relations = sorted(by_rel.keys())
    for rel in relations:
        rng = random.Random(_relation_seed(seed, rel))
        rng.shuffle(by_rel[rel])

    ordered: list[dict] = []
    indices = {r: 0 for r in relations}
    while True:
        progressed = False
        for rel in relations:
            batch = by_rel[rel]
            idx = indices[rel]
            if idx < len(batch):
                ordered.append(batch[idx])
                indices[rel] = idx + 1
                progressed = True
        if not progressed:
            break
    return ordered


def split_1hop_edge_disjoint(
    records: list[dict],
    seed: int,
    train_fraction: float = 0.7,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Same greedy edge-disjoint algorithm as 2-hop, ordered by relation."""
    # Temporarily inject pattern=relation so we can reuse split_records_edge_disjoint
    # via round_robin_by_pattern — but that uses fixed PATTERNS. So run locally.
    ordered = order_by_relation(records, seed)
    edge_side: dict[tuple, str] = {}
    train: list[dict] = []
    test: list[dict] = []
    dropped: list[dict] = []

    for rec in ordered:
        edges = edge_triplets(rec)
        train_ok = all(edge_side.get(t) in (None, "train") for t in edges)
        test_ok = all(edge_side.get(t) in (None, "test") for t in edges)

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


def split_1hop_random(
    records: list[dict], seed: int, train_fraction: float = 0.7,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Naive shuffle by relation (may leak if duplicate edges exist)."""
    by_rel: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_rel[rec["edges"][0]["relation_type"]].append(rec)

    train: list[dict] = []
    test: list[dict] = []
    for rel, batch in sorted(by_rel.items()):
        rng = random.Random(_relation_seed(seed, rel))
        shuffled = batch.copy()
        rng.shuffle(shuffled)
        n_train = int(round(len(shuffled) * train_fraction))
        train.extend(shuffled[:n_train])
        test.extend(shuffled[n_train:])
        print(f"  {rel}: {n_train:,} train + {len(shuffled) - n_train:,} test")
    return train, test, []


def relation_counts(records: list[dict]) -> dict[str, int]:
    c: Counter[str] = Counter()
    for rec in records:
        c[rec["edges"][0]["relation_type"]] += 1
    return dict(c.most_common())


def apply_styled(
    split_dir: Path,
    qa_dir: Path,
    train: list[dict],
    test: list[dict],
    dropped: list[dict],
    style: str,
) -> None:
    styled_path = qa_dir / f"qa_1_hop_{style}.jsonl"
    if not styled_path.exists():
        print(f"Skipping {styled_path.name} (not found)")
        return
    by_qaid = {rec["qaid"]: rec for rec in (json.loads(l) for l in styled_path.open())}
    train_s = [by_qaid[r["qaid"]] for r in train if r["qaid"] in by_qaid]
    test_s = [by_qaid[r["qaid"]] for r in test if r["qaid"] in by_qaid]
    dropped_s = [by_qaid[r["qaid"]] for r in dropped if r["qaid"] in by_qaid]
    print(f"Splitting qa_1_hop_{style}.jsonl:")
    # write_split hardcodes qa_2_hop prefix — write locally
    _write_1hop(split_dir, train_s, test_s, dropped_s, suffix=f"_{style}")


def _write_1hop(
    split_dir: Path,
    train: list[dict],
    test: list[dict],
    dropped: list[dict],
    suffix: str = "",
) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    fname = f"qa_1_hop{suffix}.jsonl"
    for subset, name in ((train, "train"), (test, "test"), (dropped, "dropped")):
        if not subset and name == "dropped":
            continue
        out_path = split_dir / fname.replace(".jsonl", f"_{name}.jsonl")
        with out_path.open("w") as out:
            for rec in subset:
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Wrote {len(subset):,} records to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Edge-disjoint train/test split for 1-hop QA.",
    )
    parser.add_argument("--dir", default="syndata_archive/data/bio/run5")
    parser.add_argument(
        "--qa-subdir",
        default=DEFAULT_QA_SUBDIR,
        help=f"QA folder under run dir (default: {DEFAULT_QA_SUBDIR}).",
    )
    parser.add_argument(
        "--split-subdir",
        default="qa_split",
        help="Output subfolder under qa-subdir (default: qa_split).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument(
        "--random-split",
        action="store_true",
        help="Naive shuffle split (default is edge-disjoint).",
    )
    parser.add_argument(
        "--short",
        action="store_true",
        help="Only first 200 base records.",
    )
    args = parser.parse_args()

    run_dir = Path(args.dir)
    qa_dir = run_dir / args.qa_subdir
    split_dir = qa_dir / args.split_subdir
    base_path = qa_dir / "qa_1_hop.jsonl"
    if not base_path.exists():
        raise FileNotFoundError(base_path)

    records = [json.loads(l) for l in base_path.open()]
    if args.short:
        records = records[:200]
        print(f"Short mode: splitting first {len(records)} records")

    # Ensure edges present
    missing = sum(1 for r in records if not r.get("edges"))
    if missing:
        raise RuntimeError(f"{missing} records missing edges in {base_path}")

    mode = "random" if args.random_split else "edge_disjoint"
    print(
        f"Splitting {len(records):,} 1-hop records "
        f"(mode={mode}, train_fraction={args.train_fraction})"
    )

    if args.random_split:
        train, test, dropped = split_1hop_random(
            records, args.seed, args.train_fraction,
        )
    else:
        train, test, dropped = split_1hop_edge_disjoint(
            records, args.seed, args.train_fraction,
        )

    overlap = verify_edge_disjoint(train, test)
    kept = len(train) + len(test)
    report = {
        "hop": 1,
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
        "by_relation": {
            "train": relation_counts(train),
            "test": relation_counts(test),
            "dropped": relation_counts(dropped),
        },
    }

    print(
        f"\nSummary: {len(train):,} train + {len(test):,} test "
        f"+ {len(dropped):,} dropped"
    )
    print(f"  actual train fraction: {report['actual_train_fraction']:.1%}")
    print(f"  edge overlap train∩test: {overlap} (expect 0)")
    print(f"  relations: {len(report['by_relation']['train'])} in train, "
          f"{len(report['by_relation']['test'])} in test")

    if overlap:
        raise RuntimeError(f"Edge overlap: {overlap} triplets")

    _write_1hop(split_dir, train, test, dropped, suffix="")
    for style in ("direct", "sentence", "reasoning"):
        apply_styled(split_dir, qa_dir, train, test, dropped, style)

    report_path = split_dir / "split_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nReport → {report_path}")


if __name__ == "__main__":
    main()

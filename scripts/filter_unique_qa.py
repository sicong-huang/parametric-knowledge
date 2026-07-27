#!/usr/bin/env python3
"""Drop QA examples whose answer is not unique against the ground-truth graph.

A question is ambiguous when the entity it describes resolves to more than one
answer in ``data/bio/edges.jsonl``. This happens mostly for the 2-hop
``NP<-P->NP`` pattern ("the person who <relation> X"), where several people may
share the same relation to X.

Answer set per row, computed from the full graph:
  1-hop (S,rel,T)          -> out[S, rel]
  P,P->NP  (shared target) -> out[S1,r1] & out[S2,r2]
  NP<-P->NP (shared source)-> union of out[bridge, r_ans] over in[anchor, r_a]
  P->P->NP  (chain)        -> union of out[bridge, r2] over out[P0, r1]
A row is kept iff its answer set has exactly one element.

Reads a qa/ dir and writes the same structure, minus the ambiguous rows, to
--output.

Usage:
    uv run scripts/filter_unique_qa.py --output data/qa_unique
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def build_index(edges_path: Path):
    """out[(source,rel)] -> {targets}, inc[(target,rel)] -> {sources}."""
    out = defaultdict(set)
    inc = defaultdict(set)
    with open(edges_path) as f:
        for line in f:
            e = json.loads(line)
            out[(e["source_id"], e["relation_type"])].add(e["target_id"])
            inc[(e["target_id"], e["relation_type"])].add(e["source_id"])
    return out, inc


def answer_ids(row: dict, out, inc) -> set:
    """All target_ids that satisfy the row's question given the full graph."""
    edges = row["edges"]
    if len(edges) == 1:
        e = edges[0]
        return out[(e["source_id"], e["relation_type"])]

    e0, e1 = edges
    if e0["target_id"] == e1["target_id"]:            # P,P->NP
        return out[(e0["source_id"], e0["relation_type"])] & out[(e1["source_id"], e1["relation_type"])]
    if e0["source_id"] == e1["source_id"]:            # NP<-P->NP
        ans, anc = (e0, e1) if e0["target_name"] == row["answer"] else (e1, e0)
        bridges = inc[(anc["target_id"], anc["relation_type"])]
        return {t for b in bridges for t in out[(b, ans["relation_type"])]}
    # chain P->P->NP: anchor.target == bridge == answer_edge.source
    a, b = (e0, e1) if e0["target_id"] == e1["source_id"] else (e1, e0)
    bridges = out[(a["source_id"], a["relation_type"])]
    return {t for br in bridges for t in out[(br, b["relation_type"])]}


def main():
    parser = argparse.ArgumentParser(description="Drop non-unique-answer QA examples")
    parser.add_argument("--qa-dir", type=Path, default=Path("data/qa_ambiguous"))
    parser.add_argument("--output", type=Path, required=True, help="output qa/ dir to write filtered splits into")
    parser.add_argument("--edges", type=Path, default=Path("data/bio/edges.jsonl"))
    args = parser.parse_args()

    print(f"Building edge index from {args.edges}")
    out, inc = build_index(args.edges)

    files = sorted(args.qa_dir.glob("*/qa_*.jsonl"))
    if not files:
        raise SystemExit(f"No */qa_*.jsonl files under {args.qa_dir}")

    dropped_total = kept_total = 0
    for path in files:
        rows = [json.loads(l) for l in open(path)]
        kept = [r for r in rows if len(answer_ids(r, out, inc)) == 1]
        n_drop = len(rows) - len(kept)
        dropped_total += n_drop
        kept_total += len(kept)
        rel = path.relative_to(args.qa_dir)
        dst = args.output / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(dst, "w") as f:
            for r in kept:
                f.write(json.dumps(r) + "\n")
        print(f"{rel}: {len(rows)} -> {len(kept)} (dropped {n_drop}, {n_drop / len(rows):.1%})")

    print(f"\nWrote filtered qa/ to {args.output}. Total: kept {kept_total}, dropped {dropped_total}")


if __name__ == "__main__":
    main()

"""
Validate stratified 2-hop QA datasets.

Usage:
    uv run syndata_archive/stratified_2hop/validate_qa.py \\
        --dir syndata_archive/data/bio/run5 --short

    uv run syndata_archive/stratified_2hop/validate_qa.py \\
        --dir syndata_archive/data/bio/run5 --expect-per-pattern 20000
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    DEFAULT_QA_SUBDIR,
    PATTERNS,
    check_question_vs_graph,
    iter_jsonl,
    load_graph,
    path_key,
    pattern_matches_geometry,
    qa_dir_for,
    resolve_answer_from_edges,
)


def normalize_question(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())


def validate(
    run_dir: Path,
    qa_subdir: str,
    expect_total: int | None,
    expect_per_pattern: int | None,
    check_styled: bool,
    fail_on_question: bool = True,
    write_question_details: bool = True,
) -> tuple[list[str], list[str]]:
    qa_dir = qa_dir_for(run_dir, qa_subdir)
    base_path = qa_dir / "qa_2_hop.jsonl"
    if not base_path.exists():
        raise FileNotFoundError(f"{base_path} not found")

    id_to_name, id_to_type, *_ = load_graph(run_dir)
    records = list(iter_jsonl(base_path))

    errors: list[str] = []
    warnings: list[str] = []

    # Total count
    if expect_total is not None and len(records) != expect_total:
        errors.append(f"Total records: {len(records):,} (expected {expect_total:,})")
    else:
        print(f"Total records: {len(records):,}  OK")

    # Per-pattern counts
    by_pattern = Counter(rec.get("pattern") for rec in records)
    for pattern in PATTERNS:
        count = by_pattern.get(pattern, 0)
        if expect_per_pattern is not None and count != expect_per_pattern:
            errors.append(f"{pattern}: {count:,} (expected {expect_per_pattern:,})")
        else:
            print(f"  {pattern}: {count:,}  OK")

    missing_meta = sum(
        1 for rec in records
        if not rec.get("pattern") or not rec.get("edges") or len(rec["edges"]) != 2
    )
    if missing_meta:
        errors.append(f"Missing pattern/edges: {missing_meta}")
    else:
        print("Required fields (pattern + 2 edges): 100%  OK")

    # path_key uniqueness
    keys = [path_key(rec) for rec in records if rec.get("edges")]
    dup_paths = len(keys) - len(set(keys))
    if dup_paths:
        errors.append(f"Duplicate path_key: {dup_paths}")
    else:
        print("Duplicate path_key: 0  OK")

    # Answer vs graph
    answer_mismatches = 0
    for rec in records:
        if not rec.get("edges"):
            continue
        expected = resolve_answer_from_edges(rec, id_to_name)
        if rec.get("answer") != expected:
            answer_mismatches += 1
    if answer_mismatches:
        errors.append(f"Answer vs graph mismatches: {answer_mismatches}")
    else:
        print("Answer vs graph: 0 mismatches  OK")

    # Pattern geometry
    geometry_mismatches = 0
    for rec in records:
        if not rec.get("edges"):
            continue
        if not pattern_matches_geometry(rec, id_to_type):
            geometry_mismatches += 1
    if geometry_mismatches:
        errors.append(f"Pattern geometry mismatches: {geometry_mismatches}")
    else:
        print("Pattern geometry: 0 mismatches  OK")

    # Duplicate questions (warning only)
    questions = [normalize_question(rec["question"]) for rec in records if rec.get("question")]
    dup_q = len(questions) - len(set(questions))
    warnings.append(f"Duplicate normalized questions: {dup_q}")
    print(f"Duplicate normalized questions: {dup_q}  (warning)")

    # Question NL vs graph (entity name / hide rules)
    q_hard_fail = 0
    q_soft_warn = 0
    issue_counts: Counter = Counter()
    detail_rows: list[dict] = []
    for rec in records:
        if not rec.get("edges") or not rec.get("pattern"):
            continue
        result = check_question_vs_graph(rec)
        if result["errors"]:
            q_hard_fail += 1
            for iss in result["errors"]:
                issue_counts[iss.split(":")[0]] += 1
        if result["warnings"]:
            q_soft_warn += 1
            for iss in result["warnings"]:
                issue_counts[f"warn:{iss.split(':')[0]}"] += 1
        if result["errors"] or result["warnings"]:
            detail_rows.append({
                "qaid": rec.get("qaid"),
                "pattern": rec.get("pattern"),
                "question": rec.get("question"),
                "answer": rec.get("answer"),
                "errors": result["errors"],
                "warnings": result["warnings"],
            })

    checked = sum(1 for r in records if r.get("edges") and r.get("pattern"))
    print(
        f"Question vs graph: {checked - q_hard_fail:,}/{checked:,} hard-OK "
        f"({q_hard_fail:,} errors, {q_soft_warn:,} with soft warnings)"
    )
    if issue_counts:
        top = ", ".join(f"{k}={v}" for k, v in issue_counts.most_common(8))
        print(f"  top question issues: {top}")
    if q_hard_fail and fail_on_question:
        errors.append(f"Question vs graph hard failures: {q_hard_fail}")
    elif q_hard_fail:
        warnings.append(f"Question vs graph hard failures (not failing): {q_hard_fail}")
    if q_soft_warn:
        warnings.append(f"Question vs graph soft warnings (missing clue name): {q_soft_warn}")

    if write_question_details and detail_rows:
        details_path = qa_dir / "question_graph_issues.jsonl"
        with details_path.open("w") as out:
            for row in detail_rows:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  wrote {len(detail_rows):,} issue rows → {details_path}")

    if check_styled:
        base_qaids = {rec["qaid"] for rec in records}
        for style in ("direct", "sentence", "reasoning"):
            styled_path = qa_dir / f"qa_2_hop_{style}.jsonl"
            if not styled_path.exists():
                warnings.append(f"Missing styled file: {styled_path.name}")
                continue
            styled = list(iter_jsonl(styled_path))
            styled_qaids = {rec["qaid"] for rec in styled}
            if styled_qaids != base_qaids:
                errors.append(
                    f"{styled_path.name}: qaid mismatch "
                    f"(base={len(base_qaids)}, styled={len(styled_qaids)})"
                )
            else:
                print(f"{styled_path.name}: qaid alignment  OK")

            missing_meta_styled = sum(
                1 for rec in styled
                if style != "direct" and (not rec.get("pattern") or not rec.get("edges"))
            )
            if missing_meta_styled:
                warnings.append(f"{styled_path.name}: {missing_meta_styled} records missing pattern/edges")

    return errors, warnings


def main():
    parser = argparse.ArgumentParser(description="Validate stratified 2-hop QA datasets.")
    parser.add_argument("--dir", default="syndata_archive/data/bio/run5")
    parser.add_argument("--qa-subdir", default=DEFAULT_QA_SUBDIR)
    parser.add_argument("--expect-total", type=int, default=None)
    parser.add_argument("--expect-per-pattern", type=int, default=None)
    parser.add_argument(
        "--short",
        action="store_true",
        help="Expect smoke-test sizes (201 total, 67 per pattern).",
    )
    parser.add_argument(
        "--check-styled",
        action="store_true",
        default=True,
        help="Verify styled full-answer files align by qaid (default: on).",
    )
    parser.add_argument(
        "--no-check-styled",
        action="store_false",
        dest="check_styled",
        help="Skip styled file checks.",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Write validation report to this path (default: <qa_subdir>/validation_report.txt).",
    )
    parser.add_argument(
        "--fail-on-question",
        action="store_true",
        default=True,
        help="Fail if question NL violates graph naming rules (default: on).",
    )
    parser.add_argument(
        "--no-fail-on-question",
        action="store_false",
        dest="fail_on_question",
        help="Report question NL issues as warnings only.",
    )
    args = parser.parse_args()

    run_dir = Path(args.dir)
    qa_dir = qa_dir_for(run_dir, args.qa_subdir)

    expect_total = args.expect_total
    expect_per_pattern = args.expect_per_pattern
    if args.short:
        expect_total = expect_total or 201
        expect_per_pattern = expect_per_pattern or 67

    print(f"validate_qa — {qa_dir / 'qa_2_hop.jsonl'}\n")

    errors, warnings = validate(
        run_dir,
        args.qa_subdir,
        expect_total,
        expect_per_pattern,
        args.check_styled,
        fail_on_question=args.fail_on_question,
    )

    report_lines = []
    if errors:
        report_lines.append("ERRORS:")
        report_lines.extend(f"  - {e}" for e in errors)
    if warnings:
        report_lines.append("WARNINGS:")
        report_lines.extend(f"  - {w}" for w in warnings)

    status = "PASSED" if not errors else "FAILED"
    report_lines.append(status)
    report = "\n".join(report_lines)

    report_path = Path(args.report) if args.report else qa_dir / "validation_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report + "\n")
    print(f"\nReport written to {report_path}")
    print(status)

    if errors:
        for e in errors:
            print(f"  ERROR: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

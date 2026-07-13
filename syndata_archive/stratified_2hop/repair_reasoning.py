"""
Repair prompt-failing 2-hop reasoning traces (staging → recheck → approve merge).

Never writes qa_2_hop_reasoning.jsonl unless you pass ``apply --approve``.

Typical flow:
    # 1) After check_reasoning_grounding.py
    uv run syndata_archive/stratified_2hop/repair_reasoning.py select \\
        --dir syndata_archive/data/bio/run5

    # 2) Regenerate into the repair run dir only
    export QA_BASE_URL=http://localhost:8001/v1
    uv run syndata_archive/stratified_2hop/repair_reasoning.py regenerate \\
        --run-dir syndata_archive/data/bio/run5/stratified_2hop/repair/<ts> \\
        --max-workers 32 --max-retries 3

    # 3) Recheck regen only
    uv run syndata_archive/stratified_2hop/repair_reasoning.py recheck \\
        --run-dir syndata_archive/data/bio/run5/stratified_2hop/repair/<ts>

    # 4) ONLY after human approval:
    uv run syndata_archive/stratified_2hop/repair_reasoning.py apply \\
        --run-dir ... --approve --passed-only
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from check_reasoning_grounding import (  # noqa: E402
    check_record,
    summarize,
)
from common import (  # noqa: E402
    DEFAULT_QA_SUBDIR,
    iter_jsonl,
    make_openai_client,
    qa_dir_for,
    style_reasoning_2hop,
)
from create_qa import _load_wiki_bio_map  # noqa: E402

# Categories from check_reasoning_grounding top_prompt_issues / report.
DEFAULT_ISSUES = (
    "post_think_answer_mismatch",
    "answer_fact_not_in_middle_sentence",
    "source_leak",
    "sentence_count_out_of_band",
)


def _issue_key(issue: str) -> str:
    key = issue.split(":")[0] if ":" in issue else issue
    if key.startswith("sentence_count"):
        return "sentence_count_out_of_band"
    return key


def _matches_issues(prompt_issues: list[str], wanted: set[str]) -> list[str]:
    matched = []
    for raw in prompt_issues:
        key = _issue_key(raw)
        if key in wanted:
            matched.append(key)
    return sorted(set(matched))


def _load_entity_names(run_dir: Path) -> list[str]:
    names = [entity["name"] for entity in iter_jsonl(run_dir / "entities.jsonl")]
    return sorted(set(names), key=len, reverse=True)


def _load_id_to_name(run_dir: Path) -> dict[str, str]:
    return {e["id"]: e["name"] for e in iter_jsonl(run_dir / "entities.jsonl")}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_manifest(run_dir: Path) -> dict:
    path = run_dir / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path} — run 'select' first.")
    return json.loads(path.read_text())


def _write_manifest(run_dir: Path, manifest: dict) -> None:
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )


def _resolve_run_dir(args, qa_dir: Path) -> Path:
    if getattr(args, "run_dir", None):
        return Path(args.run_dir)
    # Latest under qa_dir/repair/
    repair_root = qa_dir / "repair"
    if not repair_root.is_dir():
        raise FileNotFoundError(
            f"No --run-dir and no {repair_root}. Run 'select' first."
        )
    candidates = sorted(
        [p for p in repair_root.iterdir() if p.is_dir() and (p / "manifest.json").exists()],
        key=lambda p: p.name,
    )
    if not candidates:
        raise FileNotFoundError(f"No repair runs under {repair_root}")
    return candidates[-1]


def cmd_select(args) -> None:
    run_dir = Path(args.dir)
    qa_dir = qa_dir_for(run_dir, args.qa_subdir)
    failures_path = Path(args.failures) if args.failures else qa_dir / "grounding_failures.jsonl"
    reasoning_path = qa_dir / "qa_2_hop_reasoning.jsonl"
    if not failures_path.exists():
        raise FileNotFoundError(
            f"{failures_path} not found — run check_reasoning_grounding.py first."
        )
    if not reasoning_path.exists():
        raise FileNotFoundError(reasoning_path)

    wanted = {s.strip() for s in args.issues.split(",") if s.strip()}
    if not wanted:
        raise ValueError("Empty --issues")

    selected_meta: dict[str, dict] = {}
    issue_counter: Counter[str] = Counter()
    for row in iter_jsonl(failures_path):
        issues = row.get("prompt", {}).get("prompt_issues", [])
        matched = _matches_issues(issues, wanted)
        if not matched:
            continue
        qaid = row["qaid"]
        issue_counter.update(matched)
        if qaid not in selected_meta:
            selected_meta[qaid] = {
                "qaid": qaid,
                "pattern": row.get("pattern"),
                "matched_issues": matched,
                "all_prompt_issues": issues,
            }
        else:
            # union matched issues if duplicate rows
            prev = set(selected_meta[qaid]["matched_issues"])
            selected_meta[qaid]["matched_issues"] = sorted(prev | set(matched))

    qaids = sorted(selected_meta)
    print(f"Failures file: {failures_path}")
    print(f"Wanted issues: {sorted(wanted)}")
    print(f"Unique qaids selected: {len(qaids):,}")
    print(f"Issue hit counts (non-unique): {dict(issue_counter)}")

    by_qaid = {
        rec["qaid"]: rec
        for rec in iter_jsonl(reasoning_path)
        if rec.get("qaid") in selected_meta
    }
    missing = [q for q in qaids if q not in by_qaid]
    if missing:
        raise RuntimeError(
            f"{len(missing)} selected qaids missing from {reasoning_path} "
            f"(e.g. {missing[:5]})"
        )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = qa_dir / "repair" / ts
    out_dir.mkdir(parents=True, exist_ok=False)

    before_rows = [by_qaid[q] for q in qaids]
    _write_jsonl(out_dir / "before.jsonl", before_rows)
    (out_dir / "selected_qaids.json").write_text(
        json.dumps([selected_meta[q] for q in qaids], indent=2, ensure_ascii=False) + "\n"
    )

    manifest = {
        "created_utc": ts,
        "run_dir": str(out_dir),
        "data_dir": str(run_dir),
        "qa_subdir": args.qa_subdir or DEFAULT_QA_SUBDIR,
        "failures_path": str(failures_path),
        "reasoning_path": str(reasoning_path),
        "wanted_issues": sorted(wanted),
        "n_selected": len(qaids),
        "issue_hit_counts": dict(issue_counter),
        "status": "selected",
        "applied": False,
    }
    _write_manifest(out_dir, manifest)
    print(f"Wrote staging run → {out_dir}")
    print("  before.jsonl, selected_qaids.json, manifest.json")
    print("Next: regenerate --run-dir", out_dir)


def _prompt_ok_for_targets(check_row: dict, wanted: set[str] | None) -> bool:
    """Pass if grounded and no remaining issues in *wanted* (or full prompt_ok)."""
    if check_row.get("outside"):
        return False
    if not wanted:
        return bool(check_row.get("prompt_ok"))
    remaining = _matches_issues(
        check_row.get("prompt", {}).get("prompt_issues", []),
        wanted,
    )
    return len(remaining) == 0


def cmd_regenerate(args) -> None:
    data_dir = Path(args.dir) if args.dir else None
    qa_dir = qa_dir_for(data_dir, args.qa_subdir) if data_dir else None
    run_path = _resolve_run_dir(args, qa_dir) if qa_dir else Path(args.run_dir)
    manifest = _read_manifest(run_path)
    data_dir = Path(manifest["data_dir"])
    qa_dir = qa_dir_for(data_dir, manifest.get("qa_subdir"))

    before_path = run_path / "before.jsonl"
    if not before_path.exists():
        raise FileNotFoundError(before_path)

    before_rows = list(iter_jsonl(before_path))
    # Prefer full QA records (same fields create_qa uses) when available.
    base_by_qaid = {
        rec["qaid"]: rec
        for rec in iter_jsonl(qa_dir / "qa_2_hop.jsonl")
        if rec.get("qaid")
    }
    records = []
    for old in before_rows:
        qaid = old["qaid"]
        base = base_by_qaid.get(qaid, old)
        # Ensure question/answer/edges/pattern present for generation.
        merged = dict(base)
        for k in ("question", "answer", "edges", "pattern", "qaid"):
            if k in old and k not in merged:
                merged[k] = old[k]
        records.append(merged)

    wanted = set(manifest.get("wanted_issues") or DEFAULT_ISSUES)
    print(f"Repair run: {run_path}")
    print(f"Regenerating {len(records):,} reasoning traces (max_retries={args.max_retries})")

    print("Loading wiki biographies + entity names…")
    bio_map = _load_wiki_bio_map(data_dir)
    id_to_name = _load_id_to_name(data_dir)
    names_longest_first = _load_entity_names(data_dir)
    print(f"Loaded {len(bio_map):,} bios, {len(id_to_name):,} entities")

    client = make_openai_client(args)

    def regen_one(rec: dict) -> dict:
        last = None
        last_check = None
        attempts = []
        for attempt in range(1, args.max_retries + 1):
            last = style_reasoning_2hop(
                client, args.model, rec, bio_map, id_to_name,
            )
            last_check = check_record(last, bio_map, names_longest_first)
            ok = _prompt_ok_for_targets(last_check, wanted)
            attempts.append(
                {
                    "attempt": attempt,
                    "prompt_ok": last_check["prompt_ok"],
                    "target_ok": ok,
                    "prompt_issues": last_check["prompt"]["prompt_issues"],
                    "post_think_answer": last_check.get("post_think_answer"),
                }
            )
            if ok:
                break
        assert last is not None and last_check is not None
        return {
            "record": last,
            "check": last_check,
            "attempts": attempts,
            "passed": _prompt_ok_for_targets(last_check, wanted),
        }

    results: list[dict | None] = [None] * len(records)
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_idx = {
            executor.submit(regen_one, rec): i for i, rec in enumerate(records)
        }
        for future in tqdm(
            as_completed(future_to_idx),
            total=len(records),
            desc="Repair regenerate",
        ):
            idx = future_to_idx[future]
            results[idx] = future.result()

    regen_rows = [r["record"] for r in results]  # type: ignore[index]
    passed_rows = [r["record"] for r in results if r["passed"]]  # type: ignore[index]
    failed_rows = [r["record"] for r in results if not r["passed"]]  # type: ignore[index]
    attempt_log = [
        {
            "qaid": records[i]["qaid"],
            "passed": results[i]["passed"],  # type: ignore[index]
            "attempts": results[i]["attempts"],  # type: ignore[index]
            "final_issues": results[i]["check"]["prompt"]["prompt_issues"],  # type: ignore[index]
        }
        for i in range(len(records))
    ]

    _write_jsonl(run_path / "regen.jsonl", regen_rows)
    _write_jsonl(run_path / "passed.jsonl", passed_rows)
    _write_jsonl(run_path / "failed_still.jsonl", failed_rows)
    (run_path / "attempt_log.json").write_text(
        json.dumps(attempt_log, indent=2, ensure_ascii=False) + "\n"
    )

    checks = [r["check"] for r in results]  # type: ignore[index]
    stats = summarize(checks)
    (run_path / "regen_check_summary.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n"
    )

    manifest["status"] = "regenerated"
    manifest["n_regen"] = len(regen_rows)
    manifest["n_passed"] = len(passed_rows)
    manifest["n_failed_still"] = len(failed_rows)
    manifest["max_retries"] = args.max_retries
    manifest["regen_prompt_ok"] = stats.get("prompt_ok")
    manifest["regen_top_prompt_issues"] = stats.get("top_prompt_issues")
    _write_manifest(run_path, manifest)

    print(f"Wrote regen.jsonl ({len(regen_rows):,})")
    print(f"  passed: {len(passed_rows):,}  still failing: {len(failed_rows):,}")
    if stats.get("top_prompt_issues"):
        print("  top remaining issues:")
        for issue, count in stats["top_prompt_issues"][:8]:
            print(f"    {count:4d}  {issue}")
    print("Next: recheck --run-dir", run_path)
    print("NOTE: qa_2_hop_reasoning.jsonl was NOT modified.")


def cmd_recheck(args) -> None:
    data_dir = Path(args.dir) if args.dir else None
    qa_dir = qa_dir_for(data_dir, args.qa_subdir) if data_dir else None
    run_path = _resolve_run_dir(args, qa_dir) if qa_dir else Path(args.run_dir)
    manifest = _read_manifest(run_path)
    data_dir = Path(manifest["data_dir"])

    regen_path = run_path / "regen.jsonl"
    if not regen_path.exists():
        raise FileNotFoundError(f"{regen_path} — run regenerate first.")

    records = list(iter_jsonl(regen_path))
    print(f"Rechecking {len(records):,} regen rows from {regen_path}")

    names_longest_first = _load_entity_names(data_dir)
    bio_map = _load_wiki_bio_map(data_dir)
    rows = [check_record(rec, bio_map, names_longest_first) for rec in records]
    stats = summarize(rows)

    wanted = set(manifest.get("wanted_issues") or DEFAULT_ISSUES)
    target_pass = [
        r for r in rows if _prompt_ok_for_targets(r, wanted) and not r.get("outside")
    ]
    target_fail = [
        r for r in rows if not _prompt_ok_for_targets(r, wanted) or r.get("outside")
    ]

    _write_jsonl(run_path / "recheck_details.jsonl", rows)
    _write_jsonl(
        run_path / "recheck_failures.jsonl",
        [r for r in rows if not r["prompt_ok"] or r["outside"]],
    )
    report = {
        "summary": stats,
        "wanted_issues": sorted(wanted),
        "n_target_pass": len(target_pass),
        "n_target_fail": len(target_fail),
        "target_fail_qaids": [r["qaid"] for r in target_fail],
    }
    (run_path / "recheck_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )

    # Refresh passed/failed from recheck (authoritative)
    by_qaid = {rec["qaid"]: rec for rec in records}
    _write_jsonl(
        run_path / "passed.jsonl",
        [by_qaid[r["qaid"]] for r in target_pass if r["qaid"] in by_qaid],
    )
    _write_jsonl(
        run_path / "failed_still.jsonl",
        [by_qaid[r["qaid"]] for r in target_fail if r["qaid"] in by_qaid],
    )

    manifest["status"] = "rechecked"
    manifest["n_passed"] = len(target_pass)
    manifest["n_failed_still"] = len(target_fail)
    manifest["recheck_prompt_ok"] = stats.get("prompt_ok")
    manifest["recheck_top_prompt_issues"] = stats.get("top_prompt_issues")
    _write_manifest(run_path, manifest)

    print(f"Prompt OK: {stats.get('prompt_ok')}/{stats.get('total')}")
    print(f"Target-issue clear: {len(target_pass):,} pass / {len(target_fail):,} fail")
    if stats.get("top_prompt_issues"):
        print("Top prompt issues after regen:")
        for issue, count in stats["top_prompt_issues"][:8]:
            print(f"  {count:4d}  {issue}")
    print(f"Report → {run_path / 'recheck_report.json'}")
    print("NOTE: qa_2_hop_reasoning.jsonl was NOT modified.")
    print("After review, merge with: apply --run-dir", run_path, "--approve --passed-only")

    if target_fail and args.fail_on_remaining:
        raise SystemExit(1)


def cmd_apply(args) -> None:
    if not args.approve:
        raise SystemExit(
            "Refusing to modify qa_2_hop_reasoning.jsonl without --approve.\n"
            "Review recheck_report.json, then re-run with --approve "
            "(recommended: --passed-only)."
        )

    data_dir = Path(args.dir) if args.dir else None
    qa_dir = qa_dir_for(data_dir, args.qa_subdir) if data_dir else None
    run_path = _resolve_run_dir(args, qa_dir) if qa_dir else Path(args.run_dir)
    manifest = _read_manifest(run_path)
    if manifest.get("applied"):
        raise SystemExit(f"This repair run was already applied: {run_path}")

    data_dir = Path(manifest["data_dir"])
    qa_dir = qa_dir_for(data_dir, manifest.get("qa_subdir"))
    reasoning_path = qa_dir / "qa_2_hop_reasoning.jsonl"

    src_name = "passed.jsonl" if args.passed_only else "regen.jsonl"
    src_path = run_path / src_name
    if not src_path.exists():
        raise FileNotFoundError(src_path)

    replacements = {rec["qaid"]: rec for rec in iter_jsonl(src_path)}
    if not replacements:
        raise SystemExit(f"No rows to apply in {src_path}")

    existing = list(iter_jsonl(reasoning_path))
    existing_ids = {rec["qaid"] for rec in existing}
    missing = sorted(set(replacements) - existing_ids)
    if missing:
        raise RuntimeError(
            f"{len(missing)} replacement qaids not in {reasoning_path} "
            f"(e.g. {missing[:5]})"
        )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = qa_dir / f"qa_2_hop_reasoning_pre_repair_{ts}.jsonl"
    shutil.copy2(reasoning_path, backup)
    print(f"Backup → {backup}")

    n_replaced = 0
    merged = []
    for rec in existing:
        qaid = rec.get("qaid")
        if qaid in replacements:
            merged.append(replacements[qaid])
            n_replaced += 1
        else:
            merged.append(rec)

    _write_jsonl(reasoning_path, merged)

    manifest["applied"] = True
    manifest["applied_utc"] = ts
    manifest["applied_source"] = src_name
    manifest["n_replaced"] = n_replaced
    manifest["backup_path"] = str(backup)
    manifest["status"] = "applied"
    _write_manifest(run_path, manifest)

    print(f"Replaced {n_replaced:,} rows in {reasoning_path}")
    print(f"Total rows now: {len(merged):,}")


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dir", default="syndata_archive/data/bio/run5")
    parser.add_argument("--qa-subdir", default=DEFAULT_QA_SUBDIR)
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Repair staging directory (default: latest under <qa>/repair/).",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage / regenerate / recheck / approve-merge reasoning repairs.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_sel = sub.add_parser("select", help="Select failing qaids into a repair run dir.")
    _add_common(p_sel)
    p_sel.add_argument(
        "--failures",
        default=None,
        help="Path to grounding_failures.jsonl (default: <qa>/grounding_failures.jsonl).",
    )
    p_sel.add_argument(
        "--issues",
        default=",".join(DEFAULT_ISSUES),
        help=f"Comma-separated issue keys (default: {','.join(DEFAULT_ISSUES)}).",
    )

    p_reg = sub.add_parser(
        "regenerate",
        help="Regenerate reasoning into repair run (does NOT touch main file).",
    )
    _add_common(p_reg)
    p_reg.add_argument("--model", default="gemma4")
    p_reg.add_argument("--qa-base-url", default=None)
    p_reg.add_argument("--qa-api-key", default=None)
    p_reg.add_argument("--max-workers", type=int, default=32)
    p_reg.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Attempts per qaid until target issues clear (default: 3).",
    )

    p_chk = sub.add_parser("recheck", help="Re-run grounding checks on regen.jsonl.")
    _add_common(p_chk)
    p_chk.add_argument(
        "--fail-on-remaining",
        action="store_true",
        help="Exit 1 if any target issues remain.",
    )

    p_app = sub.add_parser(
        "apply",
        help="Merge into qa_2_hop_reasoning.jsonl (requires --approve).",
    )
    _add_common(p_app)
    p_app.add_argument(
        "--approve",
        action="store_true",
        help="Required confirmation to write the main reasoning file.",
    )
    p_app.add_argument(
        "--passed-only",
        action="store_true",
        help="Only merge rows in passed.jsonl (recommended).",
    )

    args = parser.parse_args()
    if args.command == "select":
        cmd_select(args)
    elif args.command == "regenerate":
        cmd_regenerate(args)
    elif args.command == "recheck":
        cmd_recheck(args)
    elif args.command == "apply":
        cmd_apply(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()

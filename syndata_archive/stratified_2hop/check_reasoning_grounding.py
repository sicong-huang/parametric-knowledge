"""
Deterministic grounding + prompt-compliance check for 2-hop reasoning traces.

Checks:
  1. Entity grounding — KG names in thinking vs path edges ∪ person bios
  2. Step count — number of sentences in the thinking trace
  3. Prompt compliance — rules from the synthetic-generation system prompts
     (imported from create_qa.py so they stay in sync)

Usage:
    uv run syndata_archive/stratified_2hop/check_reasoning_grounding.py \\
        --dir syndata_archive/data/bio/run5

    uv run syndata_archive/stratified_2hop/check_reasoning_grounding.py \\
        --dir syndata_archive/data/bio/run5 --fail-on-outside --fail-on-prompt
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from create_qa import (  # noqa: E402
    _2HOP_REASONING_1P_INSTRUCTION,
    _2HOP_REASONING_2P_INSTRUCTION,
    _2HOP_REASONING_JOIN_INSTRUCTION,
    _get_person_ids_from_edges,
)
from common import (  # noqa: E402
    DEFAULT_QA_SUBDIR,
    PATTERNS,
    _load_wiki_bio_map,
    iter_jsonl,
    qa_dir_for,
)

# ── Generation system prompts (source of truth: create_qa.py) ────────────────
# Kept here for documentation / report embedding; verification uses the imports.
REASONING_SYSTEM_PROMPTS = {
    "1_person": _2HOP_REASONING_1P_INSTRUCTION,
    "2_person": _2HOP_REASONING_2P_INSTRUCTION,
    "join": _2HOP_REASONING_JOIN_INSTRUCTION,
}

# Sentence-count bands from the system prompts:
#   1-person paths: "write 3–5 sentences"
#   2-person paths: "write 4–6 sentences"
#   join (P,P->NP): "write 3–5 sentences"
SENTENCE_BAND = {
    "1_person": (3, 5),
    "2_person": (4, 6),
    "join": (3, 5),
}

SOURCE_LEAK_PHRASES = (
    "biography",
    "biographies",
    "document",
    "wikipedia",
    "according to",
    "the text",
    "the source",
    "background knowledge",
    "provided knowledge",
    "mentioned in the query",
    "mentioned in the context",
)

META_TASK_PHRASES = (
    "now i need to",
    "i need to find",
    "looking at",
    "the query",
)


def parse_thinking(full_answer: str) -> str:
    m = re.search(r"<think>\n(.*?)\n</think>", full_answer, re.S)
    return m.group(1).strip() if m else full_answer.strip()


def parse_post_think_answer(full_answer: str) -> str:
    """Direct answer text after </think> (the field that must match gold)."""
    m = re.search(r"</think>\s*(.*)\s*$", full_answer, re.S)
    if not m:
        return ""
    return m.group(1).strip()


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def path_variant(record: dict) -> str:
    """Match create_qa._generate_2hop_reasoning_answer branch."""
    if record.get("pattern") == "P,P->NP":
        return "join"
    n = len(_get_person_ids_from_edges(record["edges"]))
    return "2_person" if n >= 2 else "1_person"


def path_names(record: dict) -> set[str]:
    names: set[str] = set()
    for edge in record["edges"]:
        names.add(edge["source_name"])
        names.add(edge["target_name"])
    return names


def bridge_name(record: dict) -> str:
    e1 = record["edges"][0]
    if record["pattern"] == "NP<-P->NP":
        return e1["source_name"]
    if record["pattern"] == "P,P->NP":
        # Shared NP is the answer (not a hidden bridge).
        return e1["target_name"]
    return e1["target_name"]


def find_entity_mentions(text: str, names_longest_first: list[str]) -> list[str]:
    found: list[str] = []
    claimed = [False] * len(text)
    for name in names_longest_first:
        if len(name) < 3:
            continue
        start = 0
        while True:
            idx = text.find(name, start)
            if idx < 0:
                break
            if any(claimed[idx : idx + len(name)]):
                start = idx + 1
                continue
            for i in range(idx, idx + len(name)):
                claimed[i] = True
            found.append(name)
            start = idx + len(name)
    return list(dict.fromkeys(found))


def relation_tokens_hinted(thinking: str, edges: list[dict]) -> list[tuple[str, bool]]:
    tl = thinking.lower()
    hits: list[tuple[str, bool]] = []
    for edge in edges:
        rel = edge["relation_type"]
        tokens = [t for t in rel.split("_") if len(t) > 3]
        hit = any(t in tl for t in tokens) or rel.replace("_", " ") in tl
        hits.append((rel, hit))
    return hits


def check_prompt_compliance(
    record: dict,
    thinking: str,
    sentences: list[str],
    variant: str,
    post_think_answer: str = "",
) -> dict:
    """Verify thinking against the synthetic-generation system prompt rules."""
    issues: list[str] = []
    lo, hi = SENTENCE_BAND[variant]
    n_steps = len(sentences)
    answer = record.get("answer", "")
    question = record.get("question", "")
    tl = thinking.lower()

    # Rule: sentence / step count band
    sentence_count_ok = lo <= n_steps <= hi
    if not sentence_count_ok:
        issues.append(f"sentence_count={n_steps} not in [{lo},{hi}] for {variant}")

    # Rule: NEVER reference biography / document / text / source
    source_leaks = [p for p in SOURCE_LEAK_PHRASES if p in tl]
    if source_leaks:
        issues.append(f"source_leak:{','.join(source_leaks)}")

    # Rule: gold answer must appear as the direct answer AFTER </think>
    # (not required in the final thinking sentence).
    post_think_has_answer = bool(answer) and post_think_answer == answer
    if answer and not post_think_has_answer:
        issues.append(
            f"post_think_answer_mismatch:got={post_think_answer!r}"
        )

    # Informational only: whether thinking's last sentence names the answer
    final_has_answer = bool(answer) and n_steps > 0 and answer in sentences[-1]

    # Rule (1-person): answer-bearing fact MUST appear in the middle
    # (not the first or last sentence)
    answer_in_middle = False
    answer_in_first = False
    answer_in_last = False
    if answer and n_steps >= 3:
        locs = [i for i, s in enumerate(sentences) if answer in s]
        answer_in_first = 0 in locs
        answer_in_last = (n_steps - 1) in locs
        answer_in_middle = any(0 < i < n_steps - 1 for i in locs)
        if variant == "1_person" and not answer_in_middle:
            issues.append("answer_fact_not_in_middle_sentence")
    elif answer and n_steps > 0:
        answer_in_first = answer in sentences[0]
        answer_in_last = answer in sentences[-1]

    # Rule: Do NOT restate the question
    restates_question = False
    if question:
        q_norm = re.sub(r"\s+", " ", question.strip().lower())
        # strip trailing ? for softer match
        q_core = q_norm.rstrip("?").strip()
        if len(q_core) >= 20 and q_core in tl:
            restates_question = True
            issues.append("restates_question")

    # Soft signals (reported, not hard-fail by default)
    meta_hits = [p for p in META_TASK_PHRASES if p in tl]

    prompt_ok = (
        sentence_count_ok
        and not source_leaks
        and (post_think_has_answer if answer else True)
        and not restates_question
        and not (variant == "1_person" and n_steps >= 3 and not answer_in_middle)
    )

    return {
        "variant": variant,
        "system_prompt_key": variant,
        "sentence_band": [lo, hi],
        "num_steps": n_steps,
        "sentence_count_ok": sentence_count_ok,
        "source_leaks": source_leaks,
        "post_think_answer": post_think_answer,
        "post_think_has_answer": post_think_has_answer,
        "final_sentence_has_answer": final_has_answer,
        "answer_in_first_sentence": answer_in_first,
        "answer_in_middle_sentence": answer_in_middle,
        "answer_in_last_sentence": answer_in_last,
        "restates_question": restates_question,
        "meta_task_phrases": meta_hits,
        "prompt_issues": issues,
        "prompt_ok": prompt_ok,
    }


def check_record(
    record: dict,
    bio_map: dict[str, str],
    names_longest_first: list[str],
) -> dict:
    full_answer = record.get("full_answer", "")
    thinking = parse_thinking(full_answer)
    post_think_answer = parse_post_think_answer(full_answer)
    sentences = split_sentences(thinking)
    variant = path_variant(record)
    path = path_names(record)
    pids = _get_person_ids_from_edges(record["edges"])
    bio = "\n".join(bio_map.get(pid, "") for pid in pids)

    mentioned = find_entity_mentions(thinking, names_longest_first)
    on_path = [n for n in mentioned if n in path]
    in_bio_only = [n for n in mentioned if n not in path and n in bio]
    outside = [n for n in mentioned if n not in path and n not in bio]

    bridge = bridge_name(record)
    answer = record.get("answer", "")
    rel_hits = relation_tokens_hinted(thinking, record["edges"])
    prompt = check_prompt_compliance(
        record, thinking, sentences, variant, post_think_answer=post_think_answer,
    )

    return {
        "qaid": record["qaid"],
        "pattern": record.get("pattern"),
        "question": record.get("question", ""),
        "num_steps": prompt["num_steps"],
        "sentences": sentences,
        "path_names": sorted(path),
        "mentioned_kg_entities": mentioned,
        "on_path": on_path,
        "bio_only": in_bio_only,
        "outside": outside,
        "bridge": bridge,
        "bridge_named": bridge in thinking,
        "answer": answer,
        "answer_named": bool(answer) and answer in thinking,
        "post_think_answer": post_think_answer,
        "post_think_answer_ok": prompt["post_think_has_answer"],
        "relation_hints": [{"relation": r, "hinted": h} for r, h in rel_hits],
        "both_relations_hinted": all(h for _, h in rel_hits),
        "path_coherent": len(in_bio_only) == 0 and len(outside) == 0,
        "grounded": len(outside) == 0,
        "prompt": prompt,
        "prompt_ok": prompt["prompt_ok"],
        "thinking": thinking,
    }


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {}

    by_pattern: dict[str, list[dict]] = defaultdict(list)
    step_hist: Counter[int] = Counter()
    for row in rows:
        by_pattern[row.get("pattern", "?")].append(row)
        step_hist[row["num_steps"]] += 1

    outside_counter: Counter[str] = Counter()
    bio_only_counter: Counter[str] = Counter()
    prompt_issue_counter: Counter[str] = Counter()
    for row in rows:
        outside_counter.update(row["outside"])
        bio_only_counter.update(row["bio_only"])
        for issue in row["prompt"]["prompt_issues"]:
            # normalize issue keys
            key = issue.split(":")[0] if ":" in issue else issue
            if key.startswith("sentence_count"):
                key = "sentence_count_out_of_band"
            prompt_issue_counter[key] += 1

    return {
        "total": n,
        "grounded": sum(1 for r in rows if r["grounded"]),
        "grounded_rate": sum(1 for r in rows if r["grounded"]) / n,
        "with_outside_entities": sum(1 for r in rows if r["outside"]),
        "with_bio_extra": sum(1 for r in rows if r["bio_only"]),
        "path_coherent": sum(1 for r in rows if r["path_coherent"]),
        "path_coherent_rate": sum(1 for r in rows if r["path_coherent"]) / n,
        "prompt_ok": sum(1 for r in rows if r["prompt_ok"]),
        "prompt_ok_rate": sum(1 for r in rows if r["prompt_ok"]) / n,
        "avg_num_steps": sum(r["num_steps"] for r in rows) / n,
        "step_histogram": dict(sorted(step_hist.items())),
        "missing_bridge": sum(1 for r in rows if not r["bridge_named"]),
        "missing_answer": sum(1 for r in rows if not r["answer_named"]),
        "post_think_answer_mismatch": sum(
            1 for r in rows if not r.get("post_think_answer_ok", True)
        ),
        "both_relations_hinted": sum(1 for r in rows if r["both_relations_hinted"]),
        "top_outside_entities": outside_counter.most_common(20),
        "top_bio_extra_entities": bio_only_counter.most_common(20),
        "top_prompt_issues": prompt_issue_counter.most_common(20),
        "by_pattern": {
            pat: {
                "count": len(rs),
                "grounded_rate": sum(1 for r in rs if r["grounded"]) / len(rs),
                "bio_extra_rate": sum(1 for r in rs if r["bio_only"]) / len(rs),
                "path_coherent_rate": sum(1 for r in rs if r["path_coherent"]) / len(rs),
                "prompt_ok_rate": sum(1 for r in rs if r["prompt_ok"]) / len(rs),
                "avg_num_steps": sum(r["num_steps"] for r in rs) / len(rs),
                "missing_bridge": sum(1 for r in rs if not r["bridge_named"]),
            }
            for pat, rs in sorted(
                by_pattern.items(),
                key=lambda x: PATTERNS.index(x[0]) if x[0] in PATTERNS else 99,
            )
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check 2-hop reasoning grounding and generation-prompt compliance.",
    )
    parser.add_argument("--dir", default="syndata_archive/data/bio/run5")
    parser.add_argument("--qa-subdir", default=DEFAULT_QA_SUBDIR)
    parser.add_argument(
        "--input",
        default=None,
        help="Reasoning JSONL path (default: <qa_subdir>/qa_2_hop_reasoning.jsonl).",
    )
    parser.add_argument(
        "--fail-on-outside",
        action="store_true",
        help="Exit non-zero if any thinking cites a KG entity outside path+bios.",
    )
    parser.add_argument(
        "--fail-on-prompt",
        action="store_true",
        help="Exit non-zero if any record fails generation-prompt compliance.",
    )
    parser.add_argument(
        "--max-bio-extra",
        type=int,
        default=None,
        help="If set, exit non-zero when a record cites more than this many bio-only entities.",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    run_dir = Path(args.dir)
    qa_dir = qa_dir_for(run_dir, args.qa_subdir)
    input_path = Path(args.input) if args.input else qa_dir / "qa_2_hop_reasoning.jsonl"
    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} not found")

    print(f"Loading entities and biographies from {run_dir} …")
    names: list[str] = []
    for entity in iter_jsonl(run_dir / "entities.jsonl"):
        names.append(entity["name"])
    names_longest_first = sorted(set(names), key=len, reverse=True)

    bio_map = _load_wiki_bio_map(run_dir)
    print(f"Loaded {len(names_longest_first):,} entity names, {len(bio_map):,} wiki bios")

    records = list(iter_jsonl(input_path))
    if args.limit is not None:
        records = records[: args.limit]
    print(f"Checking {len(records):,} reasoning records from {input_path}")

    rows = [check_record(rec, bio_map, names_longest_first) for rec in records]
    stats = summarize(rows)

    details_path = qa_dir / "grounding_details.jsonl"
    with details_path.open("w") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Failures only — easier to inspect than scanning all details.
    fail_rows = [r for r in rows if not r["prompt_ok"] or r["outside"]]
    failures_path = qa_dir / "grounding_failures.jsonl"
    with failures_path.open("w") as out:
        for row in fail_rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    prompt_fail_examples = [
        {
            "qaid": r["qaid"],
            "pattern": r["pattern"],
            "question": r.get("question", ""),
            "answer": r.get("answer", ""),
            "post_think_answer": r.get("post_think_answer", ""),
            "num_steps": r["num_steps"],
            "variant": r["prompt"]["variant"],
            "sentence_band": r["prompt"]["sentence_band"],
            "prompt_issues": r["prompt"]["prompt_issues"],
            "thinking": r["thinking"][:400],
        }
        for r in rows
        if not r["prompt_ok"]
    ][:30]

    outside_examples = [
        {
            "qaid": r["qaid"],
            "pattern": r["pattern"],
            "question": r.get("question", ""),
            "outside": r["outside"],
            "path_names": r["path_names"],
            "thinking": r["thinking"][:400],
        }
        for r in rows
        if r["outside"]
    ][:50]

    bio_extra_examples = [
        {
            "qaid": r["qaid"],
            "pattern": r["pattern"],
            "bio_only": r["bio_only"],
            "path_names": r["path_names"],
            "thinking": r["thinking"][:400],
        }
        for r in rows
        if r["bio_only"]
    ][:20]

    report = {
        "summary": stats,
        "generation_system_prompts": REASONING_SYSTEM_PROMPTS,
        "sentence_bands": SENTENCE_BAND,
        "prompt_fail_examples": prompt_fail_examples,
        "outside_examples": outside_examples,
        "bio_extra_examples": bio_extra_examples,
        "definitions": {
            "num_steps": "Number of sentences in the <think> block",
            "on_path": "KG entity appears in the two graph edges",
            "bio_only": "KG entity not on path, but present in relevant person bios",
            "outside": "KG entity not on path and not in bios (true hallucination)",
            "path_coherent": "no bio_only and no outside entities (path-minimal)",
            "grounded": "no outside entities (may still have bio_only padding)",
            "prompt_ok": (
                "Passes generation system-prompt checks: sentence band, no source leaks, "
                "post-</think> answer matches gold, 1-person answer fact in middle, "
                "does not restate question"
            ),
        },
    }
    report_path = qa_dir / "grounding_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    print("\n=== Grounding + prompt summary ===")
    print(f"Total:                 {stats['total']:,}")
    print(f"Grounded (no outside): {stats['grounded']:,} ({stats['grounded_rate']:.1%})")
    print(f"With outside entities: {stats['with_outside_entities']:,}")
    print(f"With bio-extra:        {stats['with_bio_extra']:,}")
    print(f"Path-coherent:         {stats['path_coherent']:,} ({stats['path_coherent_rate']:.1%})")
    print(f"Prompt OK:             {stats['prompt_ok']:,} ({stats['prompt_ok_rate']:.1%})")
    print(f"Avg steps (sentences): {stats['avg_num_steps']:.2f}")
    print(f"Step histogram:        {stats['step_histogram']}")
    print(f"Missing bridge:        {stats['missing_bridge']:,}")
    print(f"Missing answer in CoT: {stats['missing_answer']:,}")
    print(f"Post-think mismatch:   {stats.get('post_think_answer_mismatch', 0):,}")
    print(f"Both relation hints:   {stats['both_relations_hinted']:,}")
    if stats["top_prompt_issues"]:
        print("\nTop prompt issues:")
        for issue, count in stats["top_prompt_issues"][:8]:
            print(f"  {count:3d}  {issue}")
    print("\nBy pattern:")
    for pat, info in stats["by_pattern"].items():
        print(
            f"  {pat:<12} grounded={info['grounded_rate']:.1%}  "
            f"bio_extra={info['bio_extra_rate']:.1%}  "
            f"path_coherent={info['path_coherent_rate']:.1%}  "
            f"prompt_ok={info['prompt_ok_rate']:.1%}  "
            f"avg_steps={info['avg_num_steps']:.1f}"
        )
    print(f"\nWrote {details_path}")
    print(f"Wrote {failures_path} ({len(fail_rows)} failures)")
    print(f"Wrote {report_path}")

    exit_code = 0
    if args.fail_on_outside and stats["with_outside_entities"]:
        print(
            f"\nFAIL: {stats['with_outside_entities']} records cite entities "
            "outside path edges and biographies.",
            file=sys.stderr,
        )
        exit_code = 1
    if args.fail_on_prompt and stats["prompt_ok"] < stats["total"]:
        n_fail = stats["total"] - stats["prompt_ok"]
        print(f"\nFAIL: {n_fail} records fail generation-prompt compliance.", file=sys.stderr)
        exit_code = 1
    if args.max_bio_extra is not None:
        over = [r for r in rows if len(r["bio_only"]) > args.max_bio_extra]
        if over:
            print(
                f"\nFAIL: {len(over)} records exceed --max-bio-extra={args.max_bio_extra}.",
                file=sys.stderr,
            )
            exit_code = 1
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()

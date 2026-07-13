"""Shared helpers for stratified 2-hop QA generation."""

from __future__ import annotations

import hashlib
import os
import random
import sys
from pathlib import Path

# Allow importing from syndata_archive/create_qa.py when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from create_qa import (  # noqa: E402
    _build_indexes,
    _enrich_edges,
    _enumerate_np_p_np,
    _enumerate_p_np_p,
    _enumerate_p_p_np,
    _generate_2hop_question,
    _generate_2hop_reasoning_answer,
    _generate_2hop_sentence_answer,
    _is_person,
    _load_wiki_bio_map,
    iter_jsonl,
)

DEFAULT_QA_SUBDIR = "stratified_2hop"
SHORT_PER_PATTERN = 67  # ~201 total across 3 patterns

PATTERNS = ("P,P->NP", "NP<-P->NP", "P->P->NP")


def pattern_seed(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}:{label}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def make_openai_client(args) -> "OpenAI":
    """Build OpenAI client for local vLLM or remote API.

    vLLM exposes an OpenAI-compatible HTTP API and does not require a real
    key, but the openai Python SDK insists on *some* api_key value — we
    default to \"EMPTY\" when none is configured.
    """
    from openai import OpenAI

    api_key = (
        getattr(args, "qa_api_key", None)
        or os.environ.get("QA_API_KEY")
        or os.environ.get("BIO_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or "EMPTY"
    )
    base_url = (
        getattr(args, "qa_base_url", None)
        or os.environ.get("QA_BASE_URL")
        or os.environ.get("BIO_BASE_URL")
    )
    return OpenAI(base_url=base_url, api_key=api_key)


def qa_dir_for(run_dir: Path, qa_subdir: str | None = None) -> Path:
    return run_dir / (qa_subdir or DEFAULT_QA_SUBDIR)


def path_key(rec: dict) -> tuple:
    e1, e2 = rec["edges"]
    return (
        rec["pattern"],
        e1["source_id"],
        e1["relation_type"],
        e1["target_id"],
        e2["source_id"],
        e2["relation_type"],
        e2["target_id"],
    )


def dedup_paths(batch: list[dict]) -> list[dict]:
    seen: set[tuple] = set()
    out: list[dict] = []
    for rec in batch:
        key = path_key(rec)
        if key in seen:
            continue
        seen.add(key)
        out.append(rec)
    return out


def build_enumerators(
    outgoing,
    incoming,
    is_person,
    id_to_name,
) -> list[tuple[str, callable]]:
    return [
        ("P,P->NP", lambda: _enumerate_p_np_p(incoming, is_person, id_to_name)),
        ("NP<-P->NP", lambda: _enumerate_np_p_np(outgoing, is_person, id_to_name)),
        ("P->P->NP", lambda: _enumerate_p_p_np(outgoing, incoming, is_person, id_to_name)),
    ]


def effective_sample_per_pattern(args) -> int | None:
    if getattr(args, "short", False):
        return SHORT_PER_PATTERN
    return getattr(args, "sample_per_pattern", None)


def sample_stratified(
    enumerators: list[tuple[str, callable]],
    sample_per_pattern: int | None,
    seed: int,
) -> list[dict]:
    """Sample up to *sample_per_pattern* paths per pattern. If None, keep all."""
    records: list[dict] = []
    for label, fn in enumerators:
        batch = dedup_paths(fn())
        print(f"  {label:<15s}  {len(batch):,} paths (after dedup)")

        if sample_per_pattern is None:
            sampled = batch
            print(f"  {label}: kept all {len(sampled):,}")
        else:
            n = min(sample_per_pattern, len(batch))
            if len(batch) < sample_per_pattern:
                print(
                    f"WARNING: {label} only has {len(batch):,} paths, "
                    f"requested {sample_per_pattern:,}"
                )
            pattern_rng = random.Random(pattern_seed(seed, label))
            sampled = pattern_rng.sample(batch, n)
            print(f"  {label}: sampled {n:,} / {len(batch):,}")

        records.extend(sampled)

    print(f"Total stratified paths: {len(records):,}")
    return records


def _require_run_files(run_dir: Path) -> None:
    required = ("entities.jsonl", "edges.jsonl")
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        listed = "\n".join(f"  - {run_dir / name}" for name in missing)
        raise FileNotFoundError(
            f"Run directory is missing required graph files:\n{listed}\n\n"
            "The stratified 2-hop pipeline reads an existing graph run "
            "(entities.jsonl + edges.jsonl). Copy your run5 data into this "
            "directory, or point --dir at the location where those files live.\n\n"
            "Expected layout:\n"
            f"  {run_dir}/entities.jsonl\n"
            f"  {run_dir}/edges.jsonl\n"
            f"  {run_dir}/biographies.jsonl   (needed for sentence/reasoning styles)"
        )


def load_graph(run_dir: Path):
    run_dir = Path(run_dir)
    _require_run_files(run_dir)

    id_to_name: dict[str, str] = {}
    id_to_type: dict[str, str] = {}
    for entity in iter_jsonl(run_dir / "entities.jsonl"):
        id_to_name[entity["id"]] = entity["name"]
        id_to_type[entity["id"]] = entity["entity_type"]

    edges = list(iter_jsonl(run_dir / "edges.jsonl"))
    _enrich_edges(edges, id_to_name)
    outgoing, incoming, is_person = _build_indexes(edges, id_to_type)
    return id_to_name, id_to_type, edges, outgoing, incoming, is_person


def style_direct_2hop(record: dict) -> dict:
    return {
        "qaid": record["qaid"],
        "question": record["question"],
        "full_answer": record["answer"],
        "answer": record["answer"],
        "pattern": record.get("pattern"),
        "edges": record.get("edges"),
    }


def style_sentence_2hop(
    client, model: str, record: dict, bio_map: dict[str, str], id_to_name: dict[str, str],
) -> dict:
    result = _generate_2hop_sentence_answer(client, model, record, bio_map, id_to_name)
    result["pattern"] = record.get("pattern")
    result["edges"] = record.get("edges")
    return result


def style_reasoning_2hop(
    client, model: str, record: dict, bio_map: dict[str, str], id_to_name: dict[str, str],
) -> dict:
    result = _generate_2hop_reasoning_answer(client, model, record, bio_map, id_to_name)
    result["pattern"] = record.get("pattern")
    result["edges"] = record.get("edges")
    return result


def resolve_answer_from_edges(record: dict, id_to_name: dict[str, str]) -> str:
    """Return the graph answer name implied by pattern + edges."""
    pattern = record["pattern"]
    e1, e2 = record["edges"]
    if pattern == "P,P->NP":
        # Shared non-person (join target).
        answer_id = e1["target_id"]
    elif pattern in ("NP<-P->NP", "P->P->NP"):
        answer_id = e2["target_id"]
    else:
        raise ValueError(f"Unknown pattern: {pattern}")
    return id_to_name.get(answer_id, answer_id)


def pattern_matches_geometry(
    record: dict,
    id_to_type: dict[str, str],
) -> bool:
    pattern = record["pattern"]
    e1, e2 = record["edges"]

    def edge_person(edge: dict, end: str) -> bool:
        if end == "source":
            t = edge.get("source_type") or id_to_type.get(edge["source_id"], "")
        else:
            t = edge.get("target_type") or id_to_type.get(edge["target_id"], "")
        return _is_person(t)

    if pattern == "P,P->NP":
        shared_type = e1.get("target_type") or id_to_type.get(e1["target_id"], "")
        return (
            edge_person(e1, "source")
            and edge_person(e2, "source")
            and e1["target_id"] == e2["target_id"]
            and not _is_person(shared_type)
        )
    if pattern == "NP<-P->NP":
        return (
            edge_person(e1, "source")
            and edge_person(e2, "source")
            and e1["source_id"] == e2["source_id"]
            and not edge_person(e1, "target")
            and not edge_person(e2, "target")
        )
    if pattern == "P->P->NP":
        return (
            edge_person(e1, "source")
            and edge_person(e1, "target")
            and edge_person(e2, "source")
            and e1["target_id"] == e2["source_id"]
            and not edge_person(e2, "target")
        )
    return False


def question_roles(record: dict) -> dict[str, list[str]]:
    """Entity roles the question text must name vs hide, derived from edges.

    Returns:
      must_name_hard: missing → error (join: both people)
      must_name_soft: missing → warning (classic 2-hop clue entity)
      must_hide: present → error (answer and/or hidden bridge)
    """
    pattern = record["pattern"]
    e1, e2 = record["edges"]
    answer = record.get("answer") or ""

    if pattern == "P,P->NP":
        return {
            "must_name_hard": [e1["source_name"], e2["source_name"]],
            "must_name_soft": [],
            "must_hide": [n for n in (answer,) if n],
        }
    if pattern == "NP<-P->NP":
        # Clue NP should appear; bridge person + answer must stay hidden.
        return {
            "must_name_hard": [],
            "must_name_soft": [e1["target_name"]],
            "must_hide": [n for n in (e1["source_name"], answer) if n],
        }
    if pattern == "P->P->NP":
        # Clue person should appear; bridge person + answer must stay hidden.
        return {
            "must_name_hard": [],
            "must_name_soft": [e1["source_name"]],
            "must_hide": [n for n in (e1["target_name"], answer) if n],
        }
    raise ValueError(f"Unknown pattern: {pattern}")


def _name_in_text(name: str, text: str) -> bool:
    """Exact substring match; skip very short names to avoid false positives."""
    if not name or len(name) < 3:
        return False
    return name in text


def check_question_vs_graph(record: dict) -> dict:
    """Check that question NL matches graph-derived naming rules.

    Returns a dict with:
      ok: no hard errors
      errors: hard failures (answer/bridge leak, missing required names)
      warnings: soft failures (missing clue entity on classic 2-hop)
      roles: the role lists used
    """
    question = record.get("question") or ""
    if not record.get("edges") or not record.get("pattern"):
        return {
            "ok": False,
            "errors": ["missing_pattern_or_edges"],
            "warnings": [],
            "roles": {},
        }
    if not question.strip():
        return {
            "ok": False,
            "errors": ["empty_question"],
            "warnings": [],
            "roles": {},
        }

    roles = question_roles(record)
    errors: list[str] = []
    warnings: list[str] = []

    for name in roles["must_hide"]:
        if _name_in_text(name, question):
            if name == record.get("answer"):
                errors.append(f"answer_leaked:{name}")
            else:
                errors.append(f"bridge_leaked:{name}")

    for name in roles["must_name_hard"]:
        if not _name_in_text(name, question):
            errors.append(f"missing_required_name:{name}")

    for name in roles["must_name_soft"]:
        if not _name_in_text(name, question):
            warnings.append(f"missing_clue_name:{name}")

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "roles": roles,
    }

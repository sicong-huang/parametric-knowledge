"""Create an uneven-frequency biography dataset from existing biographies.

Reads biographies.jsonl (uniform N bios per person), assigns each person to a
frequency bucket, then subsamples or generates additional biographies so each
person ends up with exactly their bucket's target count.

Outputs:
    biographies_uneven.jsonl  – same schema as biographies.jsonl
    bio_freq_uneven.jsonl     – per-person frequency manifest
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from random import Random
from typing import Any

from openai import OpenAI
from tqdm import tqdm

from bio_pipeline.biographies import (
    STYLE_PROMPTS,
    format_edge_as_triple,
    index_relations,
    render_all_biographies_for_person,
)
from bio_pipeline.io import read_jsonl, ensure_output_dir

FREQ_BUCKETS: list[tuple[int, int]] = [
    # (target_frequency, num_persons)
    (1, 2_000),
    (2, 2_000),
    (5, 1_500),
    (10, 1_500),
    (20, 1_500),
    (50, 500),
    (100, 400),
    (200, 300),
    (500, 200),
    (1000, 100),
]

GENERATION_BATCH_SIZE = 20
AVAILABLE_STYLES = list(STYLE_PROMPTS.keys())


def subsample_bios(
    bios: list[dict[str, Any]], target: int, rng: Random
) -> list[dict[str, Any]]:
    """Randomly keep *target* bios, guaranteeing at least one wiki-style bio."""
    if target >= len(bios):
        return list(bios)

    wiki_bios = [b for b in bios if b["style"] == "wiki"]
    if wiki_bios:
        kept_wiki = rng.choice(wiki_bios)
        rest = [b for b in bios if b is not kept_wiki]
        sampled = rng.sample(rest, target - 1)
        return [kept_wiki] + sampled
    return rng.sample(bios, target)


def _assign_persons_to_buckets(
    bios_by_person: dict[str, list[dict[str, Any]]],
    seed: int,
) -> dict[str, int]:
    """Return {person_id: target_frequency} using stratified random assignment.

    Persons are grouped by entity_type so every bucket gets a proportional mix
    of subtypes.
    """
    rng = Random(seed)

    by_subtype: dict[str, list[str]] = defaultdict(list)
    for pid, bios in bios_by_person.items():
        subtype = bios[0]["entity_type"]
        by_subtype[subtype].append(pid)
    for pids in by_subtype.values():
        rng.shuffle(pids)

    total_persons_needed = sum(count for _, count in FREQ_BUCKETS)
    total_persons_available = len(bios_by_person)
    if total_persons_needed > total_persons_available:
        raise ValueError(
            f"FREQ_BUCKETS require {total_persons_needed} persons but only "
            f"{total_persons_available} are available"
        )

    subtypes = sorted(by_subtype.keys())
    subtype_sizes = {st: len(by_subtype[st]) for st in subtypes}
    total_available = sum(subtype_sizes.values())

    assignment: dict[str, int] = {}
    consumed: dict[str, int] = {st: 0 for st in subtypes}

    for freq, count in FREQ_BUCKETS:
        for st in subtypes:
            share = round(count * subtype_sizes[st] / total_available)
            share = min(share, len(by_subtype[st]) - consumed[st])
            for pid in by_subtype[st][consumed[st]: consumed[st] + share]:
                assignment[pid] = freq
            consumed[st] += share

        assigned_so_far = sum(1 for v in assignment.values() if v == freq)
        deficit = count - assigned_so_far
        if deficit > 0:
            pool = [
                pid
                for st in subtypes
                for pid in by_subtype[st][consumed[st]:]
                if pid not in assignment
            ]
            for pid in pool[:deficit]:
                assignment[pid] = freq
                st = bios_by_person[pid][0]["entity_type"]
                consumed[st] += 1

    return assignment


def _build_generation_context(
    output_dir: str,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, list[str]],
]:
    """Load entities/edges/relations and return (entities_by_id, triples_by_person)."""
    entities = read_jsonl(f"{output_dir}/entities.jsonl")
    edges = read_jsonl(f"{output_dir}/edges.jsonl")
    relations = read_jsonl(f"{output_dir}/relations.jsonl")

    entities_by_id: dict[str, dict[str, Any]] = {}
    person_ids: set[str] = set()
    for ent in entities:
        eid = str(ent["id"])
        if eid:
            entities_by_id[eid] = ent
        et = str(ent["entity_type"])
        if et == "Person" or et.startswith("Person."):
            person_ids.add(eid)

    relation_descriptions = index_relations(relations)

    edges_by_person: dict[str, list[dict[str, Any]]] = {pid: [] for pid in person_ids}
    for edge in edges:
        sid = str(edge["source_id"])
        tid = str(edge["target_id"])
        if sid in edges_by_person:
            edges_by_person[sid].append(edge)
        if tid in edges_by_person and tid != sid:
            edges_by_person[tid].append(edge)

    triples_by_person: dict[str, list[str]] = {}
    for pid in person_ids:
        person_edges = sorted(
            edges_by_person.get(pid, []),
            key=lambda e: 0 if str(e.get("relation_type", "")) == "occupation_is" else 1,
        )
        triples_by_person[pid] = [
            format_edge_as_triple(edge, entities_by_id, relation_descriptions)
            for edge in person_edges
        ]

    return entities_by_id, triples_by_person


def _make_generation_work_items(
    pid: str,
    existing_bios: list[dict[str, Any]],
    target: int,
    rng: Random,
) -> list[list[tuple[str, int]]]:
    """Split the extra bios needed into batched style specs.

    Returns a list of batches, each batch being a list of (style, count) tuples
    whose total equals <= GENERATION_BATCH_SIZE.
    """
    extra = target - len(existing_bios)
    if extra <= 0:
        return []

    batches: list[list[tuple[str, int]]] = []
    remaining = extra
    while remaining > 0:
        batch_size = min(remaining, GENERATION_BATCH_SIZE)
        chosen = [rng.choice(AVAILABLE_STYLES) for _ in range(batch_size)]
        counts = Counter(chosen)
        batches.append(sorted(counts.items()))
        remaining -= batch_size
    return batches


def run(
    output_dir: str,
    bio_model: str | None,
    bio_base_url: str | None,
    bio_api_key: str | None,
    max_workers: int,
    seed: int,
    limit: int | None = None,
) -> None:
    rng = Random(seed)
    t0 = time.time()

    if limit is not None:
        print(f"[TEST MODE] --limit {limit}: capping each bucket to {limit} person(s)")

    # --- Load existing biographies and group by person ---
    print("Loading existing biographies...")
    all_bios = read_jsonl(f"{output_dir}/biographies.jsonl")
    bios_by_person: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for bio in all_bios:
        bios_by_person[str(bio["id"])].append(bio)

    existing_per_person = len(next(iter(bios_by_person.values())))
    print(f"  {len(bios_by_person)} persons, {existing_per_person} bios each")

    # --- Assign persons to frequency buckets ---
    assignment = _assign_persons_to_buckets(bios_by_person, seed)

    # Apply per-bucket limit for quick test runs
    if limit is not None:
        bucket_persons: dict[int, list[str]] = defaultdict(list)
        for pid, freq in assignment.items():
            bucket_persons[freq].append(pid)
        assignment = {
            pid: freq
            for freq, pids in bucket_persons.items()
            for pid in pids[:limit]
        }

    needs_generation = any(freq > existing_per_person for freq in assignment.values())
    if needs_generation and not bio_model:
        raise ValueError(
            "Some frequency buckets exceed existing bio count — "
            "--bio-model is required for generation"
        )

    # --- Print bucket summary ---
    bucket_counts: dict[int, int] = Counter(assignment.values())
    print("\nFrequency bucket assignments:")
    for freq, count in sorted(bucket_counts.items()):
        source = "subsample" if freq < existing_per_person else (
            "keep" if freq == existing_per_person else "generate"
        )
        print(f"  freq={freq:>5d}: {count:>5d} persons  ({source})")

    # --- Subsample phase (in-memory, no LLM) ---
    print("\nSubsampling biographies...")
    output_records: list[dict[str, Any]] = []
    generation_persons: list[tuple[str, int]] = []

    for pid, freq in sorted(assignment.items(), key=lambda x: x[0]):
        existing = bios_by_person[pid]
        if freq <= len(existing):
            output_records.extend(subsample_bios(existing, freq, rng))
        else:
            output_records.extend(existing)
            generation_persons.append((pid, freq))

    print(f"  {len(output_records)} bios after subsample phase")

    # --- Generation phase (LLM calls) ---
    if generation_persons:
        print(f"\nGenerating extra biographies for {len(generation_persons)} persons...")

        resolved_api_key = (
            bio_api_key
            or os.environ.get("BIO_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        client = OpenAI(base_url=bio_base_url, api_key=resolved_api_key)
        entities_by_id, triples_by_person = _build_generation_context(output_dir)

        # Build flat list of (pid, batch_styles) work items
        work_items: list[tuple[str, list[tuple[str, int]]]] = []
        for pid, freq in generation_persons:
            batches = _make_generation_work_items(
                pid, bios_by_person[pid], freq, rng
            )
            for batch_styles in batches:
                work_items.append((pid, batch_styles))

        print(f"  {len(work_items)} generation batches queued")

        # Pre-compute max existing rendition per style per person for renumbering
        max_rendition: dict[str, dict[str, int]] = {}
        for pid, freq in generation_persons:
            style_max: dict[str, int] = {}
            for bio in bios_by_person[pid]:
                s = bio["style"]
                style_max[s] = max(style_max.get(s, -1), bio["rendition"])
            max_rendition[pid] = style_max

        # Thread-safe counter for rendition offsets (per person per style)
        import threading
        rendition_lock = threading.Lock()

        def process_batch(
            pid: str, batch_styles: list[tuple[str, int]]
        ) -> list[dict[str, Any]]:
            person_ent = entities_by_id.get(pid, {
                "id": pid, "name": pid, "entity_type": "Person"
            })
            triples = triples_by_person.get(pid, [])

            if not triples:
                name = str(person_ent.get("name", pid))
                et = str(person_ent.get("entity_type", "Person"))
                fallback = f"{name} is a {et}. No additional relationships are recorded."
                records = []
                for style, count in batch_styles:
                    for r in range(count):
                        with rendition_lock:
                            base = max_rendition[pid].get(style, -1) + 1
                            max_rendition[pid][style] = base + count - 1
                        records.append({
                            "id": pid,
                            "entity_type": person_ent.get("entity_type", "Person"),
                            "name": person_ent.get("name", pid),
                            "style": style,
                            "rendition": base + r,
                            "biography": fallback,
                            "triples": triples,
                        })
                return records

            bio_entries = render_all_biographies_for_person(
                client, bio_model, person_ent, triples, batch_styles
            )

            records = []
            with rendition_lock:
                for entry in bio_entries:
                    s = entry["style"]
                    base = max_rendition[pid].get(s, -1) + 1
                    entry["rendition"] = base
                    max_rendition[pid][s] = base
                    records.append({
                        "id": pid,
                        "entity_type": person_ent.get("entity_type", "Person"),
                        "name": person_ent.get("name", pid),
                        "style": entry["style"],
                        "rendition": entry["rendition"],
                        "biography": entry["biography"],
                        "triples": triples,
                    })
            return records

        generated_records: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
            futures = {
                executor.submit(process_batch, pid, batch_styles): (pid, batch_styles)
                for pid, batch_styles in work_items
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Generating biographies",
                unit="batch",
            ):
                try:
                    generated_records.extend(future.result())
                except Exception as e:
                    pid, _ = futures[future]
                    print(f"Error generating batch for {pid}: {e}")

        output_records.extend(generated_records)
        print(f"  {len(generated_records)} bios generated")

    # --- Sort output by person id for consistency ---
    output_records.sort(key=lambda r: (r["id"], r["style"], r["rendition"]))

    # --- Write biographies_uneven.jsonl ---
    bio_path = f"{output_dir}/biographies_uneven.jsonl"
    ensure_output_dir(bio_path)
    with open(bio_path, "w") as f:
        for record in output_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"\nBiographies saved to {bio_path}  ({len(output_records)} total)")

    # --- Write bio_freq_uneven.jsonl ---
    freq_path = f"{output_dir}/bio_freq_uneven.jsonl"
    freq_records = []
    for pid, freq in sorted(assignment.items()):
        sample_bio = bios_by_person[pid][0]
        freq_records.append({
            "id": pid,
            "name": sample_bio["name"],
            "entity_type": sample_bio["entity_type"],
            "frequency": freq,
        })
    with open(freq_path, "w") as f:
        for record in freq_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Frequency manifest saved to {freq_path}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")

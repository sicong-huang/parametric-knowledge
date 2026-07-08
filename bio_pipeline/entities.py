"""Step 1: Generate fictional entity names from entity type definitions."""

from __future__ import annotations

import json
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Any

from openai import OpenAI
from tqdm import tqdm

from bio_pipeline.io import ensure_output_dir, read_jsonl
from bio_pipeline.models import EntityNamesResponse


def generate_entity_id(entity_type: str, counter: int) -> str:
    category_map = {
        "Person": "PER",
        "Org": "ORG",
        "Work": "WRK",
        "Place": "PLC",
    }
    parts = entity_type.split(".")
    category = parts[0] if len(parts) > 0 else entity_type
    specific_type = parts[1] if len(parts) > 1 else category
    cat_abbr = category_map.get(category, category[:3].upper())
    type_abbr = (
        specific_type[:3].upper()
        if len(specific_type) >= 3
        else specific_type.upper()
    )
    return f"{cat_abbr}_{type_abbr}_{counter:05d}"


def generate_names_for_type(
    client: OpenAI,
    model: str,
    entity_type: str,
    count: int,
    existing_names: set[str],
    lock: Lock,
) -> list[str]:
    parts = entity_type.split(".")
    category = parts[0]
    specific_type = parts[1] if len(parts) > 1 else None
    is_person = category == "Person"

    if is_person:
        prompt = (
            f"Generate {count} completely random, unique, and fake person names.\n"
            "The names should be:\n"
            "- Completely random (no connection to the person's profession or role)\n"
            "- Unique and fake (no real-world names)\n"
            "- Diverse in style and origin\n"
            "- Each name should be a full name"
        )
    else:
        type_desc = specific_type or category
        prompt = (
            f"Generate {count} unique, fake entity names for {type_desc} type entities.\n"
            "The names should:\n"
            f"- Sometimes reflect the nature of {type_desc} (but not always - mix it up)\n"
            "- Be completely unique and fake (no real-world names)\n"
            "- Be diverse and creative\n"
            f"- Be appropriate for {type_desc} entities"
        )

    try:
        response = client.responses.parse(
            model=model,
            instructions="You are a helpful assistant that generates unique, fake entity names for a knowledge graph.",
            input=prompt,
            text_format=EntityNamesResponse,
        )
        names = response.output_parsed.names
    except Exception as e:
        print(f"Error generating names for {entity_type}: {e}")
        names = [f"{entity_type}_fallback_{i + 1}" for i in range(count)]

    unique_names: list[str] = []
    for name in names:
        name_str = str(name).strip()
        if not name_str:
            continue
        with lock:
            if name_str in existing_names:
                continue
            existing_names.add(name_str)
        unique_names.append(name_str)

    return unique_names[:count]


def run(
    output_dir: str,
    entity_types_path: str,
    model: str,
    base_url: str | None,
    max_workers: int,
    chunk_size: int,
) -> str:
    client = OpenAI(base_url=base_url)

    entity_types: list[dict[str, Any]] = []
    with open(entity_types_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                entity_types.append(json.loads(line))

    all_names: set[str] = set()
    names_lock = Lock()

    valid_entities: list[dict[str, Any]] = []
    for ed in entity_types:
        if ed.get("entity_type") and ed.get("count") is not None:
            valid_entities.append(ed)

    results_by_index: dict[int, list[str]] = {i: [] for i in range(len(valid_entities))}
    desired_counts: dict[int, int] = {
        i: int(ed["count"]) for i, ed in enumerate(valid_entities)
    }
    workers = min(max_workers, max(1, len(valid_entities)))

    def submit_chunk(
        executor: ThreadPoolExecutor,
        futures: dict[Any, int],
        idx: int,
    ) -> bool:
        ed = valid_entities[idx]
        et = str(ed["entity_type"])
        current = len(results_by_index[idx])
        target = desired_counts[idx]
        if current >= target:
            return False
        request_count = min(chunk_size, target - current)
        future = executor.submit(
            generate_names_for_type,
            client,
            model,
            et,
            request_count,
            all_names,
            names_lock,
        )
        futures[future] = idx
        return True

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures: dict[Any, int] = {}
        queue: deque[int] = deque(range(len(valid_entities)))
        completed: set[int] = set()

        while len(futures) < workers and queue:
            submit_chunk(executor, futures, queue.popleft())

        total_names = sum(desired_counts.values())
        with tqdm(total=total_names, desc="Generating entities", unit="name") as progress:
            while futures:
                for future in as_completed(list(futures)):
                    idx = futures.pop(future)
                    try:
                        chunk_names = future.result()
                    except Exception as e:
                        print(f"Error generating entities for index {idx}: {e}")
                        chunk_names = []

                    if chunk_names:
                        remaining = desired_counts[idx] - len(results_by_index[idx])
                        if remaining > 0:
                            add_count = min(len(chunk_names), remaining)
                            results_by_index[idx].extend(chunk_names[:add_count])
                            progress.update(add_count)
                        if len(results_by_index[idx]) > desired_counts[idx]:
                            results_by_index[idx] = results_by_index[idx][: desired_counts[idx]]

                    if len(results_by_index[idx]) >= desired_counts[idx]:
                        completed.add(idx)
                    else:
                        queue.append(idx)

                    while len(futures) < workers and queue:
                        submit_chunk(executor, futures, queue.popleft())

    output_path = f"{output_dir}/entities.jsonl"
    ensure_output_dir(output_path)
    entity_counters: dict[str, int] = {}
    with open(output_path, "w") as out_f:
        for idx, ed in enumerate(valid_entities):
            et = str(ed["entity_type"])
            names = results_by_index.get(idx, [])
            if et not in entity_counters:
                entity_counters[et] = 0
            for name in names:
                entity_counters[et] += 1
                eid = generate_entity_id(et, entity_counters[et])
                out_f.write(
                    json.dumps({"id": eid, "entity_type": et, "name": name}, ensure_ascii=False)
                    + "\n"
                )

    print(f"Entities saved to {output_path}")
    print(f"Total unique names generated: {len(all_names)}")
    return output_path

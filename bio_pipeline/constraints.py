"""Step 3: Generate cardinality constraints for each relation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from openai import OpenAI
from tqdm import tqdm

from bio_pipeline.io import read_jsonl, ensure_output_dir
from bio_pipeline.models import ConstraintsResponse

import json


def normalize_constraints(response: ConstraintsResponse) -> dict[str, int]:
    constraints: dict[str, int] = {}
    for key in ("min_per_source", "max_per_source", "min_per_target", "max_per_target"):
        value = getattr(response, key)
        if isinstance(value, int) and value > 0:
            constraints[key] = value

    min_src = constraints.get("min_per_source")
    max_src = constraints.get("max_per_source")
    if min_src is not None and max_src is not None and max_src < min_src:
        constraints["max_per_source"] = min_src

    min_tgt = constraints.get("min_per_target")
    max_tgt = constraints.get("max_per_target")
    if min_tgt is not None and max_tgt is not None and max_tgt < min_tgt:
        constraints["max_per_target"] = min_tgt

    return constraints


def generate_constraints_for_relation(
    client: OpenAI,
    model: str,
    relation: dict[str, Any],
) -> dict[str, int]:
    relation_type = str(relation.get("relation_type", "")).strip()
    source_type = str(relation.get("source_type", "")).strip()
    target_type = str(relation.get("target_type", "")).strip()
    description = str(relation.get("description", "")).strip()

    prompt = (
        "You are deciding cardinality constraints for a knowledge graph relation.\n"
        "Choose which constraints make sense for this relation. You may choose none.\n\n"
        "Constraints (positive integers):\n"
        "- min_per_source: minimum edges per source entity (e.g., everyone has at least 1 birthplace)\n"
        "- max_per_source: maximum edges per source entity (e.g., born_in should be at most 1)\n"
        "- min_per_target: minimum edges per target entity\n"
        "- max_per_target: maximum edges per target entity\n\n"
        "Use small realistic values (usually 1-5). If a constraint does not apply, omit it.\n\n"
        f'Relation: "{relation_type}"\n'
        f'Source type: "{source_type}"\n'
        f'Target type: "{target_type}"\n'
        f'Description: "{description}"\n'
    )

    response = client.responses.parse(
        model=model,
        instructions="Return structured JSON that matches the provided schema.",
        input=prompt,
        text_format=ConstraintsResponse,
    )
    return normalize_constraints(response.output_parsed)


def run(
    output_dir: str,
    model: str,
    base_url: str | None,
    max_workers: int,
) -> str:
    client = OpenAI(base_url=base_url)

    relations_path = f"{output_dir}/relations.jsonl"
    relations = read_jsonl(relations_path)

    results: list[tuple[int, dict[str, Any]]] = []
    workers = min(max_workers, max(1, len(relations)))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(generate_constraints_for_relation, client, model, rel): idx
            for idx, rel in enumerate(relations)
        }
        for future in tqdm(
            as_completed(future_map),
            total=len(future_map),
            desc="Generating constraints",
            unit="relation",
        ):
            idx = future_map[future]
            base = relations[idx]
            try:
                constraints = future.result()
            except Exception as e:
                print(f"Error generating constraints for relation {idx}: {e}")
                constraints = {}
            merged = dict(base)
            merged.update(constraints)
            results.append((idx, merged))

    results.sort(key=lambda item: item[0])

    output_path = f"{output_dir}/relations_w_constraints.jsonl"
    ensure_output_dir(output_path)
    with open(output_path, "w") as out_f:
        for _, rel in results:
            out_f.write(json.dumps(rel, ensure_ascii=False) + "\n")

    print(f"Constraints saved to {output_path}")
    print(f"Total relations processed: {len(relations)}")
    return output_path

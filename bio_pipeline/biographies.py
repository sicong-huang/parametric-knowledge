"""Step 5: Render natural language biographies for Person entities.

Supports multiple writing styles and multiple renditions per style.
Uses a separate LLM endpoint (OpenAI-compatible) from the main pipeline.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from openai import OpenAI
from tqdm import tqdm

from bio_pipeline.io import read_jsonl, ensure_output_dir


STYLE_PROMPTS = {
    "wiki": "Write in a neutral, encyclopedic Wikipedia style. Use third person and formal tone.",
    "blog": "Write in a casual, engaging blog post style. Be conversational and lively.",
    "news": "Write in a formal news article style. Be objective and concise, like a newspaper profile.",
    "social": "Write in a social media post style such as a tweet thread or Facebook post. Be engaging and conversational.",
}

def parse_style_spec(spec: str) -> list[tuple[str, int]]:
    """Parse style spec like 'wiki:2, blog:3, news:1' into [(style, count), ...]."""
    styles: list[tuple[str, int]] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            style, count_str = part.split(":", 1)
            styles.append((style.strip(), int(count_str.strip())))
        else:
            styles.append((part.strip(), 1))
    return styles


def format_edge_as_triple(
    edge: dict[str, Any],
    entities_by_id: dict[str, dict[str, Any]],
    relation_descriptions: dict[tuple[str, str, str], str],
) -> str:
    source_id = str(edge["source_id"])
    target_id = str(edge["target_id"])
    source_type = str(edge["source_type"])
    target_type = str(edge["target_type"])
    relation_type = str(edge["relation_type"])

    source = entities_by_id.get(
        source_id, {"id": source_id, "name": source_id, "entity_type": source_type}
    )
    target = entities_by_id.get(
        target_id, {"id": target_id, "name": target_id, "entity_type": target_type}
    )

    def display_type(et: str) -> str:
        return "Person" if et == "Person" or et.startswith("Person.") else et

    src_type = display_type(source.get("entity_type", source_type))
    tgt_type = display_type(target.get("entity_type", target_type))

    triple = f'({source.get("name", source_id)} | {src_type}) '
    triple += f"-[{relation_type}]-> "
    triple += f'({target.get("name", target_id)} | {tgt_type})'
    return triple


def index_relations(
    relations: list[dict[str, Any]],
) -> dict[tuple[str, str, str], str]:
    lookup: dict[tuple[str, str, str], str] = {}
    for rel in relations:
        rt = str(rel["relation_type"])
        st = str(rel["source_type"])
        tt = str(rel["target_type"])
        desc = str(rel["description"])
        if rt and st and tt and desc:
            lookup[(rt, st, tt)] = desc
    return lookup


def render_all_biographies_for_person(
    client: OpenAI,
    model: str,
    person: dict[str, Any],
    triples: list[str],
    styles: list[tuple[str, int]],
) -> list[dict[str, Any]]:
    """Render all biographies for one person as a single conversation chain.

    The first message establishes the person and triples along with the first
    style instruction.  Every subsequent message asks for a new rendition,
    carrying the full conversation history so the model can actively avoid
    repeating phrasing it has already used.
    """
    name = str(person["name"])
    entity_type = str(person["entity_type"])
    person_label = f"{name} ({entity_type})" if name else entity_type

    BASE_RULES = (
        "Use ONLY the provided knowledge triples and do not invent details. "
        "Do not miss any information from the triples. Include all the information from the triples.\n"
        "DO NOT use any formatting (no markdown, no HTML, no tables, no code blocks, "
        "no special characters, etc.).\n"
        "If information is sparse, keep the biography short."
    )

    messages: list[dict[str, str]] = []
    results: list[dict[str, Any]] = []
    is_first = True

    for style, count in styles:
        style_instruction = STYLE_PROMPTS.get(style, f"Write in {style} style.")
        for rendition in range(count):
            if is_first:
                prompt = (
                    f"Write a biography for the person below.\n"
                    f"{style_instruction}\n"
                    f"{BASE_RULES}\n\n"
                    f"Person: {person_label}\n"
                    "Knowledge triples:\n"
                    + "\n".join(f"- {triple}" for triple in triples)
                )
                is_first = False
            elif rendition == 0:
                # Switching to a new style — include the style instruction again.
                prompt = (
                    f"Now write another biography of the same person, this time in a different style.\n"
                    f"{style_instruction}\n"
                    "Cover all of the same information from the triples. Include all the information from the triples. But use completely "
                    "different wording, sentence structure, and organization from every "
                    "biography written above."
                )
            else:
                # Same style, additional rendition.
                prompt = (
                    f"Write yet another biography of the same person in {style} style. "
                    "Cover exactly the same information. Include all the information from the triples. But use completely different wording, "
                    "sentence structure, and organization from every biography written above."
                )

            messages.append({"role": "user", "content": prompt})
            response = client.chat.completions.create(model=model, messages=messages)
            bio = response.choices[0].message.content.strip()
            messages.append({"role": "assistant", "content": bio})

            results.append({"style": style, "rendition": rendition, "biography": bio})

    return results


def run(
    output_dir: str,
    bio_model: str,
    bio_base_url: str | None,
    bio_api_key: str | None,
    style_spec: str,
    max_workers: int,
) -> str:
    resolved_api_key = (
        bio_api_key
        or os.environ.get("BIO_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    client = OpenAI(base_url=bio_base_url, api_key=resolved_api_key)

    entities = read_jsonl(f"{output_dir}/entities.jsonl")
    edges = read_jsonl(f"{output_dir}/edges.jsonl")
    relations = read_jsonl(f"{output_dir}/relations.jsonl")

    styles = parse_style_spec(style_spec)
    if not styles:
        styles = [("wiki", 1)]

    entities_by_id: dict[str, dict[str, Any]] = {}
    person_entities: list[dict[str, Any]] = []
    for ent in entities:
        eid = str(ent["id"])
        if eid:
            entities_by_id[eid] = ent
        et = str(ent["entity_type"])
        if et == "Person" or et.startswith("Person."):
            person_entities.append(ent)

    person_ids = {str(p["id"]) for p in person_entities}
    edges_by_person: dict[str, list[dict[str, Any]]] = {pid: [] for pid in person_ids}
    for edge in edges:
        sid = str(edge["source_id"])
        tid = str(edge["target_id"])
        if sid in edges_by_person:
            edges_by_person[sid].append(edge)
        if tid in edges_by_person and tid != sid:
            edges_by_person[tid].append(edge)

    relation_descriptions = index_relations(relations)

    # Pre-compute triples per person; occupation_is edges are sorted first.
    triples_by_person: dict[str, list[str]] = {}
    for person in person_entities:
        pid = str(person["id"])
        person_edges = sorted(
            edges_by_person.get(pid, []),
            key=lambda e: 0 if str(e.get("relation_type", "")) == "occupation_is" else 1,
        )
        triples_by_person[pid] = [
            format_edge_as_triple(edge, entities_by_id, relation_descriptions)
            for edge in person_edges
        ]

    # One work item per person; all styles/renditions for that person are
    # generated as a single conversation chain inside the worker.
    results: list[tuple[int, list[dict[str, Any]]]] = []

    def process_person(
        idx: int, person: dict[str, Any]
    ) -> tuple[int, list[dict[str, Any]]]:
        pid = str(person["id"])
        triples = triples_by_person.get(pid, [])

        if not triples:
            name = str(person["name"]) or pid
            et = str(person["entity_type"]) or "Person"
            fallback_bio = f"{name} is a {et}. No additional relationships are recorded."
            records = [
                {
                    "id": pid,
                    "entity_type": person["entity_type"],
                    "name": person["name"],
                    "style": style,
                    "rendition": rendition,
                    "biography": fallback_bio,
                    "triples": triples,
                }
                for style, count in styles
                for rendition in range(count)
            ]
            return (idx, records)

        bio_entries = render_all_biographies_for_person(
            client, bio_model, person, triples, styles
        )
        records = [
            {
                "id": pid,
                "entity_type": person["entity_type"],
                "name": person["name"],
                "style": entry["style"],
                "rendition": entry["rendition"],
                "biography": entry["biography"],
                "triples": triples,
            }
            for entry in bio_entries
        ]
        return (idx, records)

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
        futures = {
            executor.submit(process_person, idx, person): idx
            for idx, person in enumerate(person_entities)
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Rendering biographies",
            unit="person",
        ):
            try:
                results.append(future.result())
            except Exception as e:
                idx = futures[future]
                print(f"Error rendering biographies for person {idx}: {e}")

    results.sort(key=lambda item: item[0])

    output_path = f"{output_dir}/biographies.jsonl"
    ensure_output_dir(output_path)
    total_bios = 0
    with open(output_path, "w") as out_f:
        for _, records in results:
            for record in records:
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_bios += 1

    total_styles = {s for s, _ in styles}
    print(f"Biographies saved to {output_path}")
    print(f"Total biographies: {total_bios} ({len(person_entities)} persons × {len(total_styles)} styles, chained per person)")
    return output_path

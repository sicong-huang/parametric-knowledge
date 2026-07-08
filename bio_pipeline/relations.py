"""Step 2: Generate relations between Person types and other entity types.

For each (Person subtype, non-Person type) pair, prompt the LLM to produce
1-3 relations or null if no sensible relation exists. Additionally, generate
exactly 5 generic Person→Person relations (no subtype constraint).
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm

from bio_pipeline.io import ensure_output_dir
from bio_pipeline.models import Relation, RelationOrNull


def read_entity_types(path: str) -> list[str]:
    entity_types: list[str] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            et = data.get("entity_type")
            if et:
                entity_types.append(str(et))
    return entity_types


def generate_pair_relations(
    client: OpenAI,
    model: str,
    source_type: str,
    target_type: str,
) -> list[Relation]:
    prompt = (
        "You are generating relation types for a synthetic knowledge graph.\n\n"
        f'Source entity type: "{source_type}"\n'
        f'Target entity type: "{target_type}"\n\n'
        "Does it make sense for these two entity types to be connected by a relation "
        f"where {source_type} is the subject and {target_type} is the object?\n\n"
        "If YES: create exactly 1 relation that best captures the relationship. It should have:\n"
        f'- "source_type" = exactly "{source_type}"\n'
        f'- "target_type" = exactly "{target_type}"\n'
        '- "relation_type" = a concise snake_case verb phrase (e.g., acted_in, works_at)\n'
        '- "description" = a short human-readable explanation\n\n'
        "If NO (the pair doesn't make sense): return null for relations.\n\n"
        "Example: source=Person.Actor, target=Org.FilmStudio → "
        'relations=[{relation_type: "signed_with", source_type: "Person.Actor", '
        'target_type: "Org.FilmStudio", description: "The actor is signed with the film studio."}]\n'
        "Example: source=Person.Actor, target=Work.Software → relations=null"
    )

    response = client.responses.parse(
        model=model,
        instructions="Return structured JSON that matches the provided schema.",
        input=prompt,
        text_format=RelationOrNull,
    )
    result = response.output_parsed
    if result.relations is None:
        return []
    # Ensure source/target types are correct
    valid: list[Relation] = []
    for rel in result.relations:
        if rel.source_type == source_type and rel.target_type == target_type:
            valid.append(rel)
    return valid[:1]


def generate_person_to_person_relations(
    client: OpenAI,
    model: str,
) -> list[Relation]:
    prompt = (
        "You are generating relation types for a synthetic knowledge graph.\n\n"
        "Generate exactly 5 relations that connect one Person to another Person.\n"
        "These are generic relations that apply to any type of person regardless of "
        "their profession or role (e.g., Actor, Engineer, Writer, etc.).\n\n"
        "Each relation should have:\n"
        '- "source_type" = "Person"\n'
        '- "target_type" = "Person"\n'
        '- "relation_type" = a concise snake_case verb phrase\n'
        '- "description" = a short human-readable explanation\n\n'
        "Do NOT include any family or marital relations such as is_parent_of, "
        "is_sibling_of, married_to, is_child_of, is_spouse_of, or any other "
        "parent, sibling, or marital relation.\n"
        "Examples of good Person→Person relations: mentored_by, collaborated_with, "
        "is_colleague_of, inspired_by, is_friend_of.\n"
        "Make the relations diverse and common-sense. Return exactly 5."
    )

    response = client.responses.parse(
        model=model,
        instructions="Return structured JSON that matches the provided schema.",
        input=prompt,
        text_format=RelationOrNull,
    )
    result = response.output_parsed
    if result.relations is None:
        return []
    valid: list[Relation] = []
    for rel in result.relations:
        rel.source_type = "Person"
        rel.target_type = "Person"
        valid.append(rel)
    return valid[:5]


def run(
    output_dir: str,
    entity_types_path: str,
    model: str,
    base_url: str | None,
    max_workers: int,
) -> str:
    client = OpenAI(base_url=base_url)

    all_types = read_entity_types(entity_types_path)
    if not all_types:
        raise SystemExit(f"No entity types found in {entity_types_path}")

    person_types = [t for t in all_types if t.startswith("Person.")]
    non_person_types = [t for t in all_types if not t.startswith("Person.")]

    # Build list of (source, target) pairs to process
    pairs: list[tuple[str, str]] = []
    for pt in person_types:
        for npt in non_person_types:
            pairs.append((pt, npt))

    all_relations: list[Relation] = []
    seen: set[tuple[str, str, str]] = set()

    def add_relations(rels: list[Relation]) -> None:
        for rel in rels:
            key = (rel.relation_type, rel.source_type, rel.target_type)
            if key not in seen:
                seen.add(key)
                all_relations.append(rel)

    # Generate pair-by-pair relations in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(generate_pair_relations, client, model, src, tgt): (src, tgt)
            for src, tgt in pairs
        }
        for future in tqdm(
            as_completed(future_map),
            total=len(future_map),
            desc="Generating relations (pairs)",
            unit="pair",
        ):
            src, tgt = future_map[future]
            try:
                rels = future.result()
                add_relations(rels)
            except Exception as e:
                print(f"Error generating relations for {src} → {tgt}: {e}")

    # Generate Person→Person relations
    print("Generating Person→Person relations...")
    try:
        p2p_rels = generate_person_to_person_relations(client, model)
        add_relations(p2p_rels)
        print(f"  Generated {len(p2p_rels)} Person→Person relations")
    except Exception as e:
        print(f"Error generating Person→Person relations: {e}")

    output_path = f"{output_dir}/relations.jsonl"
    ensure_output_dir(output_path)
    with open(output_path, "w") as out_f:
        for rel in all_relations:
            out_f.write(
                json.dumps(
                    {
                        "relation_type": rel.relation_type,
                        "source_type": rel.source_type,
                        "target_type": rel.target_type,
                        "description": rel.description,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    p2p_count = sum(1 for r in all_relations if r.source_type == "Person" and r.target_type == "Person")
    print(f"Relations saved to {output_path}")
    print(f"Total relations: {len(all_relations)} ({p2p_count} Person→Person)")
    return output_path

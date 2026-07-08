"""Step 4: Generate random graph edges from entities and constrained relations."""

from __future__ import annotations

import json
import random
from typing import Any, Optional

from bio_pipeline.io import read_jsonl, ensure_output_dir


LOCATION_TOKENS = ("born_in", "based_in", "located_in", "headquartered_in")


def is_location_relation(relation_type: str) -> bool:
    rel = relation_type.lower().strip()
    return any(token in rel for token in LOCATION_TOKENS)


def is_place_city(entity_type: str) -> bool:
    return entity_type.strip().startswith("Place.City")


def is_place_country(entity_type: str) -> bool:
    return entity_type.strip().startswith("Place.Country")


def index_entities(
    entities: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    by_type: dict[str, list[dict[str, Any]]] = {}
    for ent in entities:
        entity_type = str(ent.get("entity_type", "")).strip()
        if entity_type:
            by_type.setdefault(entity_type, []).append(ent)
    return by_type


def resolve_entities(
    type_str: str,
    entities_by_type: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Resolve entity type to list of entities.

    Exact match first; if not found, prefix-expand (e.g. "Person" matches
    all "Person.*" subtypes).
    """
    if type_str in entities_by_type:
        return entities_by_type[type_str]
    # Prefix expansion
    result: list[dict[str, Any]] = []
    prefix = type_str + "."
    for key, ents in entities_by_type.items():
        if key.startswith(prefix):
            result.extend(ents)
    return result


def is_person_to_person(relation: dict[str, Any]) -> bool:
    src = str(relation.get("source_type", "")).strip()
    tgt = str(relation.get("target_type", "")).strip()
    return src == "Person" and tgt == "Person"


def generate_edges_for_relation(
    relation: dict[str, Any],
    entities_by_type: dict[str, list[dict[str, Any]]],
    density: float,
    rng: random.Random,
    shared_source_cap: Optional[dict[str, int]] = None,
) -> list[dict[str, Any]]:
    source_type = str(relation.get("source_type", "")).strip()
    target_type = str(relation.get("target_type", "")).strip()
    relation_type = str(relation.get("relation_type", "")).strip()
    if not source_type or not target_type or not relation_type:
        return []

    sources = resolve_entities(source_type, entities_by_type)
    targets = resolve_entities(target_type, entities_by_type)
    if not sources or not targets:
        return []

    # Read inline constraints
    min_per_source = 0
    max_per_source = 2
    min_per_target = 0
    max_per_target: int | None = None

    for key, default in [
        ("min_per_source", min_per_source),
        ("max_per_source", max_per_source),
    ]:
        val = relation.get(key)
        if isinstance(val, int) and val > 0:
            if key.startswith("min"):
                min_per_source = max(min_per_source, val)
            else:
                max_per_source = min(max_per_source, val)

    val = relation.get("min_per_target")
    if isinstance(val, int) and val > 0:
        min_per_target = max(min_per_target, val)
    val = relation.get("max_per_target")
    if isinstance(val, int) and val > 0:
        max_per_target = val

    if max_per_source < min_per_source:
        max_per_source = min_per_source
    if max_per_target is not None and max_per_target < min_per_target:
        max_per_target = min_per_target

    # Track capacities
    target_capacity: dict[str, int] | None = None
    if max_per_target is not None:
        target_capacity = {str(t["id"]): max_per_target for t in targets if "id" in t}

    source_capacity: dict[str, int] = {}
    for s in sources:
        sid = str(s.get("id", "")).strip()
        if sid:
            source_capacity[sid] = max_per_source

    edges: list[dict[str, Any]] = []
    seen_edges: set[tuple[str, str, str]] = set()

    def can_use_target(tid: str) -> bool:
        if target_capacity is None:
            return True
        return target_capacity.get(tid, 0) > 0

    def add_edge(sid: str, tid: str, s_type: str, t_type: str) -> bool:
        if source_capacity.get(sid, 0) <= 0:
            return False
        if shared_source_cap is not None and shared_source_cap.get(sid, 0) <= 0:
            return False
        if not can_use_target(tid):
            return False
        if sid == tid:
            return False
        key = (sid, relation_type, tid)
        if key in seen_edges:
            return False
        seen_edges.add(key)
        edges.append({
            "source_id": sid,
            "source_type": s_type,
            "relation_type": relation_type,
            "target_id": tid,
            "target_type": t_type,
        })
        source_capacity[sid] = max(0, source_capacity[sid] - 1)
        if shared_source_cap is not None:
            shared_source_cap[sid] = max(0, shared_source_cap[sid] - 1)
        if target_capacity is not None:
            target_capacity[tid] = max(0, target_capacity[tid] - 1)
        return True

    targets_shuffled = list(targets)
    rng.shuffle(targets_shuffled)

    # Phase 1: satisfy min_per_target
    if min_per_target > 0:
        source_ids = [(str(s["id"]).strip(), str(s["entity_type"]).strip()) for s in sources if s.get("id")]
        for target in targets_shuffled:
            tid = str(target.get("id", "")).strip()
            t_type = str(target.get("entity_type", "")).strip()
            if not tid:
                continue
            for _ in range(min_per_target):
                candidates = [
                    (sid, st)
                    for sid, st in source_ids
                    if source_capacity.get(sid, 0) > 0
                    and (sid, relation_type, tid) not in seen_edges
                    and sid != tid
                ]
                if not candidates:
                    break
                chosen_sid, chosen_st = rng.choice(candidates)
                add_edge(chosen_sid, tid, chosen_st, t_type)

    # Phase 2: satisfy min_per_source
    for source in sources:
        sid = str(source.get("id", "")).strip()
        s_type = str(source.get("entity_type", "")).strip()
        if not sid:
            continue
        current = sum(1 for e in edges if e["source_id"] == sid)
        if current >= min_per_source:
            continue
        needed = min_per_source - current
        for _ in range(needed):
            candidates = [
                t
                for t in targets
                if (tid := str(t.get("id", "")).strip())
                and can_use_target(tid)
                and (sid, relation_type, tid) not in seen_edges
                and sid != tid
            ]
            if not candidates:
                break
            chosen = rng.choice(candidates)
            tid = str(chosen["id"]).strip()
            t_type = str(chosen["entity_type"]).strip()
            add_edge(sid, tid, s_type, t_type)

    # Phase 3: density-based random edges
    density = max(0.0, min(1.0, density))
    for source in sources:
        sid = str(source.get("id", "")).strip()
        s_type = str(source.get("entity_type", "")).strip()
        if not sid:
            continue
        remaining = source_capacity.get(sid, 0)
        if remaining <= 0:
            continue
        candidates = [
            t
            for t in targets
            if (tid := str(t.get("id", "")).strip())
            and can_use_target(tid)
            and (sid, relation_type, tid) not in seen_edges
            and sid != tid
        ]
        if not candidates:
            continue
        cap = min(remaining, len(candidates))
        max_desired = int(round(cap * density))
        desired = rng.randint(0, max_desired) if max_desired > 0 else 0
        if desired <= 0:
            continue
        chosen_targets = rng.sample(candidates, k=desired)
        for t in chosen_targets:
            tid = str(t["id"]).strip()
            t_type = str(t["entity_type"]).strip()
            add_edge(sid, tid, s_type, t_type)

    return edges


def generate_occupation_edges(
    entities_by_type: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Generate an `occupation_is` edge for every Person sub-type entity.

    For an entity of type ``Person.Actor`` the edge target is the string
    ``"actor"`` (the sub-type label lowercased), stored as a literal id with
    type ``"Occupation"``.
    """
    edges: list[dict[str, Any]] = []
    prefix = "Person."
    for entity_type, entities in entities_by_type.items():
        if not entity_type.startswith(prefix):
            continue
        sub_type = entity_type[len(prefix):]
        if not sub_type:
            continue
        occupation_label = sub_type.lower()
        for ent in entities:
            sid = str(ent.get("id", "")).strip()
            if not sid:
                continue
            edges.append({
                "source_id": sid,
                "source_type": entity_type,
                "relation_type": "occupation_is",
                "target_id": occupation_label,
                "target_type": "Occupation",
            })
    return edges


def enforce_city_country_consistency(
    edges: list[dict[str, Any]],
    rng: random.Random,
) -> list[dict[str, Any]]:
    city_to_country: dict[str, str] = {}
    country_to_cities: dict[str, list[str]] = {}

    for edge in edges:
        source_type = str(edge.get("source_type", "")).strip()
        target_type = str(edge.get("target_type", "")).strip()
        if not (is_place_city(source_type) and is_place_country(target_type)):
            continue
        city_id = str(edge.get("source_id", "")).strip()
        country_id = str(edge.get("target_id", "")).strip()
        if city_id and country_id:
            city_to_country[city_id] = country_id
            country_to_cities.setdefault(country_id, []).append(city_id)

    if not city_to_country:
        return edges

    edges_by_source: dict[str, list[int]] = {}
    for idx, edge in enumerate(edges):
        sid = str(edge.get("source_id", "")).strip()
        if sid:
            edges_by_source.setdefault(sid, []).append(idx)

    edge_keys = {
        (
            str(e.get("source_id", "")).strip(),
            str(e.get("relation_type", "")).strip(),
            str(e.get("target_id", "")).strip(),
        )
        for e in edges
    }

    removed: set[int] = set()
    for source_id, idxs in edges_by_source.items():
        country_edges = [
            idx
            for idx in idxs
            if is_place_country(str(edges[idx].get("target_type", "")))
            and is_location_relation(str(edges[idx].get("relation_type", "")))
        ]
        if not country_edges:
            continue
        primary_country = ""
        for idx in country_edges:
            primary_country = str(edges[idx].get("target_id", "")).strip()
            if primary_country:
                break
        if not primary_country:
            continue
        candidate_cities = country_to_cities.get(primary_country, [])

        for idx in idxs:
            edge = edges[idx]
            target_type = str(edge.get("target_type", "")).strip()
            rel_type = str(edge.get("relation_type", "")).strip()
            if not (is_place_city(target_type) and is_location_relation(rel_type)):
                continue
            city_id = str(edge.get("target_id", "")).strip()
            if not city_id:
                continue
            expected_country = city_to_country.get(city_id)
            if not expected_country or expected_country == primary_country:
                continue

            replacement = ""
            if candidate_cities:
                candidate_ids = [
                    c
                    for c in candidate_cities
                    if (source_id, rel_type, c) not in edge_keys
                ]
                if candidate_ids:
                    replacement = rng.choice(candidate_ids)

            if replacement:
                old_key = (source_id, rel_type, city_id)
                new_key = (source_id, rel_type, replacement)
                edge_keys.discard(old_key)
                edge_keys.add(new_key)
                edge["target_id"] = replacement
            else:
                removed.add(idx)

    if removed:
        edges = [e for idx, e in enumerate(edges) if idx not in removed]
    return edges


def run(
    output_dir: str,
    density: float,
    seed: int,
) -> str:
    entities = read_jsonl(f"{output_dir}/entities.jsonl")
    relations = read_jsonl(f"{output_dir}/relations_w_constraints.jsonl")

    density = max(0.0, min(1.0, density))
    entities_by_type = index_entities(entities)
    rng = random.Random(seed)

    # Build a shared cap of 2 outgoing P2P edges per Person, across all P2P relations.
    P2P_MAX_OUTGOING = 2
    all_person_entities = resolve_entities("Person", entities_by_type)
    p2p_shared_cap: dict[str, int] = {
        str(e["id"]): P2P_MAX_OUTGOING
        for e in all_person_entities
        if e.get("id")
    }

    all_edges: list[dict[str, Any]] = []
    for relation in relations:
        edges = generate_edges_for_relation(
            relation=relation,
            entities_by_type=entities_by_type,
            density=density,
            rng=rng,
            shared_source_cap=p2p_shared_cap if is_person_to_person(relation) else None,
        )
        all_edges.extend(edges)

    all_edges = enforce_city_country_consistency(all_edges, rng)

    occupation_edges = generate_occupation_edges(entities_by_type)
    all_edges.extend(occupation_edges)

    output_path = f"{output_dir}/edges.jsonl"
    ensure_output_dir(output_path)
    with open(output_path, "w") as out_f:
        for edge in all_edges:
            out_f.write(json.dumps(edge, ensure_ascii=False) + "\n")

    print(f"Edges saved to {output_path}")
    print(f"Total edges generated: {len(all_edges)}")
    return output_path

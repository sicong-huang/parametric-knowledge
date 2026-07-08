#!/usr/bin/env python3
"""
create_bio - Synthetic Knowledge Graph Generator

Single entry point for the full pipeline or individual steps.

Full pipeline:
    uv run create_bio.py --dir data/bio/run1 --entity-types entity_types.jsonl

Individual steps:
    uv run create_bio.py entity     --dir data/bio/run1 --entity-types entity_types.jsonl
    uv run create_bio.py relation   --dir data/bio/run1 --entity-types entity_types.jsonl
    uv run create_bio.py constraint --dir data/bio/run1
    uv run create_bio.py graph      --dir data/bio/run1 --density 0.7 --seed 42
    uv run create_bio.py biography  --dir data/bio/run1 --bio-model MODEL --bio-base-url URL --style 'wiki:2'
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="synkg",
        description="Synthetic Knowledge Graph Generator",
    )

    # Global options
    parser.add_argument(
        "--dir", required=True,
        help="Working directory: outputs are written here, and intermediate steps read required files from here."
    )
    parser.add_argument(
        "--model", default="gpt-5.4", help="OpenAI model for steps 1-4 (default: gpt-5-nano)"
    )
    parser.add_argument(
        "--base-url", default=None, help="OpenAI API base URL for steps 1-4"
    )
    parser.add_argument(
        "--max-workers", type=int, default=8, help="Max parallel workers (default: 8)"
    )

    # Full pipeline options (also used by subcommands where relevant)
    parser.add_argument(
        "--entity-types", default=None, help="Input JSONL file with entity types"
    )
    parser.add_argument(
        "--relation-model", default=None,
        help="Model override for relation generation (default: uses --model)"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=100, help="Max names per API call for entity generation (default: 100)"
    )
    parser.add_argument(
        "--density", type=float, default=0.7, help="Edge density 0.0-1.0 for graph generation (default: 0.7)"
    )
    parser.add_argument(
        "--seed", type=int, default=7, help="Random seed for graph generation (default: 7)"
    )
    parser.add_argument(
        "--bio-model", default=None, help="Model for biography rendering (OpenAI-compatible)"
    )
    parser.add_argument(
        "--bio-base-url", default=os.environ.get("BIO_BASE_URL"),
        help="Base URL for biography LLM endpoint (default: $BIO_BASE_URL)"
    )
    parser.add_argument(
        "--bio-api-key", default=None,
        help="API key for biography LLM endpoint (default: $BIO_API_KEY, then $OPENAI_API_KEY)"
    )
    parser.add_argument(
        "--bio-max-workers", type=int, default=None,
        help="Max parallel workers for biography generation (default: uses --max-workers)"
    )
    parser.add_argument(
        "--style", default="wiki:1",
        help="Biography style spec, e.g. 'wiki:2, blog:3, news:1' (default: wiki:1)"
    )

    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("entity", help="Step 1: Generate entity names")
    subparsers.add_parser("relation", help="Step 2: Generate relations")
    subparsers.add_parser("constraint", help="Step 3: Generate relation constraints")
    subparsers.add_parser("graph", help="Step 4: Generate graph edges")
    subparsers.add_parser("biography", help="Step 5: Render biographies")
    bio_uneven_parser = subparsers.add_parser("biography_uneven", help="Step 6: Create uneven-frequency biography set")
    bio_uneven_parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap persons per bucket (for quick test runs, e.g. --limit 2)"
    )

    return parser


def run_entity(args: argparse.Namespace) -> None:
    if not args.entity_types:
        print("Error: --entity-types is required for entity generation", file=sys.stderr)
        sys.exit(1)
    from bio_pipeline.entities import run
    run(
        output_dir=args.dir,
        entity_types_path=args.entity_types,
        model=args.model,
        base_url=args.base_url,
        max_workers=args.max_workers,
        chunk_size=args.chunk_size,
    )


def run_relation(args: argparse.Namespace) -> None:
    if not args.entity_types:
        print("Error: --entity-types is required for relation generation", file=sys.stderr)
        sys.exit(1)
    from bio_pipeline.relations import run
    model = args.relation_model or args.model
    run(
        output_dir=args.dir,
        entity_types_path=args.entity_types,
        model=model,
        base_url=args.base_url,
        max_workers=args.max_workers,
    )


def run_constraint(args: argparse.Namespace) -> None:
    from bio_pipeline.constraints import run
    run(
        output_dir=args.dir,
        model=args.model,
        base_url=args.base_url,
        max_workers=args.max_workers,
    )


def run_graph(args: argparse.Namespace) -> None:
    from bio_pipeline.graph import run
    run(
        output_dir=args.dir,
        density=args.density,
        seed=args.seed,
    )


def run_biography(args: argparse.Namespace) -> None:
    if not args.bio_model:
        print("Error: --bio-model is required for biography generation", file=sys.stderr)
        sys.exit(1)
    from bio_pipeline.biographies import run
    run(
        output_dir=args.dir,
        bio_model=args.bio_model,
        bio_base_url=args.bio_base_url,
        bio_api_key=args.bio_api_key,
        style_spec=args.style,
        max_workers=args.bio_max_workers if args.bio_max_workers is not None else args.max_workers,
    )


def run_biography_uneven(args: argparse.Namespace) -> None:
    from bio_pipeline.biographies_uneven import run
    run(
        output_dir=args.dir,
        bio_model=args.bio_model,
        bio_base_url=args.bio_base_url,
        bio_api_key=args.bio_api_key,
        max_workers=args.bio_max_workers if args.bio_max_workers is not None else args.max_workers,
        seed=args.seed,
        limit=getattr(args, "limit", None),
    )


def run_full_pipeline(args: argparse.Namespace) -> None:
    if not args.entity_types:
        print("Error: --entity-types is required for the full pipeline", file=sys.stderr)
        sys.exit(1)
    if not args.bio_model:
        print("Error: --bio-model is required for the full pipeline", file=sys.stderr)
        sys.exit(1)

    Path(args.dir).mkdir(parents=True, exist_ok=True)

    print("\n==> Step 1: Generate entities")
    run_entity(args)

    print("\n==> Step 2: Generate relations")
    run_relation(args)

    print("\n==> Step 3: Generate constraints")
    run_constraint(args)

    print("\n==> Step 4: Generate graph edges")
    run_graph(args)

    print("\n==> Step 5: Render biographies")
    run_biography(args)

    print(f"\nPipeline complete. Outputs in {args.dir}/")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    Path(args.dir).mkdir(parents=True, exist_ok=True)

    dispatch = {
        "entity": run_entity,
        "relation": run_relation,
        "constraint": run_constraint,
        "graph": run_graph,
        "biography": run_biography,
        "biography_uneven": run_biography_uneven,
    }

    if args.command:
        dispatch[args.command](args)
    else:
        run_full_pipeline(args)


if __name__ == "__main__":
    main()

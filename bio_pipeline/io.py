"""Shared JSONL I/O helpers."""

from __future__ import annotations

import json
import os
from typing import Any, Iterable


def read_jsonl(path: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def write_jsonl(path: str, items: Iterable[dict[str, Any]]) -> None:
    ensure_output_dir(path)
    with open(path, "w") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def ensure_output_dir(path: str) -> None:
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

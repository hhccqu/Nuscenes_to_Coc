"""Dataset export helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def write_json(path: Path, payload: Dict | List) -> None:
    """Write JSON with utf-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def read_json(path: Path) -> Dict | List:
    """Read JSON with utf-8 encoding."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    """Write JSONL with utf-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def read_jsonl(path: Path) -> List[Dict]:
    """Read JSONL with utf-8 encoding."""
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows

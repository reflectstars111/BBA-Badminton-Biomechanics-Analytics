from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv_rows(csv_path: str | Path) -> list[dict[str, str]]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(
    csv_path: str | Path,
    fieldnames: list[str],
    rows: list[dict[str, object]],
) -> None:
    csv_path = ensure_parent(csv_path)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_json(path: str | Path) -> Any:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    path = ensure_parent(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def missing_fields(row: dict[str, object], required_fields: set[str]) -> list[str]:
    return [field for field in sorted(required_fields) if row.get(field) in (None, "")]


def require_csv_fields(fieldnames: list[str] | None, required_fields: set[str]) -> list[str]:
    available = set(fieldnames or [])
    return [field for field in sorted(required_fields) if field not in available]


from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

from .paths import discover_project_root


def _coerce_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"", "null", "None", "~"}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if re.fullmatch(r"-?(\d+\.\d*|\d*\.\d+)", value):
        return float(value)
    return value.strip("\"'")


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    lines = [line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]
    for index, raw_line in enumerate(lines):
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if line.startswith("- "):
            parent.append(_coerce_scalar(line[2:]))
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if value:
            parent[key] = _coerce_scalar(value)
            continue
        next_line = ""
        for candidate in lines[index + 1:]:
            next_indent = len(candidate) - len(candidate.lstrip(" "))
            if next_indent <= indent:
                break
            next_line = candidate.strip()
            break
        container = [] if next_line.startswith("- ") else {}
        parent[key] = container
        stack.append((indent, container))
    return root


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
    except ImportError:
        return _parse_simple_yaml(text)
    payload = yaml.safe_load(text)
    return payload or {}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def default_config_path(root: str | Path | None = None) -> Path:
    project_root = Path(root) if root is not None else discover_project_root()
    default_path = project_root / "configs" / "default.yaml"
    if default_path.exists():
        return default_path
    return project_root / "configs" / "project_config.yaml"


def load_config(
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
    root: str | Path | None = None,
) -> dict[str, Any]:
    base = load_yaml(default_config_path(root))
    if config_path:
        base = deep_merge(base, load_yaml(config_path))
    if overrides:
        base = deep_merge(base, overrides)
    return base

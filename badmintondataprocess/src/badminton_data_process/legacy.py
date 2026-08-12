from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def scripts_dir() -> Path:
    return project_root() / "scripts"


@lru_cache(maxsize=None)
def load_legacy_module(script_name: str):
    script_path = scripts_dir() / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Legacy script not found: {script_path}")
    module_name = f"_bdp_legacy_{script_path.stem}"
    if str(scripts_dir()) not in sys.path:
        sys.path.insert(0, str(scripts_dir()))
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load legacy script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def run_legacy_main(script_name: str, argv: list[str] | None = None) -> int:
    module = load_legacy_module(script_name)
    previous_argv = sys.argv[:]
    sys.argv = [str(scripts_dir() / script_name), *(argv or [])]
    try:
        result = module.main()
        return int(result or 0)
    finally:
        sys.argv = previous_argv


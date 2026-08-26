from __future__ import annotations

from badminton_data_process.legacy import load_legacy_module, run_legacy_main


def _module():
    return load_legacy_module("tactical_analysis.py")


def analyze_tactics(*args, **kwargs):
    return _module().analyze_tactics(*args, **kwargs)


def main(argv: list[str] | None = None) -> int:
    return run_legacy_main("tactical_analysis.py", argv)

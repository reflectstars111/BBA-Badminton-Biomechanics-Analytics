from __future__ import annotations

from badminton_data_process.legacy import load_legacy_module, run_legacy_main


def _module():
    return load_legacy_module("register_local_match.py")


def register_local_match(*args, **kwargs):
    return _module().register_local_match(*args, **kwargs)


def main(argv: list[str] | None = None) -> int:
    return run_legacy_main("register_local_match.py", argv)

from __future__ import annotations

from badminton_data_process.legacy import load_legacy_module, run_legacy_main


def _module():
    return load_legacy_module("compare_shuttle_trackers.py")


def compare_trackers(*args, **kwargs):
    return _module().compare_trackers(*args, **kwargs)


def main(argv: list[str] | None = None) -> int:
    return run_legacy_main("compare_shuttle_trackers.py", argv)

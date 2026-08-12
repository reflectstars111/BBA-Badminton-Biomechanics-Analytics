from __future__ import annotations

from badminton_data_process.legacy import load_legacy_module, run_legacy_main


def _module():
    return load_legacy_module("player_tracking.py")


def detect_players(*args, **kwargs):
    return _module().detect_players(*args, **kwargs)


def track_players(*args, **kwargs):
    return _module().track_players(*args, **kwargs)


def main(argv: list[str] | None = None) -> int:
    return run_legacy_main("player_tracking.py", argv)


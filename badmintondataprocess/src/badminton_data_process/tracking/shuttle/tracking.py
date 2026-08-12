from __future__ import annotations

from badminton_data_process.legacy import load_legacy_module, run_legacy_main


def _module():
    return load_legacy_module("shuttle_tracking.py")


def candidate_points(*args, **kwargs):
    return _module().candidate_points(*args, **kwargs)


def track_shuttle(*args, **kwargs):
    return _module().track_shuttle(*args, **kwargs)


def main(argv: list[str] | None = None) -> int:
    return run_legacy_main("shuttle_tracking.py", argv)


from __future__ import annotations

from badminton_data_process.legacy import run_legacy_main


def main(argv: list[str] | None = None) -> int:
    return run_legacy_main("visualize_tracking.py", argv)


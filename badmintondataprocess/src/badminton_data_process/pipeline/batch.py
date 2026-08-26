from __future__ import annotations

from badminton_data_process.legacy import load_legacy_module, run_legacy_main


def _module():
    return load_legacy_module("batch_pipeline.py")


def batch_run_matches(*args, **kwargs):
    return _module().batch_run_matches(*args, **kwargs)


def main(argv: list[str] | None = None) -> int:
    return run_legacy_main("batch_pipeline.py", argv)

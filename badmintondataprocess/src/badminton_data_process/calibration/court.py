from __future__ import annotations

from badminton_data_process.legacy import load_legacy_module, run_legacy_main


def _module():
    return load_legacy_module("court_calibration.py")


def detect_court_corners(*args, **kwargs):
    return _module().detect_court_corners(*args, **kwargs)


def court_line_support(*args, **kwargs):
    return _module().court_line_support(*args, **kwargs)


def calibrate_video(*args, **kwargs):
    return _module().calibrate_video(*args, **kwargs)


def calibrate_courts(*args, **kwargs):
    return _module().calibrate_courts(*args, **kwargs)


def main(argv: list[str] | None = None) -> int:
    return run_legacy_main("court_calibration.py", argv)

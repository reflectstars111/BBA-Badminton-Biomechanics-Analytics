from __future__ import annotations

from badminton_data_process.legacy import load_legacy_module, run_legacy_main


def _module():
    return load_legacy_module("trajectory_smoothing.py")


def interpolate_series(*args, **kwargs):
    return _module().interpolate_series(*args, **kwargs)


def rolling_median(*args, **kwargs):
    return _module().rolling_median(*args, **kwargs)


def ema_smooth(*args, **kwargs):
    return _module().ema_smooth(*args, **kwargs)


def smooth_trajectory(*args, **kwargs):
    return _module().smooth_trajectory(*args, **kwargs)


def main(argv: list[str] | None = None) -> int:
    return run_legacy_main("trajectory_smoothing.py", argv)


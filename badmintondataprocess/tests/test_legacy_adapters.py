from __future__ import annotations

from badminton_data_process.smoothing.trajectory import ema_smooth, interpolate_series, rolling_median


def test_legacy_smoothing_adapter_exposes_core_functions() -> None:
    filled, gap_mask = interpolate_series([0.0, None, 2.0], [True, False, True], 2)
    assert filled == [0.0, 1.0, 2.0]
    assert gap_mask == [False, True, False]
    assert rolling_median([1.0, 10.0, 2.0], 3) == [5.5, 2.0, 6.0]
    assert ema_smooth([1.0, None, 3.0], 0.5) == [1.0, 1.0, 2.0]

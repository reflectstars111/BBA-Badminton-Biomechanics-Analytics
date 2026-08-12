from __future__ import annotations

from badminton_data_process.calibration.geometry import project_point
from badminton_data_process.preprocess.timeline import smooth_main_view_segments


def test_project_point_identity_homography() -> None:
    x, y = project_point(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        (12.5, 7.25),
    )
    assert x == 12.5
    assert y == 7.25


def test_smooth_main_view_segments_filters_and_bridges_short_gap() -> None:
    segments = [
        {"start": 0.0, "end": 2.0, "label": "MAIN_LIVE_VIEW", "confidence": 0.7},
        {"start": 4.0, "end": 8.0, "label": "MAIN_LIVE_VIEW", "confidence": 0.8},
        {"start": 9.0, "end": 13.0, "label": "MAIN_LIVE_VIEW", "confidence": 0.75},
        {"start": 20.0, "end": 25.0, "label": "REPLAY", "confidence": 0.9},
    ]
    result = smooth_main_view_segments(segments, min_main_duration=3.0, max_gap=2.0)
    assert result == [
        {"start": 4.0, "end": 13.0, "label": "MAIN_LIVE_VIEW", "confidence": 0.8}
    ]


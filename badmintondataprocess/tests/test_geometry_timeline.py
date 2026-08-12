from __future__ import annotations

from badminton_data_process.calibration.geometry import project_point
from badminton_data_process.preprocess.timeline import smooth_main_view_segments
from badminton_data_process.rally.segmentation import (
    merge_candidate_segments,
    select_live_rallies,
)


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


def test_rally_boundary_follows_continuous_court_view_not_sparse_motion() -> None:
    rows = [
        {
            "sample_frame": float(frame),
            "is_court_view": 1.0,
            "is_candidate": float(frame in {150, 450}),
        }
        for frame in range(0, 600, 10)
    ]

    assert merge_candidate_segments(
        analysis_rows=rows,
        sample_every=10,
        min_rally_seconds=4.0,
        max_rally_seconds=45.0,
        max_gap_seconds=3.0,
        pad_before_seconds=0.0,
        pad_after_seconds=0.0,
        fps=30.0,
        max_pre_context_seconds=2.2,
        max_post_context_seconds=1.4,
        allowed_context_drop_samples=1,
    ) == [(0, 600)]


def test_live_rally_is_last_court_view_before_score_update() -> None:
    view_segments = [(300, 600), (700, 900), (1000, 1300)]

    assert select_live_rallies(
        view_segments,
        score_change_frames=[620, 1320],
        fps=30.0,
    ) == [(300, 600), (1000, 1300)]

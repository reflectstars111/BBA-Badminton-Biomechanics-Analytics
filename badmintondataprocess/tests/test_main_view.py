from __future__ import annotations

import pytest

from badminton_data_process.main_view.analyze import contiguous_segments
from badminton_data_process.core.schemas import MainViewLabel, parse_main_view_label
from badminton_data_process.main_view.scoring import FrameScore, geometry_score, reject_reason
from badminton_data_process.review.main_view import choose_reject_reason, projection_quality

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")


def _score(frame: int, is_main: int, score: float = 0.8) -> FrameScore:
    return FrameScore(
        sample_frame=frame,
        timestamp=frame / 30.0,
        main_view_score=score,
        court_score=0.8,
        geometry_score=0.8,
        layout_score=0.8,
        stability_score=0.8,
        line_score=0.8,
        reject_score=0.1,
        court_area_ratio=0.3,
        court_span_x=0.7,
        court_span_y=0.6,
        player_candidate_count=2,
        player_split_sides=1,
        is_main_view=is_main,
        reject_reason="" if is_main else "low_main_view_score",
    )


def test_geometry_score_accepts_reasonable_court_trapezoid() -> None:
    corners = np.array([[220, 120], [1060, 120], [1220, 700], [80, 700]], dtype=np.float32)
    assert geometry_score(corners, (720, 1280, 3)) > 0.65


def test_low_angle_profile_can_relax_reject_gate_without_lowering_global_default() -> None:
    args = (0.30, 0.31, 0.68, 0.58, 0.34, 0.28)

    assert reject_reason(*args) == "low_court_geometry_score"
    assert reject_reason(*args, max_reject_score=0.65) == ""


def test_contiguous_segments_merges_small_gaps_and_filters_short_segments() -> None:
    scores = [_score(0, 1), _score(30, 1), _score(120, 1), _score(150, 1), _score(300, 1)]
    segments = contiguous_segments(scores, fps=30.0, sample_every=30, min_segment_seconds=2.0, max_gap_seconds=2.0)
    assert len(segments) == 1
    assert segments[0]["start_frame"] == 0
    assert segments[0]["end_frame"] == 180
    assert segments[0]["label"] == MainViewLabel.MAIN_VIEW.value


def test_legacy_main_view_labels_are_normalized_at_the_compatibility_adapter() -> None:
    assert parse_main_view_label("MAIN_LIVE_VIEW") == MainViewLabel.MAIN_VIEW
    assert parse_main_view_label("MAIN_BIRDSEYE_LIVE") == MainViewLabel.MAIN_VIEW
    assert parse_main_view_label("MAIN_VIEW") == MainViewLabel.MAIN_VIEW
    with pytest.raises(ValueError, match="Unsupported Main View label"):
        parse_main_view_label("REPLAY")


def test_projection_quality_rejects_absurd_court_coordinates() -> None:
    rows = [
        {"smoothed_court_x": "6.10", "smoothed_court_y": "-20"},
        {"smoothed_court_x": "6.10", "smoothed_court_y": "25"},
        {"smoothed_court_x": "3.00", "smoothed_court_y": "6.00"},
    ]
    metrics = projection_quality(rows)
    assert metrics["absurd_y_ratio"] > 0.5
    assert choose_reject_reason(metrics, 0.75) == "court_projection_outlier"

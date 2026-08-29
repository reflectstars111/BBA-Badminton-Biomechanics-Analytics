from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np

from badminton_data_process.calibration.court import court_line_support
from badminton_data_process.calibration.geometry import project_point
from badminton_data_process.preprocess.timeline import smooth_main_view_segments
import badminton_data_process.rally.segmentation as rally_module
from badminton_data_process.core.io import write_json
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


def test_reference_court_rejects_misaligned_camera_view() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    corners = np.array([[20, 20], [80, 20], [80, 80], [20, 80]], dtype=np.float32)
    cv2.polylines(frame, [corners.astype(np.int32)], True, (255, 255, 255), 3)

    aligned = court_line_support(frame, corners)
    shifted = court_line_support(frame, corners + np.array([10, 10], dtype=np.float32))

    assert aligned > 0.2
    assert aligned > shifted * 3


def test_smooth_main_view_segments_filters_and_bridges_short_gap() -> None:
    segments = [
        {"start": 0.0, "end": 2.0, "label": "MAIN_LIVE_VIEW", "confidence": 0.7},
        {"start": 4.0, "end": 8.0, "label": "MAIN_LIVE_VIEW", "confidence": 0.8},
        {"start": 9.0, "end": 13.0, "label": "MAIN_LIVE_VIEW", "confidence": 0.75},
        {"start": 20.0, "end": 25.0, "label": "REPLAY", "confidence": 0.9},
    ]
    result = smooth_main_view_segments(segments, min_main_duration=3.0, max_gap=2.0)
    assert result == [
        {"start": 4.0, "end": 13.0, "label": "MAIN_VIEW", "confidence": 0.8}
    ]


def test_timeline_constrained_rallies_never_consume_frames_outside_main_view(
    tmp_path,
    monkeypatch,
) -> None:
    timeline_path = tmp_path / "timeline.json"
    write_json(
        timeline_path,
        [
            {
                "segment_id": "001",
                "start_frame": 100,
                "end_frame": 200,
                "label": "MAIN_VIEW",
            }
        ],
    )
    captured: dict[str, object] = {}

    def analyze_video(*_args, **_kwargs):
        rows = [
            {
                "sample_frame": float(frame),
                "is_rally_view": 1.0,
                "is_court_view": 1.0,
                "is_candidate": float(frame in {100, 150, 190}),
            }
            for frame in (90, 100, 150, 190, 200, 210)
        ]
        return rows, 30.0, 300, 1

    def write_rally_clips(_input, _output, segments, _fps, _match_id):
        captured["segments"] = segments
        return [
            {
                "rally_id": "001",
                "start_frame": segments[0][0],
                "end_frame": segments[0][1],
                "output_path": "rally.mp4",
            }
        ]

    fake_legacy = SimpleNamespace(
        extract_match_id=lambda _path: "match",
        remove_previous_outputs=lambda *_args: 0,
        write_rally_clips=write_rally_clips,
        write_analysis_csv=lambda _path, rows: captured.setdefault("analysis_rows", rows),
    )
    monkeypatch.setattr(rally_module, "_module", lambda: fake_legacy)
    monkeypatch.setattr(rally_module, "analyze_video", analyze_video)

    rally_module.segment_rallies_with_timeline(
        input_path=tmp_path / "match.mp4",
        timeline_path=timeline_path,
        output_dir=tmp_path / "rallies",
        metadata_csv=tmp_path / "rallies.csv",
        decisions_csv=tmp_path / "rally_decisions.csv",
        sample_every=10,
        min_rally_seconds=1.0,
        max_rally_seconds=30.0,
        max_gap_seconds=2.0,
        min_motion_score=0.01,
        max_motion_score=0.16,
        min_center_green_ratio=0.2,
        min_bottom_green_ratio=0.2,
        min_line_ratio=0.1,
        min_top_green_ratio=0.1,
        min_middle_green_ratio=0.1,
        max_left_right_green_diff=0.2,
        min_top_dark_ratio=0.1,
        min_middle_edge_ratio=0.1,
        pre_context_seconds=0.0,
        post_context_seconds=0.0,
        min_active_samples=2,
        overwrite=True,
    )

    assert [int(row["sample_frame"]) for row in captured["analysis_rows"]] == [
        100,
        150,
        190,
    ]
    assert captured["segments"] == [(100, 200)]


def test_usable_rally_classification_reports_explicit_rejection_reasons() -> None:
    rows = [
        {"sample_frame": float(frame), "is_candidate": float(frame in {30, 60})}
        for frame in range(0, 120, 30)
    ]

    accepted = rally_module.classify_usable_rallies(
        rows,
        source_segment_id="001",
        interval_start=0,
        interval_end=120,
        sample_every=30,
        fps=30.0,
        min_rally_seconds=1.0,
        max_rally_seconds=10.0,
        max_gap_seconds=1.0,
        pre_context_seconds=0.0,
        post_context_seconds=0.0,
        min_active_samples=2,
    )
    rejected = rally_module.classify_usable_rallies(
        [{"sample_frame": 30.0, "is_candidate": 0.0}],
        source_segment_id="002",
        interval_start=0,
        interval_end=60,
        sample_every=30,
        fps=30.0,
        min_rally_seconds=1.0,
        max_rally_seconds=10.0,
        max_gap_seconds=1.0,
        pre_context_seconds=0.0,
        post_context_seconds=0.0,
        min_active_samples=2,
    )

    assert accepted[0]["status"] == "accepted"
    assert accepted[0]["reason"] == "active_play_evidence"
    assert accepted[0]["frame_interval"] == "[start_frame,end_frame)"
    assert rejected[0]["status"] == "rejected"
    assert rejected[0]["reason"] == "no_active_play_evidence"


def test_rally_clip_writer_uses_half_open_frame_intervals(tmp_path) -> None:
    input_video = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(
        str(input_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (64, 48),
    )
    assert writer.isOpened()
    for frame_id in range(5):
        writer.write(np.full((48, 64, 3), frame_id * 20, dtype=np.uint8))
    writer.release()

    output_dir = tmp_path / "rallies"
    output_dir.mkdir()
    rows = rally_module._module().write_rally_clips(
        input_video,
        output_dir,
        [(1, 4)],
        10.0,
        "match",
    )

    capture = cv2.VideoCapture(rows[0]["output_path"])
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
    capture.release()
    assert rows[0]["start_frame"] == 1
    assert rows[0]["end_frame"] == 4
    assert rows[0]["duration_seconds"] == 0.3


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


def test_rally_motion_score_focuses_on_play_area_instead_of_full_frame() -> None:
    previous = np.zeros((180, 320), dtype=np.uint8)
    current = previous.copy()
    # A distant player occupies only a small part of a broadcast frame. The
    # full-frame mean falls below the production activity threshold even
    # though the change inside the playing area is significant.
    current[80:110, 145:175] = 100

    global_score, play_area_score = rally_module.frame_motion_scores(
        current,
        previous,
    )

    assert global_score < 0.008
    assert play_area_score >= 0.008


def test_exact_minimum_rally_frame_count_is_not_rejected_by_fractional_fps() -> None:
    result = rally_module.classify_usable_rallies(
        [
            {"sample_frame": 0.0, "is_candidate": 1.0},
            {"sample_frame": 30.0, "is_candidate": 1.0},
        ],
        source_segment_id="001",
        interval_start=0,
        interval_end=60,
        sample_every=30,
        fps=30.000033690979595,
        min_rally_seconds=2.0,
        max_rally_seconds=10.0,
        max_gap_seconds=1.0,
        pre_context_seconds=0.0,
        post_context_seconds=0.0,
        min_active_samples=2,
    )

    assert result[0]["status"] == "accepted"


def test_live_rally_is_last_court_view_before_score_update() -> None:
    view_segments = [(300, 600), (700, 900), (1000, 1300)]

    assert select_live_rallies(
        view_segments,
        score_change_frames=[620, 1320],
        fps=30.0,
    ) == [(300, 600), (1000, 1300)]

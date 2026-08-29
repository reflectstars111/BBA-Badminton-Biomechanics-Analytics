from __future__ import annotations

import csv

from badminton_data_process.core.io import read_csv_rows, write_csv_rows
from badminton_data_process.legacy import load_legacy_module
from badminton_data_process.pipeline.run import _analysis_mode_for_roles


def test_tactical_events_are_limited_to_tracked_player_roles() -> None:
    tactics = load_legacy_module("tactical_analysis.py")
    assert tactics.parse_float("1.5") == 1.5
    events = [
        {"event_type": "hit", "player_id": "near", "frame_id": 10},
        {"event_type": "hit", "player_id": "far", "frame_id": 20},
        {"event_type": "landing", "player_id": "far", "frame_id": 30},
    ]

    filtered = tactics.filter_events_for_tracked_players(events, {"near"})

    assert filtered == [events[0]]


def test_event_eligibility_rejects_near_only_and_missing_roles() -> None:
    tactics = load_legacy_module("tactical_analysis.py")

    assert tactics.resolve_event_eligibility("near_only", {"near"}) == (
        "near_only",
        "not_eligible",
        "near_only_analysis_does_not_support_hit_or_landing_events",
    )
    assert tactics.resolve_event_eligibility("experimental_two_player", {"near"}) == (
        "experimental_two_player",
        "not_eligible",
        "event_analysis_requires_near_and_far_player_tracks",
    )
    assert tactics.resolve_event_eligibility("auto", {"near", "far"}) == (
        "experimental_two_player",
        "experimental",
        "",
    )


def test_pipeline_mode_is_derived_from_configured_roles() -> None:
    assert _analysis_mode_for_roles(["near"]) == "near_only"
    assert _analysis_mode_for_roles(["near", "far"]) == "experimental_two_player"


def test_player_metrics_do_not_bridge_gaps_or_implausible_positions() -> None:
    tactics = load_legacy_module("tactical_analysis.py")
    rows = [
        {"frame_id": "0", "timestamp": "0.0", "court_x": "3.0", "court_y": "10.0"},
        {"frame_id": "1", "timestamp": "0.1", "court_x": "3.1", "court_y": "10.0"},
        {"frame_id": "4", "timestamp": "0.4", "court_x": "5.0", "court_y": "10.0"},
        {"frame_id": "5", "timestamp": "0.5", "court_x": "5.1", "court_y": "10.0"},
        {"frame_id": "6", "timestamp": "0.6", "court_x": "3.0", "court_y": "2.0"},
    ]

    metrics = tactics.player_metrics("near", rows)

    assert metrics is not None
    assert metrics["frames_valid"] == 4
    assert metrics["rejected_position_rows"] == 1
    assert metrics["distance_steps"] == 2
    assert metrics["discontinuity_count"] == 1
    assert metrics["movement_duration_seconds"] == 0.2
    assert metrics["total_distance_m"] == 0.2
    assert metrics["avg_speed_m_s"] == 1.0
    assert metrics["movement_eligibility"] == "eligible"


def test_player_metrics_report_unknown_instead_of_zero_without_a_valid_step() -> None:
    tactics = load_legacy_module("tactical_analysis.py")
    metrics = tactics.player_metrics(
        "near",
        [
            {"frame_id": "0", "timestamp": "0.0", "court_x": "3.0", "court_y": "10.0"},
            {"frame_id": "4", "timestamp": "0.4", "court_x": "3.0", "court_y": "10.0"},
        ],
    )

    assert metrics is not None
    assert metrics["movement_eligibility"] == "not_eligible"
    assert metrics["movement_reject_reason"] == "no_contiguous_plausible_movement_steps"
    assert metrics["total_distance_m"] == ""
    assert metrics["avg_speed_m_s"] == ""


def test_direction_reversal_detection_does_not_bridge_missing_frames() -> None:
    tactics = load_legacy_module("tactical_analysis.py")
    sparse = [
        {"frame_id": 0, "timestamp": 0.0, "image_x": 0.0, "image_y": 0.0},
        {"frame_id": 2, "timestamp": 0.2, "image_x": 10.0, "image_y": 0.0},
        {"frame_id": 3, "timestamp": 0.3, "image_x": 0.0, "image_y": 0.0},
    ]
    contiguous = [dict(point, frame_id=index) for index, point in enumerate(sparse)]

    assert tactics.detect_strikes(sparse, 90.0, 1) == []
    assert len(tactics.detect_strikes(contiguous, 90.0, 1)) == 1


def test_experimental_reversals_are_not_reported_as_hits_or_landings() -> None:
    tactics = load_legacy_module("tactical_analysis.py")
    shuttle_rows = [
        {"frame_id": "0", "timestamp": "0.0", "x": "90", "y": "100"},
        {"frame_id": "1", "timestamp": "0.1", "x": "100", "y": "100"},
        {"frame_id": "2", "timestamp": "0.2", "x": "90", "y": "100"},
    ]
    player_series = {
        "near": {
            "frames": [1],
            "image_x": [100.0],
            "image_y": [105.0],
            "court_x": [3.0],
            "court_y": [10.0],
        }
    }

    events, candidate_counts, landing_counts = tactics.analyze_rally_events(
        "rally.mp4",
        "rally",
        "001",
        player_series,
        shuttle_rows,
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        90.0,
        1,
        20.0,
    )

    assert [event["event_type"] for event in events] == ["reversal_candidate"]
    assert events[0]["court_x"] == ""
    assert events[0]["court_y"] == ""
    assert events[0]["event_eligibility"] == "experimental"
    assert candidate_counts == {"near": 1}
    assert landing_counts == {}


def test_shuttle_analysis_skips_rejected_tracker_interpolation() -> None:
    tactics = load_legacy_module("tactical_analysis.py")

    points = tactics.shuttle_image_points(
        [
            {
                "frame_id": "0",
                "timestamp": "0",
                "x": "10",
                "y": "20",
                "visibility": "1",
                "is_smoothed_valid": "1",
                "smoothed_x": "10",
                "smoothed_y": "20",
            },
            {
                "frame_id": "1",
                "timestamp": "0.033",
                "x": "500",
                "y": "5",
                "visibility": "0",
                "is_interpolated": "1",
                "is_smoothed_valid": "0",
                "smoothed_x": "",
                "smoothed_y": "",
            },
        ]
    )

    assert [point["frame_id"] for point in points] == [0]


def test_near_only_analysis_writes_not_eligible_without_generating_events(
    tmp_path,
    monkeypatch,
) -> None:
    tactics = load_legacy_module("tactical_analysis.py")
    player_csv = tmp_path / "players.csv"
    shuttle_csv = tmp_path / "shuttle.csv"
    output_dir = tmp_path / "tactics"
    calibration_dir = tmp_path / "calibration"
    calibration_dir.mkdir()
    write_csv_rows(
        player_csv,
        [
            "video_path",
            "video_stem",
            "rally_id",
            "player_id",
            "frame_id",
            "timestamp",
            "image_x",
            "image_y",
            "court_x",
            "court_y",
            "is_smoothed_valid",
        ],
        [
            {
                "video_path": "rally.mp4",
                "video_stem": "rally",
                "rally_id": "001",
                "player_id": "near",
                "frame_id": frame_id,
                "timestamp": frame_id / 30.0,
                "image_x": 100 + frame_id,
                "image_y": 200,
                "court_x": 3.0,
                "court_y": 10.0,
                "is_smoothed_valid": 0,
            }
            for frame_id in range(3)
        ],
    )
    write_csv_rows(
        shuttle_csv,
        ["video_path", "video_stem", "rally_id", "frame_id", "timestamp", "x", "y"],
        [
            {
                "video_path": "rally.mp4",
                "video_stem": "rally",
                "rally_id": "001",
                "frame_id": frame_id,
                "timestamp": frame_id / 30.0,
                "x": x,
                "y": y,
            }
            for frame_id, (x, y) in enumerate([(50, 50), (80, 80), (50, 50)])
        ],
    )

    def unexpected_event_analysis(*_args, **_kwargs):
        raise AssertionError("near-only mode must not call event analysis")

    monkeypatch.setattr(tactics, "analyze_rally_events", unexpected_event_analysis)
    result = tactics.analyze_tactics(
        player_csv,
        shuttle_csv,
        calibration_dir,
        output_dir,
        analysis_mode="near_only",
    )

    summaries = read_csv_rows(output_dir / "tactics_summary.csv")
    events = read_csv_rows(output_dir / "tactics_events.csv")
    assert result["not_eligible_rallies"] == 1
    assert result["event_rows"] == 0
    assert len(summaries) == 1
    assert summaries[0]["analysis_mode"] == "near_only"
    assert summaries[0]["event_eligibility"] == "not_eligible"
    assert summaries[0]["event_reject_reason"] == (
        "near_only_analysis_does_not_support_hit_or_landing_events"
    )
    assert summaries[0]["hit_count"] == ""
    assert summaries[0]["landing_count"] == ""
    assert summaries[0]["reversal_candidate_count"] == ""
    assert events == []
    with (output_dir / "tactics_events.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        assert "event_type" in (csv.DictReader(stream).fieldnames or [])

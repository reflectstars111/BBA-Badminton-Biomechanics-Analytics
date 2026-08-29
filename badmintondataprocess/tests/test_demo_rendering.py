from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from badminton_data_process.core.io import write_csv_rows, write_json
import badminton_data_process.visualization.demo as demo_module
from badminton_data_process.visualization.demo import group_rows_by_rally, render_demo
from badminton_data_process.tracking.player.pose import build_pose_observation


cv2 = pytest.importorskip("cv2")


def _write_video(path: Path, color: tuple[int, int, int], frames: int = 2) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (320, 240),
    )
    assert writer.isOpened()
    for _ in range(frames):
        writer.write(np.full((240, 320, 3), color, dtype=np.uint8))
    writer.release()


def test_group_rows_by_rally_keeps_reused_frame_ids_separate() -> None:
    grouped = group_rows_by_rally(
        [
            {"video_stem": "match_rally_001", "rally_id": "001", "frame_id": "0", "x": "10"},
            {"video_stem": "match_rally_002", "rally_id": "002", "frame_id": "0", "x": "20"},
        ]
    )
    assert grouped[("match_rally_001", "001")][0]["x"] == "10"
    assert grouped[("match_rally_002", "002")][0]["x"] == "20"


def test_player_marker_prefers_body_anchor_but_keeps_legacy_fallback() -> None:
    explicit = {
        "body_image_x": "100",
        "body_image_y": "120",
        "body_anchor_valid": "1",
        "image_x": "100",
        "image_y": "220",
    }
    assert demo_module.player_image_point(explicit) == (100.0, 120.0)
    assert demo_module.player_image_point({"image_x": "100", "image_y": "220"}) == (
        100.0,
        220.0,
    )


def test_demo_draws_pose_skeleton_from_track_artifact() -> None:
    pose = build_pose_observation(
        [(80 + index * 3, 50 + index * 6) for index in range(17)],
        [0.9] * 17,
        model_name="rtmpose",
        detection_confidence=0.8,
    )
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    demo_module._draw_players(
        frame,
        {
            "near": {
                "player_id": "near",
                "pose_valid": "1",
                "pose_keypoints_json": pose.to_json(),
                "pose_keypoint_threshold": "0.35",
                "body_image_x": "100",
                "body_image_y": "120",
                "body_anchor_valid": "1",
            }
        },
    )
    assert int(frame.sum()) > 0


def test_demo_labels_ineligible_events_as_unavailable(monkeypatch) -> None:
    labels: list[str] = []
    monkeypatch.setattr(
        demo_module,
        "_draw_label",
        lambda _frame, text, *_args, **_kwargs: labels.append(text),
    )

    demo_module._draw_stats(
        np.zeros((240, 320, 3), dtype=np.uint8),
        {
            "near": {
                "total_distance_m": "4.5",
                "avg_speed_m_s": "1.2",
                "hit_count": "",
                "event_eligibility": "not_eligible",
            }
        },
        "001",
        1,
        1,
    )

    assert any("events N/A" in label for label in labels)
    assert not any("hits 0" in label for label in labels)


def test_demo_labels_ineligible_movement_as_unavailable(monkeypatch) -> None:
    labels: list[str] = []
    monkeypatch.setattr(
        demo_module,
        "_draw_label",
        lambda _frame, text, *_args, **_kwargs: labels.append(text),
    )

    demo_module._draw_stats(
        np.zeros((240, 320, 3), dtype=np.uint8),
        {
            "near": {
                "movement_eligibility": "not_eligible",
                "total_distance_m": "",
                "avg_speed_m_s": "",
                "event_eligibility": "not_eligible",
            }
        },
        "001",
        1,
        1,
    )

    assert any("movement N/A" in label for label in labels)
    assert not any("0.0m" in label for label in labels)


def test_invalid_shuttle_sample_breaks_trail() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    trail: list[tuple[int, int]] = []

    demo_module._draw_shuttle(frame, {"x": "10", "y": "20"}, trail, 10)
    assert trail == [(10, 20)]

    demo_module._draw_shuttle(frame, None, trail, 10)
    assert trail == []


def test_rejected_tracker_interpolation_does_not_fall_back_to_raw_point() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    trail: list[tuple[int, int]] = []

    demo_module._draw_shuttle(frame, {"x": "10", "y": "20"}, trail, 10)
    demo_module._draw_shuttle(
        frame,
        {
            "x": "500",
            "y": "5",
            "visibility": "0",
            "is_interpolated": "1",
            "is_smoothed_valid": "0",
            "smoothed_x": "",
            "smoothed_y": "",
        },
        trail,
        10,
    )

    assert trail == []


def test_topdown_rejects_out_of_court_points_instead_of_clamping() -> None:
    rect = (10, 20, 100, 200)
    assert demo_module._topdown_pixel((3.05, 6.7), rect) == (60, 120)
    assert demo_module._topdown_pixel((-0.01, 6.7), rect) is None
    assert demo_module._topdown_pixel((3.05, 13.41), rect) is None


def test_topdown_uses_complete_regulation_badminton_markings() -> None:
    rect = (10, 20, 122, 268)
    segments = demo_module._court_marking_segments(rect)

    assert set(segments) == {
        "singles_left_sideline",
        "singles_right_sideline",
        "far_doubles_long_service",
        "near_doubles_long_service",
        "far_short_service",
        "near_short_service",
        "far_center",
        "near_center",
        "net",
    }
    net_y = segments["net"][0][1]
    assert segments["far_center"][1][1] < net_y
    assert segments["near_center"][0][1] > net_y
    assert segments["singles_left_sideline"][0][0] > rect[0]
    assert segments["singles_right_sideline"][0][0] < rect[0] + rect[2]


def test_full_rally_summary_only_appears_on_last_frame() -> None:
    assert not demo_module.should_show_full_rally_summary(0, 3)
    assert not demo_module.should_show_full_rally_summary(1, 3)
    assert demo_module.should_show_full_rally_summary(2, 3)
    assert not demo_module.should_show_full_rally_summary(0, 0)


def test_resized_calibration_maps_display_coordinates_back_to_source() -> None:
    calibration = {
        "homography_image_to_court": [
            [6.1 / 640.0, 0.0, 0.0],
            [0.0, 13.4 / 480.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    }
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    display_homography = demo_module._draw_court_outline(
        frame,
        calibration,
        scale_x=0.5,
        scale_y=0.5,
    )
    point = demo_module.image_to_court((160.0, 120.0), display_homography)

    assert point == pytest.approx((3.05, 6.7))


def test_render_demo_combines_rally_videos(tmp_path: Path, monkeypatch) -> None:
    rally_dir = tmp_path / "rallies"
    calibration_dir = tmp_path / "calibration"
    rally_dir.mkdir()
    calibration_dir.mkdir()
    videos = [rally_dir / "match_rally_001.mp4", rally_dir / "match_rally_002.mp4"]
    _write_video(videos[0], (20, 80, 20))
    _write_video(videos[1], (80, 20, 20))

    rallies = []
    players = []
    shuttles = []
    events = []
    summaries = []
    for index, video in enumerate(videos, start=1):
        rally_id = f"{index:03d}"
        rallies.append({"rally_id": rally_id, "output_path": str(video)})
        write_json(
            calibration_dir / f"{video.stem}.json",
            {
                "image_points_tl_tr_br_bl": [[40, 30], [280, 30], [300, 220], [20, 220]],
                "homography_image_to_court": [
                    [6.1 / 320.0, 0.0, 0.0],
                    [0.0, 13.4 / 240.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
            },
        )
        for frame_id in range(2):
            players.append(
                {
                    "video_stem": video.stem,
                    "rally_id": rally_id,
                    "frame_id": frame_id,
                    "player_id": "near",
                    "bbox_x1": "120",
                    "bbox_y1": "150",
                    "bbox_x2": "160",
                    "bbox_y2": "220",
                    "image_x": "140",
                    "image_y": "220",
                    "court_x": "3.0",
                    "court_y": "11.0",
                    "is_smoothed_valid": "0",
                }
            )
            shuttles.append(
                {
                    "video_stem": video.stem,
                    "rally_id": rally_id,
                    "frame_id": frame_id,
                    "x": str(170 + frame_id * 4),
                    "y": str(100 + frame_id * 4),
                    "is_smoothed_valid": "0",
                }
            )
        events.append(
            {
                "video_stem": video.stem,
                "rally_id": rally_id,
                "frame_id": "0",
                "event_type": "hit",
                "player_id": "near",
                "court_x": "3.0",
                "court_y": "10.0",
            }
        )
        summaries.append(
            {
                "video_stem": video.stem,
                "rally_id": rally_id,
                "player_id": "near",
                "total_distance_m": "4.5",
                "avg_speed_m_s": "1.2",
                "hit_count": "1",
            }
        )

    rallies_csv = tmp_path / "rallies.csv"
    players_csv = tmp_path / "players.csv"
    shuttles_csv = tmp_path / "shuttles.csv"
    events_csv = tmp_path / "events.csv"
    summaries_csv = tmp_path / "summaries.csv"
    write_csv_rows(rallies_csv, ["rally_id", "output_path"], rallies)
    write_csv_rows(players_csv, list(players[0]), players)
    write_csv_rows(shuttles_csv, list(shuttles[0]), shuttles)
    write_csv_rows(events_csv, list(events[0]), events)
    write_csv_rows(summaries_csv, list(summaries[0]), summaries)

    output = tmp_path / "demo.mp4"
    summary_calls: list[str] = []
    original_draw_stats = demo_module._draw_stats

    def record_draw_stats(frame, stats, rally_id, rally_index, rally_count):
        summary_calls.append(rally_id)
        original_draw_stats(frame, stats, rally_id, rally_index, rally_count)

    monkeypatch.setattr(demo_module, "_draw_stats", record_draw_stats)
    result = render_demo(
        rallies_csv=rallies_csv,
        player_tracks_csv=players_csv,
        shuttle_tracks_csv=shuttles_csv,
        calibration_dir=calibration_dir,
        tactics_events_csv=events_csv,
        tactics_summary_csv=summaries_csv,
        output_video=output,
    )

    assert result["rallies"] == 2
    assert result["frames"] == 4
    assert summary_calls == ["001", "002"]
    assert output.stat().st_size > 0
    capture = cv2.VideoCapture(str(output))
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 4
    capture.release()

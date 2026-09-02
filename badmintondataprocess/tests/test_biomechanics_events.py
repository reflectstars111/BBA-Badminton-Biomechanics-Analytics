from __future__ import annotations

import json

from badminton_data_process.analysis.biomechanics.events import detect_action_events
from badminton_data_process.tracking.player.pose import COCO_KEYPOINT_NAMES


def _pose(wrist_x: float) -> str:
    coordinates = {
        "left_shoulder": (90.0, 120.0),
        "right_shoulder": (110.0, 120.0),
        "left_hip": (92.0, 170.0),
        "right_hip": (108.0, 170.0),
        "left_wrist": (70.0, 130.0),
        "right_wrist": (wrist_x, 100.0),
    }
    return json.dumps(
        [
            {
                "name": name,
                "x": coordinates.get(name, (0.0, 0.0))[0],
                "y": coordinates.get(name, (0.0, 0.0))[1],
                "confidence": 0.95 if name in coordinates else 0.0,
            }
            for name in COCO_KEYPOINT_NAMES
        ]
    )


def _player(frame: int, wrist_x: float) -> dict[str, object]:
    return {
        "video_path": "rally.mp4",
        "video_stem": "rally",
        "rally_id": "001",
        "frame_id": frame,
        "timestamp": frame / 25.0,
        "player_id": "near",
        "bbox_y1": 80,
        "bbox_y2": 180,
        "pose_keypoints_json": _pose(wrist_x),
    }


def _shuttle(
    frame: int, x: float, *, interpolated: bool = False
) -> dict[str, object]:
    return {
        "video_path": "rally.mp4",
        "video_stem": "rally",
        "rally_id": "001",
        "frame_id": frame,
        "timestamp": frame / 25.0,
        "x": x,
        "y": 100,
        "confidence": 0.9,
        "visibility": 1,
        "is_interpolated": int(interpolated),
    }


def test_complete_multi_evidence_event_is_attributed_and_windowed() -> None:
    events = detect_action_events(
        [_player(9, 120), _player(10, 170), _player(11, 220)],
        [_shuttle(9, 150), _shuttle(10, 175), _shuttle(11, 150)],
    )

    assert len(events) == 1
    event = events[0]
    assert event["candidate_frame"] == 10
    assert event["player_id"] == "near"
    assert event["evidence_count"] == 3
    assert event["evidence_source"] == "shuttle_turn|shuttle_proximity|wrist_motion"
    assert event["window_start_frame"] == 0
    assert event["window_end_frame"] == 31
    assert event["classification_eligibility"] == "not_eligible"


def test_single_shuttle_turn_evidence_does_not_create_event() -> None:
    events = detect_action_events(
        [],
        [_shuttle(9, 150), _shuttle(10, 175), _shuttle(11, 150)],
    )

    assert events == []


def test_proximity_and_wrist_motion_without_turn_stay_below_release_gate() -> None:
    events = detect_action_events(
        [_player(9, 120), _player(10, 170), _player(11, 220)],
        [_shuttle(9, 170), _shuttle(10, 175), _shuttle(11, 180)],
    )

    assert events == []


def test_interpolated_shuttle_frame_is_not_used_as_turn_evidence() -> None:
    events = detect_action_events(
        [_player(9, 120), _player(10, 170), _player(11, 220)],
        [
            _shuttle(9, 150),
            _shuttle(10, 175, interpolated=True),
            _shuttle(11, 150),
        ],
    )

    assert events == []


def test_temporal_nms_keeps_only_best_candidate_in_minimum_gap() -> None:
    events = detect_action_events(
        [
            _player(9, 120),
            _player(10, 170),
            _player(11, 190),
            _player(12, 230),
        ],
        [
            _shuttle(9, 150),
            _shuttle(10, 175),
            _shuttle(11, 150),
            _shuttle(12, 180),
        ],
        min_event_gap_frames=8,
    )

    assert len(events) == 1
    assert events[0]["candidate_frame"] in {10, 11}

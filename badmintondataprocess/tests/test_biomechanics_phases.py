from __future__ import annotations

import json

from badminton_data_process.analysis.biomechanics.phases import decompose_swing_phases
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
        "video_stem": "rally",
        "rally_id": "001",
        "frame_id": frame,
        "timestamp": frame / 25.0,
        "player_id": "near",
        "bbox_y1": 80,
        "bbox_y2": 180,
        "pose_keypoints_json": _pose(wrist_x),
    }


def _event() -> dict[str, object]:
    return {
        "video_stem": "rally",
        "rally_id": "001",
        "event_id": "001_E001",
        "player_id": "near",
        "candidate_frame": 15,
        "candidate_score": 0.9,
        "window_start_frame": 0,
        "window_end_frame": 31,
    }


def _wrist_x(frame: int) -> float:
    if frame <= 7:
        return 120.0
    if frame <= 16:
        return 120.0 + (frame - 7) * 8.0
    tail = {17: 200.0, 18: 205.0, 19: 207.0, 20: 208.0}
    return tail.get(frame, 208.0)


def test_phase_decomposition_is_ordered_non_overlapping_and_contains_candidate() -> None:
    rows = [_player(frame, _wrist_x(frame)) for frame in range(31)]

    phases = decompose_swing_phases(rows, [_event()], smoothing_window=3)

    assert [row["phase"] for row in phases] == [
        "preparation",
        "acceleration",
        "contact_window",
        "follow_through",
        "recovery",
    ]
    for previous, current in zip(phases, phases[1:]):
        assert previous["end_frame"] == current["start_frame"]
    contact = next(row for row in phases if row["phase"] == "contact_window")
    assert contact["start_frame"] <= 15 < contact["end_frame"]
    assert all(row["phase_eligibility"] == "eligible" for row in phases)
    assert all(row["motion_side_candidate"] == "right" for row in phases)


def test_truncated_pose_window_is_explicitly_rejected() -> None:
    rows = [_player(frame, _wrist_x(frame)) for frame in range(14, 18)]

    phases = decompose_swing_phases(
        rows, [_event()], min_contiguous_frames=3, smoothing_window=3
    )

    assert len(phases) == 1
    assert phases[0]["phase_eligibility"] == "not_eligible"
    assert phases[0]["phase_reject_reason"] == "truncated_event_window"

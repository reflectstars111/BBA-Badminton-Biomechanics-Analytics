from __future__ import annotations

import json
from pathlib import Path

import pytest

from badminton_data_process.analysis.biomechanics.kinematics import (
    KINEMATICS_FRAME_FIELDS,
    analyze_kinematics_csv,
    build_kinematics_row,
    planar_angle_degrees,
)
from badminton_data_process.core.io import read_csv_rows, write_csv_rows
from badminton_data_process.tracking.player.pose import COCO_KEYPOINT_NAMES


def _pose(overrides: dict[str, tuple[float, float, float]]) -> str:
    return json.dumps(
        [
            {
                "name": name,
                "x": overrides.get(name, (0.0, 0.0, 0.0))[0],
                "y": overrides.get(name, (0.0, 0.0, 0.0))[1],
                "confidence": overrides.get(name, (0.0, 0.0, 0.0))[2],
            }
            for name in COCO_KEYPOINT_NAMES
        ]
    )


def _track_row(scale: float = 1.0, offset_x: float = 0.0, offset_y: float = 0.0):
    def point(x: float, y: float) -> tuple[float, float, float]:
        return offset_x + x * scale, offset_y + y * scale, 0.9

    return {
        "video_path": "rally.mp4",
        "video_stem": "rally",
        "rally_id": "001",
        "frame_id": 10,
        "timestamp": 0.4,
        "player_id": "near",
        "pose_model": "rtmpose",
        "pose_valid": 1,
        "bbox_y1": offset_y - 1.0 * scale,
        "bbox_y2": offset_y + 3.0 * scale,
        "pose_keypoints_json": _pose(
            {
                "left_shoulder": point(-0.5, 0.0),
                "right_shoulder": point(0.5, 0.0),
                "left_elbow": point(-1.5, 0.0),
                "right_elbow": point(1.5, 0.0),
                "left_wrist": point(-2.5, 0.0),
                "right_wrist": point(2.5, 0.0),
                "left_hip": point(-0.5, 1.0),
                "right_hip": point(0.5, 1.0),
                "left_knee": point(-0.5, 2.0),
                "right_knee": point(0.5, 2.0),
                "left_ankle": point(-1.0, 3.0),
                "right_ankle": point(1.0, 3.0),
            }
        ),
    }


def test_planar_angle_handles_right_straight_and_degenerate_geometry() -> None:
    assert planar_angle_degrees((1, 0), (0, 0), (0, 1)) == pytest.approx(90.0)
    assert planar_angle_degrees((-1, 0), (0, 0), (1, 0)) == pytest.approx(180.0)
    assert planar_angle_degrees((0, 0), (0, 0), (1, 0)) is None


def test_kinematics_are_translation_and_scale_invariant() -> None:
    base = build_kinematics_row(_track_row())
    transformed = build_kinematics_row(_track_row(scale=3.0, offset_x=80, offset_y=25))

    assert base["kinematics_eligibility"] == "eligible"
    assert base["left_elbow_angle_deg"] == pytest.approx(180.0)
    assert base["left_knee_angle_deg"] == pytest.approx(153.4349)
    assert base["trunk_lean_deg"] == pytest.approx(0.0)
    assert base["support_width_ratio"] == pytest.approx(0.5)
    assert base["body_support_offset_ratio"] == pytest.approx(0.0)
    for field in (
        "left_elbow_angle_deg",
        "left_knee_angle_deg",
        "trunk_lean_deg",
        "support_width_ratio",
        "body_support_offset_ratio",
    ):
        assert transformed[field] == pytest.approx(base[field])


def test_low_confidence_pose_is_explicitly_not_eligible() -> None:
    row = _track_row()
    row["pose_keypoints_json"] = _pose(
        {"left_shoulder": (10.0, 10.0, 0.2), "left_elbow": (20.0, 10.0, 0.2)}
    )

    result = build_kinematics_row(row, keypoint_threshold=0.35)

    assert result["kinematics_eligibility"] == "not_eligible"
    assert result["reject_reason"] == "insufficient_keypoint_coverage"
    assert result["left_elbow_angle_deg"] == ""


def test_kinematics_csv_preserves_rejected_rows_and_schema(tmp_path: Path) -> None:
    source = tmp_path / "player_tracks.csv"
    output = tmp_path / "kinematics_frames.csv"
    valid = _track_row()
    invalid = _track_row()
    invalid.update({"frame_id": 11, "pose_valid": 0, "pose_keypoints_json": ""})
    write_csv_rows(source, list(valid), [valid, invalid])

    summary = analyze_kinematics_csv(source, output)
    rows = read_csv_rows(output)

    assert summary["rows"] == 2
    assert summary["eligible_rows"] == 1
    assert summary["rejected_rows"] == 1
    assert list(rows[0]) == KINEMATICS_FRAME_FIELDS
    assert rows[1]["kinematics_eligibility"] == "not_eligible"
    assert rows[1]["reject_reason"] == "pose_not_valid"

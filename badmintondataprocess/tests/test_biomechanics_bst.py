from __future__ import annotations

import json

import pytest

from badminton_data_process.analysis.biomechanics.bst import (
    BST_OFFICIAL_PROFILES,
    bst_class_labels,
    build_bst_input,
)
from badminton_data_process.tracking.player.pose import COCO_KEYPOINT_NAMES


def _pose(offset: float) -> str:
    return json.dumps(
        [
            {
                "name": name,
                "x": 100.0 + offset + index,
                "y": 80.0 + index * 2,
                "confidence": 0.9,
            }
            for index, name in enumerate(COCO_KEYPOINT_NAMES)
        ]
    )


def _player(frame: int, role: str) -> dict[str, object]:
    return {
        "video_stem": "rally",
        "rally_id": "001",
        "frame_id": frame,
        "player_id": role,
        "bbox_x1": 50,
        "bbox_y1": 50,
        "bbox_x2": 250,
        "bbox_y2": 250,
        "court_x": 2.0 if role == "far" else 4.0,
        "court_y": 3.0 if role == "far" else 10.0,
        "pose_keypoints_json": _pose(0 if role == "far" else 20),
    }


def _shuttle(frame: int) -> dict[str, object]:
    return {
        "video_stem": "rally",
        "rally_id": "001",
        "frame_id": frame,
        "x": 320 + frame,
        "y": 180,
        "visibility": 1,
        "is_interpolated": 0,
    }


def test_bst_adapter_matches_official_dual_player_joint_bone_shapes() -> None:
    event = {
        "window_start_frame": 0,
        "window_end_frame": 20,
        "player_id": "near",
    }
    players = [
        _player(frame, role)
        for frame in range(20)
        for role in ("far", "near")
    ]

    sample = build_bst_input(
        event,
        players,
        [_shuttle(frame) for frame in range(20)],
        frame_width=640,
        frame_height=360,
        seq_len=30,
        pose_style="JnB_bone",
    )

    assert sample.eligibility == "eligible"
    assert sample.human_pose.shape == (30, 2, 36, 2)
    assert sample.position.shape == (30, 2, 2)
    assert sample.shuttle.shape == (30, 2)
    assert sample.valid_length == 20
    assert sample.candidate_pose_coverage == 1.0
    assert sample.opponent_pose_coverage == 1.0
    assert sample.shuttle_coverage == 1.0
    assert sample.position[0, 0, 0] == pytest.approx(2.0 / 6.10)
    assert sample.position[0, 1, 1] == pytest.approx(10.0 / 13.40)


def test_bst_adapter_rejects_missing_opponent_instead_of_fabricating_pose() -> None:
    event = {
        "window_start_frame": 0,
        "window_end_frame": 20,
        "player_id": "near",
    }

    sample = build_bst_input(
        event,
        [_player(frame, "near") for frame in range(20)],
        [_shuttle(frame) for frame in range(20)],
        frame_width=640,
        frame_height=360,
    )

    assert sample.eligibility == "not_eligible"
    assert sample.reject_reason == "insufficient_opponent_pose_coverage"
    assert sample.opponent_pose_coverage == 0.0


def test_bst_class_order_matches_official_merged_contract() -> None:
    labels = bst_class_labels(25)

    assert len(labels) == 25
    assert labels[0] == "未知球種"
    assert labels[1] == "Top_放小球"
    assert labels[12] == "Top_發長球"
    assert labels[13] == "Bottom_放小球"


def test_recommended_official_profile_is_an_exact_checkpoint_contract() -> None:
    profile = BST_OFFICIAL_PROFILES["shuttleset_merged_seq30_balanced"]

    assert profile["model_name"] == "BST_AP"
    assert profile["pose_style"] == "JnB_bone"
    assert profile["seq_len"] == 30
    assert profile["num_classes"] == 25
    assert profile["weight_filename"] == "bst_AP_JnB_bone_merged_3.pt"
    assert profile["official_macro_f1"] == pytest.approx(0.814)

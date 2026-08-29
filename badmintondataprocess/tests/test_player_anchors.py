from __future__ import annotations

import pytest

from badminton_data_process.tracking.player.anchors import (
    AnchorSource,
    adapt_legacy_track_row,
    anchors_from_bbox,
    anchors_from_observation,
)


def test_bbox_produces_distinct_body_and_ground_anchors() -> None:
    anchors = anchors_from_bbox((100, 50, 140, 250), 0.8)

    assert (anchors.body_center.x, anchors.body_center.y) == pytest.approx((120, 140))
    assert (anchors.ground_contact.x, anchors.ground_contact.y) == pytest.approx((120, 250))
    assert anchors.body_center.source == AnchorSource.BBOX_TORSO
    assert anchors.ground_contact.source == AnchorSource.BBOX_FEET


def test_predicted_bbox_has_explicit_source_and_reduced_confidence() -> None:
    anchors = anchors_from_bbox((10, 20, 30, 60), 0.6, interpolated=True)

    assert anchors.body_center.source == AnchorSource.PREDICTED_BBOX_TORSO
    assert anchors.ground_contact.source == AnchorSource.PREDICTED_BBOX_FEET
    assert anchors.body_center.confidence == pytest.approx(0.3)


def test_invalid_bbox_is_rejected() -> None:
    with pytest.raises(ValueError, match="bbox ordering"):
        anchors_from_bbox((20, 10, 5, 30), 0.8)


def test_pose_torso_and_two_ankles_are_preferred() -> None:
    anchors = anchors_from_observation(
        (80, 40, 140, 240),
        0.7,
        keypoints={
            "left_shoulder": (95, 80, 0.9),
            "right_shoulder": (125, 80, 0.8),
            "left_hip": (100, 145, 0.85),
            "right_hip": (120, 145, 0.8),
            "left_ankle": (102, 238, 0.9),
            "right_ankle": (122, 240, 0.8),
        },
    )
    assert anchors is not None
    assert anchors.body_center.source == AnchorSource.POSE_TORSO
    assert anchors.ground_contact.source == AnchorSource.POSE_ANKLES


def test_single_ankle_is_valid_but_quality_is_reduced() -> None:
    anchors = anchors_from_observation(
        (80, 40, 140, 240),
        0.7,
        keypoints={"left_ankle": (102, 238, 0.8)},
    )
    assert anchors is not None
    assert anchors.ground_contact.source == AnchorSource.POSE_SINGLE_ANKLE
    assert anchors.ground_contact.confidence == pytest.approx(0.6)


def test_complete_miss_remains_missing() -> None:
    assert anchors_from_observation(None, 0.0, keypoints={}) is None


def test_legacy_adapter_preserves_ground_semantics() -> None:
    adapted = adapt_legacy_track_row({"image_x": "100", "image_y": "220", "confidence": "0.8"})
    assert adapted["ground_image_x"] == "100"
    assert adapted["ground_image_y"] == "220"
    assert adapted["ground_anchor_source"] == AnchorSource.LEGACY_IMAGE_XY.value
    assert adapted["body_anchor_valid"] == "0"

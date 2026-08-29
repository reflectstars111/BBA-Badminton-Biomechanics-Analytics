from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AnchorSource(str, Enum):
    """How an image-space player anchor was obtained."""

    BBOX_TORSO = "bbox_torso"
    BBOX_FEET = "bbox_feet"
    PREDICTED_BBOX_TORSO = "predicted_bbox_torso"
    PREDICTED_BBOX_FEET = "predicted_bbox_feet"
    POSE_TORSO = "pose_torso"
    POSE_ANKLES = "pose_ankles"
    POSE_SINGLE_ANKLE = "pose_single_ankle"
    LEGACY_IMAGE_XY = "legacy_image_xy"


@dataclass(frozen=True, slots=True)
class ImageAnchor:
    x: float
    y: float
    source: AnchorSource
    confidence: float
    valid: bool = True


@dataclass(frozen=True, slots=True)
class PlayerAnchors:
    """Separate presentation and metric anchors for one player Observation."""

    body_center: ImageAnchor
    ground_contact: ImageAnchor


def anchors_from_bbox(
    bbox: tuple[int, int, int, int] | tuple[float, float, float, float],
    confidence: float,
    *,
    interpolated: bool = False,
    torso_height_ratio: float = 0.45,
) -> PlayerAnchors:
    """Derive explicit anchors without pretending bbox geometry is pose output.

    The torso anchor is used for labels and trails. The feet/bottom anchor is
    the only point that may be projected to metric court coordinates.
    """

    if not 0.0 <= torso_height_ratio <= 1.0:
        raise ValueError("torso_height_ratio must be between 0 and 1")
    x1, y1, x2, y2 = (float(value) for value in bbox)
    if x2 < x1 or y2 < y1:
        raise ValueError(f"invalid bbox ordering: {bbox!r}")
    center_x = (x1 + x2) / 2.0
    body_source = (
        AnchorSource.PREDICTED_BBOX_TORSO if interpolated else AnchorSource.BBOX_TORSO
    )
    ground_source = (
        AnchorSource.PREDICTED_BBOX_FEET if interpolated else AnchorSource.BBOX_FEET
    )
    bounded_confidence = min(max(float(confidence), 0.0), 1.0)
    if interpolated:
        bounded_confidence *= 0.5
    return PlayerAnchors(
        body_center=ImageAnchor(
            x=center_x,
            y=y1 + (y2 - y1) * torso_height_ratio,
            source=body_source,
            confidence=bounded_confidence,
        ),
        ground_contact=ImageAnchor(
            x=center_x,
            y=y2,
            source=ground_source,
            confidence=bounded_confidence,
        ),
    )


def anchors_from_observation(
    bbox: tuple[int, int, int, int] | tuple[float, float, float, float] | None,
    detection_confidence: float,
    *,
    keypoints: dict[str, tuple[float, float, float]] | None = None,
    keypoint_threshold: float = 0.35,
    interpolated: bool = False,
) -> PlayerAnchors | None:
    """Prefer trustworthy pose anchors and degrade explicitly to bbox anchors."""

    fallback = (
        anchors_from_bbox(bbox, detection_confidence, interpolated=interpolated)
        if bbox is not None
        else None
    )
    points = keypoints or {}

    def valid(names: tuple[str, ...]) -> list[tuple[float, float, float]]:
        return [
            points[name]
            for name in names
            if name in points and float(points[name][2]) >= keypoint_threshold
        ]

    torso_points = valid(("left_shoulder", "right_shoulder", "left_hip", "right_hip"))
    ankle_points = valid(("left_ankle", "right_ankle"))

    if len(torso_points) >= 2:
        body = ImageAnchor(
            x=sum(point[0] for point in torso_points) / len(torso_points),
            y=sum(point[1] for point in torso_points) / len(torso_points),
            source=AnchorSource.POSE_TORSO,
            confidence=sum(point[2] for point in torso_points) / len(torso_points),
        )
    elif fallback is not None:
        body = fallback.body_center
    else:
        return None

    if len(ankle_points) == 2:
        ground = ImageAnchor(
            x=sum(point[0] for point in ankle_points) / 2.0,
            y=sum(point[1] for point in ankle_points) / 2.0,
            source=AnchorSource.POSE_ANKLES,
            confidence=sum(point[2] for point in ankle_points) / 2.0,
        )
    elif len(ankle_points) == 1:
        point = ankle_points[0]
        ground = ImageAnchor(
            x=point[0],
            y=point[1],
            source=AnchorSource.POSE_SINGLE_ANKLE,
            confidence=float(point[2]) * 0.75,
        )
    elif fallback is not None:
        ground = fallback.ground_contact
    else:
        return None
    return PlayerAnchors(body_center=body, ground_contact=ground)


def adapt_legacy_track_row(row: dict[str, str]) -> dict[str, str]:
    """Versioned Adapter: legacy image_x/y always mean ground contact."""

    if row.get("ground_image_x") not in (None, ""):
        return dict(row)
    adapted = dict(row)
    adapted.update(
        {
            "schema_version": "2.0-adapted-legacy",
            "ground_image_x": row.get("image_x", ""),
            "ground_image_y": row.get("image_y", ""),
            "ground_anchor_source": AnchorSource.LEGACY_IMAGE_XY.value,
            "ground_anchor_confidence": row.get("confidence", ""),
            "ground_anchor_valid": "1" if row.get("image_x") and row.get("image_y") else "0",
            "body_image_x": "",
            "body_image_y": "",
            "body_anchor_source": "",
            "body_anchor_confidence": "",
            "body_anchor_valid": "0",
        }
    )
    return adapted

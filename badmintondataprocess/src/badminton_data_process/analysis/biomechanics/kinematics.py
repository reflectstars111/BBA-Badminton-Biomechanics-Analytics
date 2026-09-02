from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Mapping

from badminton_data_process.core.io import read_csv_rows, write_csv_rows
from badminton_data_process.tracking.player.pose import (
    COCO_KEYPOINT_NAMES,
    PoseKeypoint,
    pose_keypoints_from_json,
)


KINEMATICS_METRIC_VERSION = "bba_2d_kinematics_v1"

KINEMATICS_FRAME_FIELDS = [
    "video_path",
    "video_stem",
    "rally_id",
    "frame_id",
    "timestamp",
    "player_id",
    "pose_model",
    "pose_valid",
    "kinematics_eligibility",
    "reject_reason",
    "keypoint_coverage_ratio",
    "mean_keypoint_confidence",
    "left_elbow_angle_deg",
    "right_elbow_angle_deg",
    "left_shoulder_angle_deg",
    "right_shoulder_angle_deg",
    "left_hip_angle_deg",
    "right_hip_angle_deg",
    "left_knee_angle_deg",
    "right_knee_angle_deg",
    "trunk_lean_deg",
    "support_width_ratio",
    "body_support_offset_ratio",
    "metric_version",
]


def _float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _true(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def planar_angle_degrees(
    first: tuple[float, float],
    vertex: tuple[float, float],
    third: tuple[float, float],
) -> float | None:
    """Return the unsigned 2-D angle at ``vertex`` in the closed range [0, 180]."""

    vector_a = (first[0] - vertex[0], first[1] - vertex[1])
    vector_b = (third[0] - vertex[0], third[1] - vertex[1])
    norm_a = math.hypot(*vector_a)
    norm_b = math.hypot(*vector_b)
    if norm_a <= 1e-9 or norm_b <= 1e-9:
        return None
    cosine = (vector_a[0] * vector_b[0] + vector_a[1] * vector_b[1]) / (
        norm_a * norm_b
    )
    return math.degrees(math.acos(min(1.0, max(-1.0, cosine))))


def _valid_points(
    keypoints: Iterable[PoseKeypoint],
    threshold: float,
) -> dict[str, PoseKeypoint]:
    return {
        point.name: point
        for point in keypoints
        if point.confidence >= threshold
        and math.isfinite(point.x)
        and math.isfinite(point.y)
    }


def _angle(
    points: Mapping[str, PoseKeypoint],
    first: str,
    vertex: str,
    third: str,
) -> float | None:
    if any(name not in points for name in (first, vertex, third)):
        return None
    return planar_angle_degrees(
        (points[first].x, points[first].y),
        (points[vertex].x, points[vertex].y),
        (points[third].x, points[third].y),
    )


def _midpoint(
    points: Mapping[str, PoseKeypoint],
    first: str,
    second: str,
) -> tuple[float, float] | None:
    if first not in points or second not in points:
        return None
    return (
        (points[first].x + points[second].x) / 2.0,
        (points[first].y + points[second].y) / 2.0,
    )


def _trunk_lean(points: Mapping[str, PoseKeypoint]) -> float | None:
    shoulders = _midpoint(points, "left_shoulder", "right_shoulder")
    hips = _midpoint(points, "left_hip", "right_hip")
    if shoulders is None or hips is None:
        return None
    dx = shoulders[0] - hips[0]
    dy_up = hips[1] - shoulders[1]
    if math.hypot(dx, dy_up) <= 1e-9:
        return None
    # Image x-right is positive. Zero is an upright trunk in the image plane.
    return math.degrees(math.atan2(dx, dy_up))


def _bbox_height(row: Mapping[str, object]) -> float | None:
    top = _float(row.get("bbox_y1"))
    bottom = _float(row.get("bbox_y2"))
    if top is None or bottom is None or bottom - top <= 1e-9:
        return None
    return bottom - top


def _stability_primitives(
    points: Mapping[str, PoseKeypoint],
    bbox_height: float | None,
) -> tuple[float | None, float | None]:
    ankles = _midpoint(points, "left_ankle", "right_ankle")
    shoulders = _midpoint(points, "left_shoulder", "right_shoulder")
    hips = _midpoint(points, "left_hip", "right_hip")
    if bbox_height is None or ankles is None:
        return None, None

    support_width = abs(points["left_ankle"].x - points["right_ankle"].x) / bbox_height
    if shoulders is None or hips is None:
        return support_width, None
    body_center_x = (shoulders[0] + hips[0]) / 2.0
    return support_width, (body_center_x - ankles[0]) / bbox_height


def _rounded(value: float | None) -> float | str:
    return round(value, 4) if value is not None and math.isfinite(value) else ""


def build_kinematics_row(
    track_row: Mapping[str, object],
    *,
    keypoint_threshold: float = 0.35,
    min_keypoint_coverage_ratio: float = 0.35,
) -> dict[str, object]:
    """Build one traceable 2-D kinematics row from an existing player track row."""

    keypoints = pose_keypoints_from_json(str(track_row.get("pose_keypoints_json") or ""))
    points = _valid_points(keypoints, keypoint_threshold)
    coverage = len(points) / len(COCO_KEYPOINT_NAMES)
    confidence = (
        sum(point.confidence for point in points.values()) / len(points) if points else None
    )
    metrics: dict[str, float | None] = {
        "left_elbow_angle_deg": _angle(
            points, "left_shoulder", "left_elbow", "left_wrist"
        ),
        "right_elbow_angle_deg": _angle(
            points, "right_shoulder", "right_elbow", "right_wrist"
        ),
        "left_shoulder_angle_deg": _angle(
            points, "left_hip", "left_shoulder", "left_elbow"
        ),
        "right_shoulder_angle_deg": _angle(
            points, "right_hip", "right_shoulder", "right_elbow"
        ),
        "left_hip_angle_deg": _angle(
            points, "left_shoulder", "left_hip", "left_knee"
        ),
        "right_hip_angle_deg": _angle(
            points, "right_shoulder", "right_hip", "right_knee"
        ),
        "left_knee_angle_deg": _angle(
            points, "left_hip", "left_knee", "left_ankle"
        ),
        "right_knee_angle_deg": _angle(
            points, "right_hip", "right_knee", "right_ankle"
        ),
        "trunk_lean_deg": _trunk_lean(points),
    }
    support_width, body_support_offset = _stability_primitives(
        points, _bbox_height(track_row)
    )
    metrics["support_width_ratio"] = support_width
    metrics["body_support_offset_ratio"] = body_support_offset

    pose_valid = _true(track_row.get("pose_valid"))
    if not pose_valid:
        eligibility = "not_eligible"
        reject_reason = "pose_not_valid"
    elif not keypoints:
        eligibility = "not_eligible"
        reject_reason = "missing_pose_keypoints"
    elif coverage < min_keypoint_coverage_ratio:
        eligibility = "not_eligible"
        reject_reason = "insufficient_keypoint_coverage"
    elif not any(value is not None for value in metrics.values()):
        eligibility = "not_eligible"
        reject_reason = "no_computable_kinematics"
    else:
        eligibility = "eligible"
        reject_reason = ""

    result: dict[str, object] = {
        "video_path": track_row.get("video_path", ""),
        "video_stem": track_row.get("video_stem", ""),
        "rally_id": track_row.get("rally_id", ""),
        "frame_id": track_row.get("frame_id", ""),
        "timestamp": track_row.get("timestamp", ""),
        "player_id": track_row.get("player_id", ""),
        "pose_model": track_row.get("pose_model", ""),
        "pose_valid": int(pose_valid),
        "kinematics_eligibility": eligibility,
        "reject_reason": reject_reason,
        "keypoint_coverage_ratio": round(coverage, 4),
        "mean_keypoint_confidence": _rounded(confidence),
        "metric_version": KINEMATICS_METRIC_VERSION,
    }
    result.update({name: _rounded(value) for name, value in metrics.items()})
    return result


def analyze_kinematics_csv(
    player_tracks_csv: Path,
    output_csv: Path,
    *,
    keypoint_threshold: float = 0.35,
    min_keypoint_coverage_ratio: float = 0.35,
) -> dict[str, object]:
    rows = [
        build_kinematics_row(
            row,
            keypoint_threshold=keypoint_threshold,
            min_keypoint_coverage_ratio=min_keypoint_coverage_ratio,
        )
        for row in read_csv_rows(player_tracks_csv)
    ]
    write_csv_rows(output_csv, KINEMATICS_FRAME_FIELDS, rows)
    eligible_rows = sum(row["kinematics_eligibility"] == "eligible" for row in rows)
    return {
        "rows": len(rows),
        "eligible_rows": eligible_rows,
        "rejected_rows": len(rows) - eligible_rows,
        "output_csv": str(output_csv),
        "metric_version": KINEMATICS_METRIC_VERSION,
    }

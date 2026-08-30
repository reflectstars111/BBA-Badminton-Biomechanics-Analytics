from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

import cv2
import numpy as np

from badminton_data_process.calibration.reference import (
    STANDARD_COURT,
    CourtGeometry,
    LineSupportReport,
    score_court_line_support,
)


class CalibrationCandidateSource(str, Enum):
    MANUAL = "manual"
    GREEN_CONTOUR = "green_contour"
    HOUGH_LINES = "hough_lines"


@dataclass(slots=True)
class CalibrationCandidate:
    corners: np.ndarray
    source: CalibrationCandidateSource
    frame_index: int
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.corners = np.asarray(self.corners, dtype=np.float32)


@dataclass(frozen=True, slots=True)
class CalibrationThresholds:
    min_area_ratio: float = 0.08
    min_line_support: float = 0.15
    max_condition_number: float = 1.0e10
    max_reprojection_error_px: float = 1.0
    max_out_of_bounds_ratio: float = 0.0


@dataclass(slots=True)
class CalibrationQuality:
    area_ratio: float = 0.0
    line_support: LineSupportReport | None = None
    condition_number: float | None = None
    reprojection_error_px: float | None = None
    out_of_bounds_ratio: float = 1.0
    quality_score: float = 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "area_ratio": round(self.area_ratio, 6),
            "line_support": self.line_support.as_dict() if self.line_support else None,
            "condition_number": self.condition_number,
            "reprojection_error_px": self.reprojection_error_px,
            "out_of_bounds_ratio": round(self.out_of_bounds_ratio, 6),
            "quality_score": round(self.quality_score, 6),
        }


@dataclass(slots=True)
class CalibrationValidationResult:
    candidate: CalibrationCandidate
    accepted: bool
    reasons: list[str]
    quality: CalibrationQuality
    homography_image_to_court: np.ndarray | None

    def as_dict(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "reasons": list(self.reasons),
            "source": self.candidate.source.value,
            "frame_index": self.candidate.frame_index,
            "image_points_tl_tr_br_bl": self.candidate.corners.astype(float).tolist(),
            "homography_image_to_court": (
                self.homography_image_to_court.astype(float).tolist()
                if self.homography_image_to_court is not None
                else None
            ),
            "quality": self.quality.as_dict(),
            "diagnostics": self.candidate.diagnostics,
        }


def _polygon_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))) / 2.0


def _ordered_tl_tr_br_bl(corners: np.ndarray) -> bool:
    top_left, top_right, bottom_right, bottom_left = corners
    return bool(
        top_left[0] < top_right[0]
        and bottom_left[0] < bottom_right[0]
        and (top_left[1] + top_right[1]) < (bottom_left[1] + bottom_right[1])
        and (top_left[0] + bottom_left[0]) < (top_right[0] + bottom_right[0])
    )


def validate_calibration_candidate(
    frame: np.ndarray,
    candidate: CalibrationCandidate,
    *,
    thresholds: CalibrationThresholds = CalibrationThresholds(),
    geometry: CourtGeometry = STANDARD_COURT,
) -> CalibrationValidationResult:
    reasons: list[str] = []
    quality = CalibrationQuality()
    corners = candidate.corners
    height, width = frame.shape[:2]

    if corners.shape != (4, 2):
        reasons.append("corners_shape")
        return CalibrationValidationResult(candidate, False, reasons, quality, None)
    if not np.all(np.isfinite(corners)):
        reasons.append("corners_non_finite")
        return CalibrationValidationResult(candidate, False, reasons, quality, None)

    inside = (
        (corners[:, 0] >= 0)
        & (corners[:, 0] < width)
        & (corners[:, 1] >= 0)
        & (corners[:, 1] < height)
    )
    quality.out_of_bounds_ratio = 1.0 - float(np.mean(inside))
    if quality.out_of_bounds_ratio > thresholds.max_out_of_bounds_ratio:
        reasons.append("corners_out_of_bounds")
    if not cv2.isContourConvex(corners.astype(np.float32)):
        reasons.append("corners_not_convex")
    if not _ordered_tl_tr_br_bl(corners):
        reasons.append("corners_wrong_order")

    quality.area_ratio = _polygon_area(corners) / max(1.0, float(width * height))
    if quality.area_ratio < thresholds.min_area_ratio:
        reasons.append("court_area_too_small")

    homography: np.ndarray | None = None
    if not reasons or all(reason not in {"corners_shape", "corners_non_finite"} for reason in reasons):
        try:
            homography = cv2.getPerspectiveTransform(corners, geometry.corners_array())
            normalized = homography / homography[2, 2] if homography[2, 2] else homography
            quality.condition_number = float(np.linalg.cond(normalized))
            if not np.isfinite(quality.condition_number) or quality.condition_number > thresholds.max_condition_number:
                reasons.append("homography_ill_conditioned")

            inverse = np.linalg.inv(homography)
            court_points = geometry.corners_array().reshape(1, -1, 2)
            reprojected = cv2.perspectiveTransform(court_points, inverse)[0]
            quality.reprojection_error_px = float(
                np.sqrt(np.mean(np.sum((reprojected - corners) ** 2, axis=1)))
            )
            if quality.reprojection_error_px > thresholds.max_reprojection_error_px:
                reasons.append("corner_reprojection_error")
        except (cv2.error, np.linalg.LinAlgError, ValueError):
            homography = None
            reasons.append("homography_invalid")

    try:
        quality.line_support = score_court_line_support(frame, corners, geometry=geometry)
        if quality.line_support.score < thresholds.min_line_support:
            reasons.append("court_line_support_too_low")
    except (cv2.error, ValueError):
        reasons.append("court_line_support_failed")

    line_score = quality.line_support.score if quality.line_support else 0.0
    area_score = min(1.0, quality.area_ratio / max(thresholds.min_area_ratio * 3.0, 1e-6))
    quality.quality_score = 0.8 * line_score + 0.2 * area_score
    reasons = list(dict.fromkeys(reasons))
    return CalibrationValidationResult(
        candidate=candidate,
        accepted=not reasons and homography is not None,
        reasons=reasons,
        quality=quality,
        homography_image_to_court=homography,
    )


def normalized_corner_rmse(
    first: Iterable[Iterable[float]],
    second: Iterable[Iterable[float]],
    image_shape: tuple[int, ...],
) -> float:
    first_points = np.asarray(first, dtype=np.float32)
    second_points = np.asarray(second, dtype=np.float32)
    if first_points.shape != (4, 2) or second_points.shape != (4, 2):
        raise ValueError("corner arrays must have shape (4, 2)")
    height, width = image_shape[:2]
    scale = np.asarray([max(1, width), max(1, height)], dtype=np.float32)
    delta = (first_points - second_points) / scale
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))


def select_stable_calibration(
    results: list[CalibrationValidationResult],
    image_shape: tuple[int, ...],
    *,
    max_corner_rmse_ratio: float = 0.04,
    min_stable_candidates: int = 2,
) -> tuple[CalibrationValidationResult | None, list[int]]:
    accepted = [result for result in results if result.accepted]
    if not accepted:
        return None, []
    if len(accepted) < min_stable_candidates:
        return None, [result.candidate.frame_index for result in accepted]

    clusters: list[list[CalibrationValidationResult]] = []
    for result in accepted:
        placed = False
        for cluster in clusters:
            if normalized_corner_rmse(
                result.candidate.corners,
                cluster[0].candidate.corners,
                image_shape,
            ) <= max_corner_rmse_ratio:
                same_frame_index = next(
                    (
                        index
                        for index, item in enumerate(cluster)
                        if item.candidate.frame_index == result.candidate.frame_index
                    ),
                    None,
                )
                if same_frame_index is None:
                    cluster.append(result)
                elif result.quality.quality_score > cluster[same_frame_index].quality.quality_score:
                    cluster[same_frame_index] = result
                placed = True
                break
        if not placed:
            clusters.append([result])

    best_cluster = max(
        clusters,
        key=lambda cluster: (len(cluster), sum(item.quality.quality_score for item in cluster)),
    )
    if len(best_cluster) < min_stable_candidates:
        return None, [item.candidate.frame_index for item in best_cluster]
    selected = max(best_cluster, key=lambda item: item.quality.quality_score)
    return selected, [item.candidate.frame_index for item in best_cluster]

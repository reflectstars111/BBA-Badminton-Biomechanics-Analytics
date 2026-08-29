from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import cv2
import numpy as np

from badminton_data_process.calibration.reference import (
    bright_line_mask,
    score_court_line_support,
)
from badminton_data_process.calibration.validation import (
    CalibrationCandidate,
    CalibrationCandidateSource,
)


@dataclass(frozen=True, slots=True)
class LineSegment:
    points: tuple[float, float, float, float]
    angle_deg: float
    length: float
    midpoint: tuple[float, float]


def _segment(values: np.ndarray) -> LineSegment:
    x1, y1, x2, y2 = (float(value) for value in values)
    dx = x2 - x1
    dy = y2 - y1
    return LineSegment(
        points=(x1, y1, x2, y2),
        angle_deg=float(np.degrees(np.arctan2(dy, dx)) % 180.0),
        length=float(np.hypot(dx, dy)),
        midpoint=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
    )


def _dedupe(segments: list[LineSegment], *, axis: int, tolerance: float) -> list[LineSegment]:
    selected: list[LineSegment] = []
    for segment in sorted(segments, key=lambda item: item.length, reverse=True):
        coordinate = segment.midpoint[axis]
        if any(abs(coordinate - other.midpoint[axis]) <= tolerance for other in selected):
            continue
        selected.append(segment)
    return selected


def detect_court_line_segments(
    frame: np.ndarray,
) -> tuple[list[LineSegment], list[LineSegment], np.ndarray]:
    height, width = frame.shape[:2]
    mask = bright_line_mask(frame)
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    edges = cv2.Canny(mask, 40, 120)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=max(24, int(min(width, height) * 0.06)),
        minLineLength=max(30, int(min(width, height) * 0.12)),
        maxLineGap=max(12, int(min(width, height) * 0.04)),
    )
    if lines is None:
        return [], [], mask

    horizontal: list[LineSegment] = []
    side: list[LineSegment] = []
    # OpenCV 4 commonly returns (N, 1, 4), while OpenCV 5 may return
    # (N, 4). Normalize at this Adapter boundary.
    for raw in np.asarray(lines).reshape(-1, 4):
        segment = _segment(raw)
        shallow_angle = min(segment.angle_deg, 180.0 - segment.angle_deg)
        if shallow_angle <= 20.0:
            horizontal.append(segment)
        elif 45.0 <= segment.angle_deg <= 135.0:
            side.append(segment)
    return (
        _dedupe(horizontal, axis=1, tolerance=max(8.0, height * 0.018))[:10],
        _dedupe(side, axis=0, tolerance=max(10.0, width * 0.02))[:10],
        mask,
    )


def _intersection(first: LineSegment, second: LineSegment) -> np.ndarray | None:
    x1, y1, x2, y2 = first.points
    x3, y3, x4, y4 = second.points
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denominator) < 1e-6:
        return None
    determinant_first = x1 * y2 - y1 * x2
    determinant_second = x3 * y4 - y3 * x4
    return np.asarray(
        [
            (determinant_first * (x3 - x4) - (x1 - x2) * determinant_second) / denominator,
            (determinant_first * (y3 - y4) - (y1 - y2) * determinant_second) / denominator,
        ],
        dtype=np.float32,
    )


def _polygon_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))) / 2.0


def generate_hough_candidates(
    frame: np.ndarray,
    *,
    frame_index: int = 0,
    max_candidates: int = 12,
) -> list[CalibrationCandidate]:
    horizontal, sides, _mask = detect_court_line_segments(frame)
    height, width = frame.shape[:2]
    candidates: list[tuple[float, CalibrationCandidate]] = []
    for first_horizontal, second_horizontal in combinations(horizontal, 2):
        top, bottom = sorted(
            (first_horizontal, second_horizontal),
            key=lambda item: item.midpoint[1],
        )
        if bottom.midpoint[1] - top.midpoint[1] < height * 0.25:
            continue
        for first_side, second_side in combinations(sides, 2):
            left, right = sorted((first_side, second_side), key=lambda item: item.midpoint[0])
            if right.midpoint[0] - left.midpoint[0] < width * 0.18:
                continue
            intersections = (
                _intersection(top, left),
                _intersection(top, right),
                _intersection(bottom, right),
                _intersection(bottom, left),
            )
            if any(point is None for point in intersections):
                continue
            corners = np.asarray(intersections, dtype=np.float32)
            if not np.all(np.isfinite(corners)):
                continue
            if np.any(corners[:, 0] < -width * 0.03) or np.any(corners[:, 0] > width * 1.03):
                continue
            if np.any(corners[:, 1] < -height * 0.03) or np.any(corners[:, 1] > height * 1.03):
                continue
            area_ratio = _polygon_area(corners) / max(1.0, float(width * height))
            if area_ratio < 0.08:
                continue
            total_length = top.length + bottom.length + left.length + right.length
            geometric_score = area_ratio + total_length / max(1.0, 2.0 * width + height)
            candidates.append(
                (
                    geometric_score,
                    CalibrationCandidate(
                        corners=corners,
                        source=CalibrationCandidateSource.HOUGH_LINES,
                        frame_index=frame_index,
                        diagnostics={
                            "geometric_score": round(geometric_score, 6),
                            "horizontal_segment_count": len(horizontal),
                            "side_segment_count": len(sides),
                        },
                    ),
                )
            )
    # Long carpet edges can outrank fragmented white doubles sidelines on raw
    # segment length alone. Keep a broad geometric shortlist, then rank it by
    # support for the complete regulation line layout in the white-line mask.
    candidates.sort(key=lambda item: item[0], reverse=True)
    shortlist = candidates[: max(80, max_candidates * 10)]
    ranked: list[tuple[float, float, CalibrationCandidate]] = []
    for geometric_score, candidate in shortlist:
        support = score_court_line_support(frame, candidate.corners, line_mask=_mask)
        candidate.diagnostics["ranking_line_support"] = round(support.score, 6)
        ranked.append((support.score, geometric_score, candidate))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [candidate for _, _, candidate in ranked[:max_candidates]]

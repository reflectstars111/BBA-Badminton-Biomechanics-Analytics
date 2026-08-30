from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import cv2
import numpy as np

from badminton_data_process.calibration.reference import (
    DOUBLES_LONG_SERVICE_FROM_BASELINE_M,
    SINGLES_SIDE_MARGIN_M,
    STANDARD_COURT,
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
    segments, mask = _detect_all_line_segments(frame)
    height, width = frame.shape[:2]
    horizontal: list[LineSegment] = []
    side: list[LineSegment] = []
    for segment in segments:
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


def _detect_all_line_segments(frame: np.ndarray) -> tuple[list[LineSegment], np.ndarray]:
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
        return [], mask

    # OpenCV 4 commonly returns (N, 1, 4), while OpenCV 5 may return
    # (N, 4). Normalize at this Adapter boundary.
    return [_segment(raw) for raw in np.asarray(lines).reshape(-1, 4)], mask


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


def _outside_regulation_line_penalty(
    frame: np.ndarray,
    corners: np.ndarray,
    line_mask: np.ndarray,
) -> float:
    """Penalize an internal regulation rectangle masquerading as the outer court.

    Badminton's singles sidelines and long-service lines form smaller,
    self-similar rectangles. A candidate can align every projected line while
    still choosing one of those internal lines as an outer boundary. If the
    next expected regulation spacing *outside* a proposed boundary contains a
    strong white line, the proposed boundary is probably internal.
    """

    height, width = frame.shape[:2]
    matrix = cv2.getPerspectiveTransform(STANDARD_COURT.corners_array(), corners)
    distance_map = cv2.distanceTransform(
        255 - (line_mask > 0).astype(np.uint8) * 255,
        cv2.DIST_L2,
        3,
    )
    tolerance = max(3.0, min(width, height) * 0.013)
    court_width = STANDARD_COURT.width_m
    court_length = STANDARD_COURT.length_m
    outside_lines = (
        ((-SINGLES_SIDE_MARGIN_M, 0.0), (-SINGLES_SIDE_MARGIN_M, court_length)),
        ((court_width + SINGLES_SIDE_MARGIN_M, 0.0), (court_width + SINGLES_SIDE_MARGIN_M, court_length)),
        ((0.0, -DOUBLES_LONG_SERVICE_FROM_BASELINE_M), (court_width, -DOUBLES_LONG_SERVICE_FROM_BASELINE_M)),
        ((0.0, court_length + DOUBLES_LONG_SERVICE_FROM_BASELINE_M), (court_width, court_length + DOUBLES_LONG_SERVICE_FROM_BASELINE_M)),
    )
    scores: list[float] = []
    for start, end in outside_lines:
        projected = cv2.perspectiveTransform(
            np.asarray([[start, end]], dtype=np.float32),
            matrix,
        )[0]
        length = float(np.linalg.norm(projected[1] - projected[0]))
        if length < 1.0:
            continue
        count = max(18, int(length / 5.0))
        xs = np.linspace(float(projected[0, 0]), float(projected[1, 0]), count)
        ys = np.linspace(float(projected[0, 1]), float(projected[1, 1]), count)
        valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        if float(np.mean(valid)) < 0.25:
            continue
        sample_x = np.clip(np.rint(xs[valid]).astype(np.int32), 0, width - 1)
        sample_y = np.clip(np.rint(ys[valid]).astype(np.int32), 0, height - 1)
        distances = distance_map[sample_y, sample_x]
        soft = float(np.mean(1.0 - np.clip(distances / tolerance, 0.0, 1.0)))
        coverage = float(np.mean(distances <= tolerance))
        scores.append(0.68 * soft + 0.32 * coverage)
    return max(scores, default=0.0)


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
    ranked: list[tuple[float, float, float, CalibrationCandidate]] = []
    for geometric_score, candidate in shortlist:
        support = score_court_line_support(frame, candidate.corners, line_mask=_mask)
        outside_penalty = _outside_regulation_line_penalty(
            frame,
            candidate.corners,
            _mask,
        )
        adjusted_support = support.score - 1.2 * outside_penalty
        candidate.diagnostics["ranking_line_support"] = round(support.score, 6)
        candidate.diagnostics["outside_line_penalty"] = round(outside_penalty, 6)
        candidate.diagnostics["adjusted_line_support"] = round(adjusted_support, 6)
        ranked.append((adjusted_support, support.score, geometric_score, candidate))
    ranked.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return [candidate for _, _, _, candidate in ranked[:max_candidates]]


def generate_low_angle_hough_candidates(
    frame: np.ndarray,
    *,
    frame_index: int = 0,
    max_candidates: int = 16,
) -> list[CalibrationCandidate]:
    """Generate candidates for extreme side views, including one off-frame corner.

    A low sideline camera can project one longitudinal court boundary almost
    horizontally and move its near corner beyond the image. The regular Hough
    Adapter deliberately assumes steep side boundaries, so it cannot represent
    that geometry. This variant searches the second boundary across every long
    white segment and lets validation/ranking decide the regulation-line fit.
    """

    segments, mask = _detect_all_line_segments(frame)
    height, width = frame.shape[:2]
    baselines = [
        segment
        for segment in segments
        if min(segment.angle_deg, 180.0 - segment.angle_deg) <= 24.0
    ]
    baselines = _dedupe(
        baselines,
        axis=1,
        tolerance=max(6.0, height * 0.012),
    )[:16]
    boundaries = sorted(segments, key=lambda item: item.length, reverse=True)[:28]
    raw_candidates: list[tuple[float, CalibrationCandidate]] = []
    for first_baseline, second_baseline in combinations(baselines, 2):
        top, bottom = sorted(
            (first_baseline, second_baseline),
            key=lambda item: item.midpoint[1],
        )
        if bottom.midpoint[1] - top.midpoint[1] < height * 0.045:
            continue
        for first_side, second_side in combinations(boundaries, 2):
            if first_side in {top, bottom} or second_side in {top, bottom}:
                continue
            first_top = _intersection(top, first_side)
            second_top = _intersection(top, second_side)
            first_bottom = _intersection(bottom, first_side)
            second_bottom = _intersection(bottom, second_side)
            if any(
                point is None
                for point in (first_top, second_top, first_bottom, second_bottom)
            ):
                continue
            assert first_top is not None and second_top is not None
            assert first_bottom is not None and second_bottom is not None
            if first_top[0] <= second_top[0]:
                corners = np.asarray(
                    [first_top, second_top, second_bottom, first_bottom],
                    dtype=np.float32,
                )
            else:
                corners = np.asarray(
                    [second_top, first_top, first_bottom, second_bottom],
                    dtype=np.float32,
                )
            if not np.all(np.isfinite(corners)) or not cv2.isContourConvex(corners):
                continue
            if np.any(corners[:, 0] < -width * 0.60) or np.any(corners[:, 0] > width * 1.65):
                continue
            if np.any(corners[:, 1] < -height * 0.20) or np.any(corners[:, 1] > height * 1.20):
                continue
            inside = (
                (corners[:, 0] >= 0)
                & (corners[:, 0] < width)
                & (corners[:, 1] >= 0)
                & (corners[:, 1] < height)
            )
            if int(np.count_nonzero(~inside)) > 1:
                continue
            outside_indices = np.flatnonzero(~inside)
            # For a camera looking from the near side, perspective can push a
            # near-baseline corner beyond the frame. A far-baseline corner
            # outside the image usually means an internal court line was
            # mistaken for an outer boundary.
            if len(outside_indices) == 1 and int(outside_indices[0]) not in {2, 3}:
                continue
            far_width = float(np.linalg.norm(corners[1] - corners[0]))
            near_width = float(np.linalg.norm(corners[2] - corners[3]))
            if near_width < far_width * 0.90:
                continue
            area_ratio = _polygon_area(corners) / max(1.0, float(width * height))
            if area_ratio < 0.06:
                continue
            total_length = top.length + bottom.length + first_side.length + second_side.length
            geometric_score = area_ratio + total_length / max(1.0, 2.0 * width + height)
            raw_candidates.append(
                (
                    geometric_score,
                    CalibrationCandidate(
                        corners=corners,
                        source=CalibrationCandidateSource.HOUGH_LINES,
                        frame_index=frame_index,
                        diagnostics={
                            "adapter": "low_angle_hough",
                            "geometric_score": round(geometric_score, 6),
                            "segment_count": len(segments),
                            "allows_one_off_frame_corner": True,
                        },
                    ),
                )
            )

    raw_candidates.sort(key=lambda item: item[0], reverse=True)
    # Exterior-line scoring is the expensive part. Geometry already removes
    # degenerate quads, so a bounded shortlist keeps five-frame low-angle
    # calibration practical without changing the validation seam.
    shortlist = raw_candidates[: max(96, max_candidates * 6)]
    ranked: list[tuple[float, float, float, float, CalibrationCandidate]] = []
    for geometric_score, candidate in shortlist:
        support = score_court_line_support(frame, candidate.corners, line_mask=mask)
        boundary_names = (
            "far_baseline",
            "right_doubles_sideline",
            "near_baseline",
            "left_doubles_sideline",
        )
        boundary_support = float(
            np.mean([support.per_line.get(name, 0.0) for name in boundary_names])
        )
        outside_penalty = _outside_regulation_line_penalty(
            frame,
            candidate.corners,
            mask,
        )
        # The regulation grid is strongly self-similar, so the exterior-line
        # evidence must outweigh a small gain in raw in-court support; without
        # this, a singles/service rectangle regularly beats the outer court.
        adjusted_support = support.score - 2.5 * outside_penalty
        candidate.diagnostics["ranking_line_support"] = round(support.score, 6)
        candidate.diagnostics["ranking_boundary_support"] = round(boundary_support, 6)
        candidate.diagnostics["outside_line_penalty"] = round(outside_penalty, 6)
        candidate.diagnostics["adjusted_line_support"] = round(adjusted_support, 6)
        ranked.append(
            (
                adjusted_support,
                support.score,
                boundary_support,
                geometric_score,
                candidate,
            )
        )
    ranked.sort(key=lambda item: (item[0], item[1], item[2], item[3]), reverse=True)
    return [candidate for _, _, _, _, candidate in ranked[:max_candidates]]

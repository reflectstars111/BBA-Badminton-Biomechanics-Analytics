from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np


COURT_WIDTH_M = 6.10
COURT_LENGTH_M = 13.40
SINGLES_WIDTH_M = 5.18
SINGLES_SIDE_MARGIN_M = (COURT_WIDTH_M - SINGLES_WIDTH_M) / 2.0
SHORT_SERVICE_FROM_NET_M = 1.98
DOUBLES_LONG_SERVICE_FROM_BASELINE_M = 0.76
NET_Y_M = COURT_LENGTH_M / 2.0


@dataclass(frozen=True, slots=True)
class CourtLine:
    name: str
    start: tuple[float, float]
    end: tuple[float, float]
    weight: float = 1.0


@dataclass(frozen=True, slots=True)
class CourtGeometry:
    """Standard doubles-court geometry in metres.

    Court coordinates start at the far-left doubles corner. ``x`` increases
    to the right and ``y`` increases from the far baseline to the near
    baseline. Image corners always use TL, TR, BR, BL order.
    """

    court_type: str
    width_m: float
    length_m: float
    lines: tuple[CourtLine, ...]

    @property
    def corners(self) -> tuple[tuple[float, float], ...]:
        return (
            (0.0, 0.0),
            (self.width_m, 0.0),
            (self.width_m, self.length_m),
            (0.0, self.length_m),
        )

    def corners_array(self) -> np.ndarray:
        return np.asarray(self.corners, dtype=np.float32)

    def project_lines(
        self,
        image_corners: Iterable[Iterable[float]],
    ) -> tuple[tuple[CourtLine, np.ndarray, np.ndarray], ...]:
        corners = np.asarray(image_corners, dtype=np.float32)
        if corners.shape != (4, 2):
            raise ValueError("image_corners must use TL, TR, BR, BL shape (4, 2)")
        matrix = cv2.getPerspectiveTransform(self.corners_array(), corners)
        projected: list[tuple[CourtLine, np.ndarray, np.ndarray]] = []
        for line in self.lines:
            points = np.asarray([[line.start, line.end]], dtype=np.float32)
            image_points = cv2.perspectiveTransform(points, matrix)[0]
            projected.append((line, image_points[0], image_points[1]))
        return tuple(projected)


STANDARD_COURT = CourtGeometry(
    court_type="badminton_doubles",
    width_m=COURT_WIDTH_M,
    length_m=COURT_LENGTH_M,
    lines=(
        CourtLine("far_baseline", (0.0, 0.0), (COURT_WIDTH_M, 0.0), 1.35),
        CourtLine("right_doubles_sideline", (COURT_WIDTH_M, 0.0), (COURT_WIDTH_M, COURT_LENGTH_M), 1.35),
        CourtLine("near_baseline", (COURT_WIDTH_M, COURT_LENGTH_M), (0.0, COURT_LENGTH_M), 1.35),
        CourtLine("left_doubles_sideline", (0.0, COURT_LENGTH_M), (0.0, 0.0), 1.35),
        CourtLine("left_singles_sideline", (SINGLES_SIDE_MARGIN_M, 0.0), (SINGLES_SIDE_MARGIN_M, COURT_LENGTH_M), 0.95),
        CourtLine("right_singles_sideline", (COURT_WIDTH_M - SINGLES_SIDE_MARGIN_M, 0.0), (COURT_WIDTH_M - SINGLES_SIDE_MARGIN_M, COURT_LENGTH_M), 0.95),
        CourtLine("far_doubles_long_service", (0.0, DOUBLES_LONG_SERVICE_FROM_BASELINE_M), (COURT_WIDTH_M, DOUBLES_LONG_SERVICE_FROM_BASELINE_M)),
        CourtLine("near_doubles_long_service", (0.0, COURT_LENGTH_M - DOUBLES_LONG_SERVICE_FROM_BASELINE_M), (COURT_WIDTH_M, COURT_LENGTH_M - DOUBLES_LONG_SERVICE_FROM_BASELINE_M)),
        CourtLine("far_short_service", (0.0, NET_Y_M - SHORT_SERVICE_FROM_NET_M), (COURT_WIDTH_M, NET_Y_M - SHORT_SERVICE_FROM_NET_M), 1.15),
        CourtLine("near_short_service", (0.0, NET_Y_M + SHORT_SERVICE_FROM_NET_M), (COURT_WIDTH_M, NET_Y_M + SHORT_SERVICE_FROM_NET_M), 1.15),
        CourtLine("far_center", (COURT_WIDTH_M / 2.0, 0.0), (COURT_WIDTH_M / 2.0, NET_Y_M - SHORT_SERVICE_FROM_NET_M), 0.8),
        CourtLine("near_center", (COURT_WIDTH_M / 2.0, NET_Y_M + SHORT_SERVICE_FROM_NET_M), (COURT_WIDTH_M / 2.0, COURT_LENGTH_M), 0.8),
        CourtLine("net", (0.0, NET_Y_M), (COURT_WIDTH_M, NET_Y_M), 0.35),
    ),
)


@dataclass(frozen=True, slots=True)
class LineSupportReport:
    score: float
    coverage: float
    supported_lines: int
    total_lines: int
    tolerance_px: float
    per_line: dict[str, float]

    def as_dict(self) -> dict[str, object]:
        return {
            "score": round(self.score, 6),
            "coverage": round(self.coverage, 6),
            "supported_lines": self.supported_lines,
            "total_lines": self.total_lines,
            "tolerance_px": round(self.tolerance_px, 3),
            "per_line": {name: round(value, 6) for name, value in self.per_line.items()},
        }


def bright_line_mask(frame: np.ndarray) -> np.ndarray:
    # A global grayscale threshold turns a bright grey/green playing surface
    # into one solid foreground region. Hough then detects the vinyl/carpet
    # perimeter instead of the regulation white lines. Court lines are both
    # near-neutral (low saturation) and in the bright tail of the current
    # frame, so keep only that intersection.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    value_threshold = min(220.0, max(170.0, float(np.percentile(value, 88.0))))
    mask = ((value > value_threshold) & (saturation <= 110)).astype(np.uint8) * 255
    kernel = np.ones((3, 3), dtype=np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)


def score_court_line_support(
    frame: np.ndarray,
    image_corners: Iterable[Iterable[float]],
    *,
    geometry: CourtGeometry = STANDARD_COURT,
    line_mask: np.ndarray | None = None,
    tolerance_px: float | None = None,
) -> LineSupportReport:
    height, width = frame.shape[:2]
    mask = bright_line_mask(frame) if line_mask is None else (line_mask > 0).astype(np.uint8) * 255
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    tolerance = tolerance_px or max(3.0, min(width, height) * 0.013)
    distance_map = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)

    weighted_score = 0.0
    total_weight = 0.0
    coverages: list[float] = []
    per_line: dict[str, float] = {}
    supported_lines = 0
    for line, start, end in geometry.project_lines(image_corners):
        length = float(np.linalg.norm(end - start))
        if length < 1.0:
            per_line[line.name] = 0.0
            continue
        count = max(18, int(length / 5.0))
        xs = np.linspace(float(start[0]), float(end[0]), count)
        ys = np.linspace(float(start[1]), float(end[1]), count)
        valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        if float(np.mean(valid)) < 0.35:
            per_line[line.name] = 0.0
            continue
        sample_x = np.clip(np.rint(xs[valid]).astype(np.int32), 0, width - 1)
        sample_y = np.clip(np.rint(ys[valid]).astype(np.int32), 0, height - 1)
        distances = distance_map[sample_y, sample_x]
        soft_score = float(np.mean(1.0 - np.clip(distances / tolerance, 0.0, 1.0)))
        coverage = float(np.mean(distances <= tolerance))
        line_score = 0.68 * soft_score + 0.32 * coverage
        per_line[line.name] = line_score
        weighted_score += line.weight * line_score
        total_weight += line.weight
        coverages.append(coverage)
        if coverage >= 0.42:
            supported_lines += 1

    score = weighted_score / total_weight if total_weight else 0.0
    return LineSupportReport(
        score=float(np.clip(score, 0.0, 1.0)),
        coverage=float(np.mean(coverages)) if coverages else 0.0,
        supported_lines=supported_lines,
        total_lines=len(geometry.lines),
        tolerance_px=float(tolerance),
        per_line=per_line,
    )

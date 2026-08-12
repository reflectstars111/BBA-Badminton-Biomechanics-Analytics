from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover - runtime dependency
    cv2 = None
    np = None


def require_opencv() -> None:
    if cv2 is None or np is None:
        raise RuntimeError("OpenCV and NumPy are required for main-view analysis.")


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def ramp(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return clamp((value - low) / (high - low))


@dataclass(slots=True)
class FrameScore:
    sample_frame: int
    timestamp: float
    main_view_score: float
    court_score: float
    geometry_score: float
    layout_score: float
    stability_score: float
    line_score: float
    reject_score: float
    court_area_ratio: float
    court_span_x: float
    court_span_y: float
    player_candidate_count: int
    player_split_sides: int
    is_main_view: int
    reject_reason: str
    corners: list[list[float]] | None = None

    def to_row(self) -> dict[str, object]:
        return {
            "sample_frame": self.sample_frame,
            "timestamp": round(self.timestamp, 3),
            "main_view_score": round(self.main_view_score, 4),
            "court_score": round(self.court_score, 4),
            "geometry_score": round(self.geometry_score, 4),
            "layout_score": round(self.layout_score, 4),
            "stability_score": round(self.stability_score, 4),
            "line_score": round(self.line_score, 4),
            "reject_score": round(self.reject_score, 4),
            "court_area_ratio": round(self.court_area_ratio, 4),
            "court_span_x": round(self.court_span_x, 4),
            "court_span_y": round(self.court_span_y, 4),
            "player_candidate_count": self.player_candidate_count,
            "player_split_sides": self.player_split_sides,
            "is_main_view": self.is_main_view,
            "reject_reason": self.reject_reason,
        }


def green_mask(frame: Any) -> Any:
    require_opencv()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    return (
        (hue >= 35)
        & (hue <= 95)
        & (saturation >= 35)
        & (value >= 30)
    ).astype(np.uint8) * 255


def edge_ratios(frame: Any) -> tuple[float, float]:
    require_opencv()
    resized = cv2.resize(frame, (320, 180))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return (
        float(np.count_nonzero(edges)) / float(edges.size),
        float(np.mean(edges[60:120, :] > 0)),
    )


def detect_court_corners(frame: Any) -> tuple[Any | None, float, float, float]:
    require_opencv()
    mask = green_mask(frame)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0.0, 0.0, 0.0

    contour = max(contours, key=cv2.contourArea)
    frame_area = float(frame.shape[0] * frame.shape[1])
    area_ratio = float(cv2.contourArea(contour)) / frame_area if frame_area else 0.0
    if area_ratio <= 0.0:
        return None, 0.0, 0.0, 0.0

    points = contour.reshape(-1, 2)
    x, y, w, h = cv2.boundingRect(contour)
    span_x = w / float(frame.shape[1])
    span_y = h / float(frame.shape[0])
    top_band = points[(points[:, 1] >= y) & (points[:, 1] <= y + h * 0.38)]
    bottom_band = points[(points[:, 1] >= y + h * 0.58) & (points[:, 1] <= y + h)]
    if len(top_band) < 4:
        top_band = points
    if len(bottom_band) < 4:
        bottom_band = points
    corners = np.array(
        [
            top_band[np.argmin(top_band[:, 0])],
            top_band[np.argmax(top_band[:, 0])],
            bottom_band[np.argmax(bottom_band[:, 0])],
            bottom_band[np.argmin(bottom_band[:, 0])],
        ],
        dtype=np.float32,
    )
    return corners, area_ratio, span_x, span_y


def polygon_area(points: Any) -> float:
    require_opencv()
    return float(abs(cv2.contourArea(points.astype(np.float32))))


def geometry_score(corners: Any | None, frame_shape: tuple[int, int, int]) -> float:
    require_opencv()
    if corners is None or len(corners) != 4:
        return 0.0
    height, width = frame_shape[:2]
    area_ratio = polygon_area(corners) / float(width * height)
    span_x = (float(np.max(corners[:, 0])) - float(np.min(corners[:, 0]))) / float(width)
    span_y = (float(np.max(corners[:, 1])) - float(np.min(corners[:, 1]))) / float(height)
    top_width = float(np.linalg.norm(corners[1] - corners[0]))
    bottom_width = float(np.linalg.norm(corners[2] - corners[3]))
    left_height = float(np.linalg.norm(corners[3] - corners[0]))
    right_height = float(np.linalg.norm(corners[2] - corners[1]))
    if min(top_width, bottom_width, left_height, right_height) <= 1.0:
        return 0.0
    width_ratio = top_width / bottom_width
    side_ratio = left_height / right_height
    y_order = float(corners[0, 1] < corners[3, 1] and corners[1, 1] < corners[2, 1])

    area_score = ramp(area_ratio, 0.10, 0.42)
    span_score = min(ramp(span_x, 0.45, 0.85), ramp(span_y, 0.38, 0.78))
    width_score = 1.0 - min(abs(math.log(max(width_ratio, 1e-6))) / math.log(4.0), 1.0)
    side_score = 1.0 - min(abs(math.log(max(side_ratio, 1e-6))) / math.log(2.5), 1.0)
    return clamp(0.35 * area_score + 0.30 * span_score + 0.20 * width_score + 0.10 * side_score + 0.05 * y_order)


def court_score(area_ratio: float, span_x: float, span_y: float, line_ratio: float, middle_edge_ratio: float) -> float:
    area_score = ramp(area_ratio, 0.10, 0.40)
    span_score = min(ramp(span_x, 0.42, 0.82), ramp(span_y, 0.35, 0.75))
    line_score = min(ramp(line_ratio, 0.07, 0.15), ramp(middle_edge_ratio, 0.10, 0.20))
    return clamp(0.45 * area_score + 0.35 * span_score + 0.20 * line_score)


def build_court_mask(frame_shape: tuple[int, int, int], corners: Any | None) -> Any:
    require_opencv()
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    if corners is not None:
        cv2.fillConvexPoly(mask, corners.astype(np.int32), 255)
    return mask


def player_layout_score(frame: Any, corners: Any | None) -> tuple[float, int, int, float]:
    require_opencv()
    if corners is None:
        return 0.0, 0, 0, 0.0
    court = build_court_mask(frame.shape, corners)
    non_green = cv2.bitwise_not(green_mask(frame))
    mask = cv2.bitwise_and(non_green, court)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = float(frame.shape[0] * frame.shape[1])
    candidates: list[tuple[int, int, int, int, float]] = []
    max_blob_ratio = 0.0
    for contour in contours:
        area = float(cv2.contourArea(contour))
        max_blob_ratio = max(max_blob_ratio, area / frame_area)
        if area < 80 or area > frame_area * 0.035:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if h < 14 or w < 5 or w > frame.shape[1] * 0.20:
            continue
        aspect = h / float(max(w, 1))
        if aspect < 0.45:
            continue
        candidates.append((x, y, w, h, area))

    count = len(candidates)
    if count == 0:
        return 0.15, 0, 0, max_blob_ratio
    mid_y = float(np.mean(corners[:, 1]))
    has_top = any(y + h / 2.0 < mid_y for _, y, w, h, _ in candidates)
    has_bottom = any(y + h / 2.0 >= mid_y for _, y, w, h, _ in candidates)
    split = int(has_top and has_bottom)
    if 2 <= count <= 5 and split:
        score = 1.0
    elif 2 <= count <= 6:
        score = 0.68
    elif count == 1:
        score = 0.42
    else:
        score = 0.52
    if max_blob_ratio > 0.10:
        score *= 0.5
    return clamp(score), count, split, max_blob_ratio


def stability_score(corners: Any | None, previous_corners: Any | None, frame_shape: tuple[int, int, int]) -> float:
    require_opencv()
    if corners is None:
        return 0.0
    if previous_corners is None:
        return 0.75
    diagonal = math.hypot(frame_shape[1], frame_shape[0])
    delta = float(np.mean(np.linalg.norm(corners - previous_corners, axis=1))) / max(diagonal, 1.0)
    return clamp(1.0 - delta / 0.08)


def reject_reason(
    court: float,
    geometry: float,
    layout: float,
    reject: float,
    main_view: float,
    threshold: float,
) -> str:
    if main_view >= threshold and reject < 0.4:
        return ""
    if reject >= 0.65:
        return "likely_closeup_or_replay"
    if geometry < 0.45:
        return "low_court_geometry_score"
    if court < 0.45:
        return "low_court_score"
    if layout < 0.35:
        return "player_layout_invalid"
    return "low_main_view_score"


def score_frame(
    frame: Any,
    frame_index: int,
    fps: float,
    previous_corners: Any | None = None,
    threshold: float = 0.75,
) -> tuple[FrameScore, Any | None]:
    require_opencv()
    line_ratio, middle_edge_ratio = edge_ratios(frame)
    corners, area_ratio, span_x, span_y = detect_court_corners(frame)
    court = court_score(area_ratio, span_x, span_y, line_ratio, middle_edge_ratio)
    geometry = geometry_score(corners, frame.shape)
    layout, player_count, split, max_blob_ratio = player_layout_score(frame, corners)
    stability = stability_score(corners, previous_corners, frame.shape)
    line = min(ramp(line_ratio, 0.07, 0.15), ramp(middle_edge_ratio, 0.10, 0.20))
    reject = clamp(max(ramp(max_blob_ratio, 0.10, 0.22), 1.0 - court) * 0.75 + max(0.0, 0.45 - geometry) * 0.35)
    main_view = clamp(
        0.34 * court
        + 0.28 * geometry
        + 0.18 * layout
        + 0.12 * stability
        + 0.08 * line
        - 0.22 * reject
    )
    is_main = int(main_view >= threshold and reject < 0.4)
    reason = reject_reason(court, geometry, layout, reject, main_view, threshold)
    return (
        FrameScore(
            sample_frame=frame_index,
            timestamp=frame_index / fps,
            main_view_score=main_view,
            court_score=court,
            geometry_score=geometry,
            layout_score=layout,
            stability_score=stability,
            line_score=line,
            reject_score=reject,
            court_area_ratio=area_ratio,
            court_span_x=span_x,
            court_span_y=span_y,
            player_candidate_count=player_count,
            player_split_sides=split,
            is_main_view=is_main,
            reject_reason=reason,
            corners=corners.astype(float).tolist() if corners is not None else None,
        ),
        corners,
    )


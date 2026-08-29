from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def frame_metrics(frame: np.ndarray) -> dict[str, float]:
    resized = cv2.resize(frame, (320, 180))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)

    edges = cv2.Canny(gray, 50, 150)
    line_ratio = float(np.count_nonzero(edges)) / float(edges.size)
    green_mask = (
        (hue >= 35)
        & (hue <= 95)
        & (saturation >= 40)
        & (value >= 35)
    )
    return {
        "line_ratio": line_ratio,
        "green_ratio": float(np.mean(green_mask)),
        "center_green_ratio": float(np.mean(green_mask[40:175, 25:295])),
        "bottom_green_ratio": float(np.mean(green_mask[70:179, 40:280])),
        "top_green_ratio": float(np.mean(green_mask[20:90, 40:280])),
        "middle_green_ratio": float(np.mean(green_mask[60:130, 40:280])),
        "left_green_ratio": float(np.mean(green_mask[:, :160])),
        "right_green_ratio": float(np.mean(green_mask[:, 160:])),
        "top_dark_ratio": float(np.mean(gray[:50, :] < 40)),
        "middle_edge_ratio": float(np.mean(edges[60:120, :] > 0)),
    }


def frame_motion_scores(
    current_gray: np.ndarray,
    previous_gray: np.ndarray,
) -> tuple[float, float]:
    """Return full-frame and normalized play-area motion scores."""
    if current_gray.shape != previous_gray.shape:
        raise ValueError("motion frames must have identical shapes")
    frame_diff = cv2.absdiff(current_gray, previous_gray)
    height, width = frame_diff.shape[:2]
    x0 = int(round(width * 0.0875))
    x1 = int(round(width * 0.9125))
    y0 = int(round(height * 0.1556))
    y1 = int(round(height * 0.9889))
    play_area = frame_diff[y0:y1, x0:x1]
    if play_area.size == 0:
        play_area = frame_diff
    return (
        float(np.mean(frame_diff)) / 255.0,
        float(np.mean(play_area)) / 255.0,
    )


def analyze_video(
    input_path: Path,
    sample_every: int,
    min_motion_score: float,
    max_motion_score: float,
    min_center_green_ratio: float,
    min_bottom_green_ratio: float,
    min_line_ratio: float,
    min_top_green_ratio: float,
    min_middle_green_ratio: float,
    max_left_right_green_diff: float,
    min_top_dark_ratio: float,
    min_middle_edge_ratio: float,
) -> tuple[list[dict[str, float]], float, int, int]:
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    analysis_rows: list[dict[str, float]] = []
    previous_gray: np.ndarray | None = None
    frame_index = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % sample_every != 0:
            frame_index += 1
            continue

        gray = cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2GRAY)
        if previous_gray is None:
            global_motion_score = 0.0
            play_area_motion_score = 0.0
        else:
            global_motion_score, play_area_motion_score = frame_motion_scores(
                gray,
                previous_gray,
            )
        metrics = frame_metrics(frame)
        is_rally_view = (
            metrics["center_green_ratio"] >= min_center_green_ratio
            and metrics["bottom_green_ratio"] >= min_bottom_green_ratio
            and metrics["top_green_ratio"] >= min_top_green_ratio
            and metrics["middle_green_ratio"] >= min_middle_green_ratio
            and abs(metrics["left_green_ratio"] - metrics["right_green_ratio"])
            <= max_left_right_green_diff
            and metrics["top_dark_ratio"] >= min_top_dark_ratio
        )
        is_court_view = (
            is_rally_view
            and metrics["line_ratio"] >= min_line_ratio
            and metrics["middle_edge_ratio"] >= min_middle_edge_ratio
        )
        is_candidate = (
            is_court_view
            and min_motion_score <= play_area_motion_score <= max_motion_score
        )
        analysis_rows.append(
            {
                "sample_frame": float(frame_index),
                "timestamp": frame_index / fps,
                "motion_score": play_area_motion_score,
                "global_motion_score": global_motion_score,
                "play_area_motion_score": play_area_motion_score,
                **metrics,
                "is_rally_view": float(is_rally_view),
                "is_court_view": float(is_court_view),
                "is_candidate": float(is_candidate),
            }
        )
        previous_gray = gray
        frame_index += 1

    capture.release()
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video shape for: {input_path}")
    return analysis_rows, fps, frame_count, width * height

from __future__ import annotations

import cv2
import numpy as np
import pytest

from badminton_data_process.calibration.reference import (
    COURT_LENGTH_M,
    COURT_WIDTH_M,
    NET_Y_M,
    STANDARD_COURT,
    bright_line_mask,
    score_court_line_support,
)
from badminton_data_process.calibration.court import _frame_candidates
from badminton_data_process.calibration.hough import generate_hough_candidates
from badminton_data_process.calibration.validation import (
    CalibrationCandidate,
    CalibrationCandidateSource,
    CalibrationThresholds,
    normalized_corner_rmse,
    select_stable_calibration,
    validate_calibration_candidate,
)


def _court_frame(corners: np.ndarray, size: tuple[int, int] = (640, 480)) -> np.ndarray:
    width, height = size
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    for _line, start, end in STANDARD_COURT.project_lines(corners):
        cv2.line(frame, tuple(np.rint(start).astype(int)), tuple(np.rint(end).astype(int)), (255, 255, 255), 4)
    return frame


def test_standard_court_geometry_is_complete_and_uses_metres() -> None:
    assert STANDARD_COURT.court_type == "badminton_doubles"
    assert STANDARD_COURT.corners == (
        (0.0, 0.0),
        (COURT_WIDTH_M, 0.0),
        (COURT_WIDTH_M, COURT_LENGTH_M),
        (0.0, COURT_LENGTH_M),
    )
    names = {line.name for line in STANDARD_COURT.lines}
    assert names >= {
        "left_singles_sideline",
        "right_singles_sideline",
        "far_doubles_long_service",
        "near_doubles_long_service",
        "far_short_service",
        "near_short_service",
        "far_center",
        "near_center",
        "net",
    }
    assert NET_Y_M == pytest.approx(6.7)


def test_complete_line_support_prefers_correct_projection() -> None:
    corners = np.asarray([[180, 80], [460, 80], [570, 430], [70, 430]], dtype=np.float32)
    frame = _court_frame(corners)
    correct = score_court_line_support(frame, corners)
    shifted = score_court_line_support(frame, corners + np.asarray([45, 0], dtype=np.float32))

    assert correct.score > 0.85
    assert correct.supported_lines >= 11
    assert correct.score > shifted.score + 0.25


def test_validation_rejects_wrong_order_and_accepts_supported_court() -> None:
    corners = np.asarray([[180, 80], [460, 80], [570, 430], [70, 430]], dtype=np.float32)
    frame = _court_frame(corners)
    accepted = validate_calibration_candidate(
        frame,
        CalibrationCandidate(corners, CalibrationCandidateSource.MANUAL, 10),
        thresholds=CalibrationThresholds(min_line_support=0.5),
    )
    wrong_order = validate_calibration_candidate(
        frame,
        CalibrationCandidate(corners[[0, 2, 1, 3]], CalibrationCandidateSource.MANUAL, 10),
        thresholds=CalibrationThresholds(min_line_support=0.0),
    )

    assert accepted.accepted
    assert accepted.homography_image_to_court is not None
    assert accepted.quality.line_support is not None
    assert not wrong_order.accepted
    assert "corners_not_convex" in wrong_order.reasons or "corners_wrong_order" in wrong_order.reasons


def test_stable_selection_rejects_single_outlier() -> None:
    base = np.asarray([[180, 80], [460, 80], [570, 430], [70, 430]], dtype=np.float32)
    frame = _court_frame(base)
    results = []
    for frame_index, shift in enumerate((0.0, 2.0, 75.0)):
        corners = base + np.asarray([shift, 0.0], dtype=np.float32)
        candidate_frame = _court_frame(corners)
        results.append(
            validate_calibration_candidate(
                candidate_frame,
                CalibrationCandidate(corners, CalibrationCandidateSource.GREEN_CONTOUR, frame_index),
                thresholds=CalibrationThresholds(min_line_support=0.5),
            )
        )

    selected, stable_frames = select_stable_calibration(
        results,
        frame.shape,
        max_corner_rmse_ratio=0.02,
        min_stable_candidates=2,
    )

    assert selected is not None
    assert set(stable_frames) == {0, 1}
    assert normalized_corner_rmse(base, base + np.asarray([2.0, 0.0]), frame.shape) < 0.01


def test_hough_adapter_generates_candidates_without_validating_them() -> None:
    corners = np.asarray([[180, 80], [460, 80], [570, 430], [70, 430]], dtype=np.float32)
    frame = _court_frame(corners)

    candidates = generate_hough_candidates(frame, frame_index=7)

    assert candidates
    assert all(candidate.source is CalibrationCandidateSource.HOUGH_LINES for candidate in candidates)
    assert all(candidate.frame_index == 7 for candidate in candidates)


def test_white_line_mask_excludes_bright_low_saturation_court_surface() -> None:
    frame = np.full((240, 360, 3), (35, 35, 105), dtype=np.uint8)
    floor = np.asarray([[110, 60], [250, 60], [300, 230], [60, 230]], dtype=np.int32)
    cv2.fillConvexPoly(frame, floor, (170, 178, 170))
    cv2.line(frame, (95, 70), (285, 70), (245, 245, 245), 4)

    mask = bright_line_mask(frame)

    assert mask[150, 180] == 0  # court surface is evidence/ROI, not a white line
    assert mask[70, 180] == 255


def test_hybrid_calibration_never_promotes_green_edge_to_court_boundary() -> None:
    corners = np.asarray([[180, 80], [460, 80], [570, 430], [70, 430]], dtype=np.float32)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.fillConvexPoly(
        frame,
        np.asarray([[145, 55], [495, 55], [625, 465], [15, 465]], dtype=np.int32),
        (70, 155, 70),
    )
    for _line, start, end in STANDARD_COURT.project_lines(corners):
        cv2.line(
            frame,
            tuple(np.rint(start).astype(int)),
            tuple(np.rint(end).astype(int)),
            (255, 255, 255),
            4,
        )

    candidates = _frame_candidates(frame, frame_index=7, detector="hybrid")

    assert candidates
    assert all(
        candidate.source is CalibrationCandidateSource.HOUGH_LINES
        for candidate in candidates
    )

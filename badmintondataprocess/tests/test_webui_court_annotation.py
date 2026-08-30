from __future__ import annotations

import cv2
import numpy as np
import pytest

from badminton_data_process.calibration.reference import STANDARD_COURT
from badminton_data_process.webui.court_annotation import (
    accept_auto_annotation,
    add_clicked_corner,
    add_clicked_model_line,
    apply_manual_annotation,
    court_corners_from_model_lines,
    format_reference_points,
    parse_reference_points_text,
    prepare_court_preview,
)
from badminton_data_process.calibration.validation import (
    CalibrationCandidate,
    CalibrationCandidateSource,
    CalibrationQuality,
    CalibrationValidationResult,
    normalized_corner_rmse,
)


def _court_rgb(corners: np.ndarray, width: int = 640, height: int = 480) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    for _line, start, end in STANDARD_COURT.project_lines(corners):
        cv2.line(
            frame,
            tuple(np.rint(start).astype(int)),
            tuple(np.rint(end).astype(int)),
            (255, 255, 255),
            4,
        )
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def test_manual_reference_text_supports_one_low_angle_off_frame_corner() -> None:
    text = "0.10,0.20; 0.80,0.20; 1.25,0.90; 0.05,0.90"

    parsed = parse_reference_points_text(text)

    assert parsed == [0.1, 0.2, 0.8, 0.2, 1.25, 0.9, 0.05, 0.9]
    assert format_reference_points(parsed).startswith("0.100000,0.200000")
    with pytest.raises(ValueError, match="四组"):
        parse_reference_points_text("0.1,0.2; 0.8,0.2")


def test_four_clicks_are_normalized_and_manual_projection_is_validated() -> None:
    corners = np.asarray([[180, 80], [460, 80], [570, 430], [70, 430]], dtype=np.float32)
    base_rgb = _court_rgb(corners)
    points: list[tuple[float, float]] = []
    text = ""
    for point in corners:
        _preview, points, text, _status = add_clicked_corner(
            base_rgb,
            points,
            (float(point[0]), float(point[1])),
        )

    preview, normalized, status = apply_manual_annotation(base_rgb, text, "overhead")

    assert preview.shape == base_rgb.shape
    assert normalized == pytest.approx(
        [180 / 640, 80 / 480, 460 / 640, 80 / 480, 570 / 640, 430 / 480, 70 / 640, 430 / 480],
        abs=1e-6,
    )
    assert "手动标定已锁定" in status


def test_accept_auto_annotation_locks_the_exact_previewed_points() -> None:
    corners = np.asarray([[180, 80], [460, 80], [570, 430], [70, 430]], dtype=np.float32)
    base_rgb = _court_rgb(corners)
    points = (corners / np.asarray([640, 480], dtype=np.float32)).reshape(-1).tolist()

    preview, confirmed, status = accept_auto_annotation(base_rgb, points, "overhead")

    assert preview.shape == base_rgb.shape
    assert confirmed == pytest.approx(points, abs=1e-6)
    assert "已接受自动标定" in status
    with pytest.raises(ValueError, match="没有可接受"):
        accept_auto_annotation(base_rgb, None, "overhead")


def test_preview_chooses_best_candidate_from_multiple_video_times(monkeypatch) -> None:
    import badminton_data_process.webui.court_annotation as annotation

    class FakeCapture:
        def isOpened(self) -> bool:
            return True

        def get(self, _property: int) -> int:
            return 100

        def release(self) -> None:
            pass

    frames = {
        10: np.zeros((100, 200, 3), dtype=np.uint8),
        80: np.full((100, 200, 3), 20, dtype=np.uint8),
    }
    corners_by_frame = {
        10: np.asarray([[20, 20], [160, 20], [180, 90], [10, 90]], dtype=np.float32),
        80: np.asarray([[30, 25], [170, 25], [190, 95], [5, 95]], dtype=np.float32),
    }

    monkeypatch.setattr(annotation.cv2, "VideoCapture", lambda _path: FakeCapture())
    monkeypatch.setattr(annotation, "representative_frame_indices", lambda _count: [10, 80])
    monkeypatch.setattr(annotation, "read_frame_at", lambda _capture, index: frames[index])
    monkeypatch.setattr(
        annotation,
        "_frame_candidates",
        lambda _frame, index, _detector: [
            CalibrationCandidate(
                corners_by_frame[index],
                CalibrationCandidateSource.HOUGH_LINES,
                index,
            )
        ],
    )

    def accepted_result(_frame, candidate, *, thresholds):
        _ = thresholds
        score = 0.55 if candidate.frame_index == 10 else 0.91
        return CalibrationValidationResult(
            candidate,
            True,
            [],
            CalibrationQuality(quality_score=score),
            np.eye(3),
        )

    monkeypatch.setattr(annotation, "validate_calibration_candidate", accepted_result)

    preview, base, suggested, status = prepare_court_preview("sample.mp4", "overhead")

    expected = (corners_by_frame[80] / np.asarray([200, 100], dtype=np.float32)).reshape(-1)
    assert preview.shape == base.shape == frames[80].shape
    assert suggested == pytest.approx(expected.tolist(), abs=1e-6)
    assert "第 80 帧" in status
    assert "尚未确认" in status


def test_semantic_lines_extrapolate_an_off_frame_outer_corner() -> None:
    corners = np.asarray(
        [[47, 291], [321, 274], [940, 284], [240, 351]],
        dtype=np.float32,
    )
    selected_names = (
        "left_singles_sideline",
        "right_singles_sideline",
        "far_short_service",
        "near_short_service",
    )
    projected = {
        line.name: [tuple(start.astype(float)), tuple(end.astype(float))]
        for line, start, end in STANDARD_COURT.project_lines(corners)
        if line.name in selected_names
    }

    fitted = court_corners_from_model_lines(projected)

    assert fitted[2, 0] > 640
    assert normalized_corner_rmse(fitted, corners, (360, 640, 3)) < 1e-4


def test_confirmed_model_allows_one_off_frame_corner_for_overhead_mode() -> None:
    corners = np.asarray(
        [[47, 291], [321, 274], [940, 284], [240, 351]],
        dtype=np.float32,
    )
    base_rgb = _court_rgb(corners, width=640, height=360)
    normalized = (corners / np.asarray([640, 360], dtype=np.float32)).reshape(-1).tolist()

    _preview, confirmed, status = apply_manual_annotation(
        base_rgb,
        format_reference_points(normalized),
        "overhead",
    )

    assert confirmed[4] > 1.0
    assert "手动标定已锁定" in status


def test_confirmed_model_allows_two_off_frame_corners_for_low_angle_mode() -> None:
    corners = np.asarray(
        [[100, 100], [540, 100], [700, 430], [-60, 430]],
        dtype=np.float32,
    )
    base_rgb = _court_rgb(corners, width=640, height=480)
    normalized = (corners / np.asarray([640, 480], dtype=np.float32)).reshape(-1).tolist()

    _preview, confirmed, status = apply_manual_annotation(
        base_rgb,
        format_reference_points(normalized),
        "low",
    )

    assert confirmed[4] > 1.0
    assert confirmed[6] < 0.0
    assert "手动标定已锁定" in status


def test_eight_model_line_clicks_generate_valid_corner_text() -> None:
    corners = np.asarray([[180, 80], [460, 80], [570, 430], [70, 430]], dtype=np.float32)
    base_rgb = _court_rgb(corners)
    selected_names = [
        "left_singles_sideline",
        "right_singles_sideline",
        "far_short_service",
        "near_short_service",
    ]
    endpoints = {
        line.name: [tuple(start.astype(float)), tuple(end.astype(float))]
        for line, start, end in STANDARD_COURT.project_lines(corners)
        if line.name in selected_names
    }
    points: list[tuple[float, float]] = []
    text = ""
    status = ""
    for name in selected_names:
        for endpoint in endpoints[name]:
            _preview, points, text, status = add_clicked_model_line(
                base_rgb,
                points,
                endpoint,
                selected_names,
                "overhead",
            )

    assert len(points) == 8
    assert len(parse_reference_points_text(text)) == 8
    assert "标准球场模型已拟合" in status
    assert "校验通过" in status

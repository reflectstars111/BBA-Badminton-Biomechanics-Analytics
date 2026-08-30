from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from badminton_data_process.calibration.court import (
    _frame_candidates,
    draw_preview,
    normalized_corners,
    read_frame_at,
    representative_frame_indices,
)
from badminton_data_process.calibration.validation import (
    CalibrationCandidate,
    CalibrationCandidateSource,
    CalibrationThresholds,
    validate_calibration_candidate,
)
from badminton_data_process.calibration.reference import STANDARD_COURT


CORNER_NAMES = ("TL", "TR", "BR", "BL")
MODEL_LINE_LABELS = {
    "left_doubles_sideline": "左侧双打边线",
    "left_singles_sideline": "左侧单打边线",
    "right_singles_sideline": "右侧单打边线",
    "right_doubles_sideline": "右侧双打边线",
    "far_baseline": "远端底线",
    "far_doubles_long_service": "远端双打长发球线",
    "far_short_service": "远端前发球线",
    "near_short_service": "近端前发球线",
    "near_doubles_long_service": "近端双打长发球线",
    "near_baseline": "近端底线",
}
MODEL_LINE_COLORS = (
    (255, 210, 40),
    (80, 210, 255),
    (255, 80, 220),
    (80, 255, 120),
)


def parse_reference_points_text(value: str) -> list[float]:
    """Parse normalized ``x,y;...`` corners, including one off-frame point."""

    try:
        pairs = [
            [float(item.strip()) for item in pair.split(",")]
            for pair in value.split(";")
            if pair.strip()
        ]
    except ValueError as exc:
        raise ValueError("角点格式应为 x,y; x,y; x,y; x,y") from exc
    if len(pairs) != 4 or any(len(pair) != 2 for pair in pairs):
        raise ValueError("角点格式应包含 TL、TR、BR、BL 四组 x,y")
    flattened = [number for pair in pairs for number in pair]
    if not all(np.isfinite(flattened)):
        raise ValueError("角点坐标必须是有限数字")
    if any(value < -2.0 or value > 3.0 for value in flattened):
        raise ValueError("归一化角点必须位于 [-2, 3] 范围内")
    return flattened


def format_reference_points(points: list[tuple[float, float]] | list[float]) -> str:
    values = np.asarray(points, dtype=np.float32).reshape(4, 2)
    return "; ".join(f"{x:.6f},{y:.6f}" for x, y in values)


def _manual_thresholds(view_kind: str) -> CalibrationThresholds:
    _ = view_kind
    return CalibrationThresholds(
        min_area_ratio=0.08,
        min_line_support=0.45,
        # A user-confirmed model may legitimately extrapolate one outer corner
        # beyond the crop in either advertised camera class.
        max_out_of_bounds_ratio=0.25,
    )


def _auto_preview_thresholds(view_kind: str) -> CalibrationThresholds:
    return CalibrationThresholds(
        min_area_ratio=0.08,
        min_line_support=0.45,
        max_out_of_bounds_ratio=0.25 if view_kind == "low" else 0.0,
    )


def prepare_court_preview(
    video_path: str | Path,
    view_kind: str,
) -> tuple[np.ndarray, np.ndarray, list[float] | None, str]:
    """Pick a court-like frame and overlay an uncommitted auto candidate.

    The preview is deliberately a confirmation gate, not an automatic promise:
    callers must explicitly accept ``auto_points`` or replace them with manual
    points before starting the full pipeline.
    """

    detector = "hough_low_angle" if view_kind == "low" else "hybrid"
    thresholds = _auto_preview_thresholds(view_kind)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError("无法打开上传的视频")
    best: tuple[Any, np.ndarray] | None = None
    fallback: np.ndarray | None = None
    sampled_indices: list[int] = []
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        sampled_indices = representative_frame_indices(frame_count)
        for frame_index in sampled_indices:
            frame = read_frame_at(capture, frame_index)
            if fallback is None:
                fallback = frame
            accepted = [
                validate_calibration_candidate(frame, candidate, thresholds=thresholds)
                for candidate in _frame_candidates(frame, frame_index, detector)
            ]
            accepted = [result for result in accepted if result.accepted]
            if not accepted:
                continue
            selected = accepted[0] if view_kind == "low" else max(
                accepted,
                key=lambda result: result.quality.quality_score,
            )
            if best is None or selected.quality.quality_score > best[0].quality.quality_score:
                best = (selected, frame)
    finally:
        capture.release()

    if fallback is None:
        raise RuntimeError("上传的视频没有可读取画面")
    if best is None:
        base_rgb = cv2.cvtColor(fallback, cv2.COLOR_BGR2RGB)
        return (
            base_rgb,
            base_rgb,
            None,
            (
                f"已检查 {len(sampled_indices)} 个代表时间点，但未找到可接受的自动候选。"
                "请按 TL → TR → BR → BL 点击四角并应用手动标注；确认前不能开始分析。"
            ),
        )
    selected, frame = best
    base_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    preview = draw_preview(frame, selected, stable_frames=[])
    height, width = frame.shape[:2]
    auto_points = (
        selected.candidate.corners / np.asarray([width, height], dtype=np.float32)
    ).reshape(-1).astype(float).tolist()
    return (
        cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
        base_rgb,
        auto_points,
        (
            f"已从 {len(sampled_indices)} 个代表时间点中选择第 "
            f"{selected.candidate.frame_index} 帧。自动建议质量 "
            f"{selected.quality.quality_score:.3f}，尚未确认。"
            "请检查所有白色双打外边线：正确则点击“接受自动标定”，否则手动调整。"
        ),
    )


def accept_auto_annotation(
    base_rgb: np.ndarray,
    auto_points: list[float] | None,
    view_kind: str,
) -> tuple[np.ndarray, list[float], str]:
    """Validate and lock the exact auto candidate that the user inspected."""

    if not auto_points:
        raise ValueError("当前没有可接受的自动标定，请改用手动四点标注")
    preview, normalized, message = apply_manual_annotation(
        base_rgb,
        format_reference_points(auto_points),
        view_kind,
    )
    return preview, normalized, "已接受自动标定。" + message.removeprefix("手动标定已锁定：")


def _model_line_axis_and_coordinate(name: str) -> tuple[str, float]:
    line = next((item for item in STANDARD_COURT.lines if item.name == name), None)
    if line is None or name not in MODEL_LINE_LABELS:
        raise ValueError(f"不支持的球场模型线：{name}")
    if abs(line.start[0] - line.end[0]) < 1e-6:
        return "x", float(line.start[0])
    if abs(line.start[1] - line.end[1]) < 1e-6:
        return "y", float(line.start[1])
    raise ValueError(f"球场模型线不是水平或纵向直线：{name}")


def _image_line(endpoints: list[tuple[float, float]]) -> np.ndarray:
    points = np.asarray(endpoints, dtype=np.float64)
    if points.shape != (2, 2) or np.linalg.norm(points[1] - points[0]) < 2.0:
        raise ValueError("每条模型线需要两个相距足够远的画面内标注点")
    line = np.cross(
        np.asarray([points[0, 0], points[0, 1], 1.0]),
        np.asarray([points[1, 0], points[1, 1], 1.0]),
    )
    norm = float(np.linalg.norm(line[:2]))
    if norm < 1e-9:
        raise ValueError("模型线标注退化")
    return line / norm


def court_corners_from_model_lines(
    image_lines: dict[str, list[tuple[float, float]]],
) -> np.ndarray:
    """Fit the standard court from semantic lines, including off-frame corners.

    Users annotate only visible portions of two longitudinal and two transverse
    regulation lines. Their infinite-line intersections provide four image ↔
    metric correspondences; the outer doubles corners are then extrapolated by
    the standard 6.10 m × 13.40 m court model.
    """

    if len(image_lines) != 4:
        raise ValueError("模型标定必须包含两条纵线和两条横线")
    coefficients = {name: _image_line(points) for name, points in image_lines.items()}
    x_lines: list[tuple[str, float]] = []
    y_lines: list[tuple[str, float]] = []
    for name in image_lines:
        axis, coordinate = _model_line_axis_and_coordinate(name)
        (x_lines if axis == "x" else y_lines).append((name, coordinate))
    if len(x_lines) != 2 or len(y_lines) != 2:
        raise ValueError("请选择两条不同的纵向边线和两条不同的横向场地线")
    x_lines.sort(key=lambda item: item[1])
    y_lines.sort(key=lambda item: item[1])
    if abs(x_lines[0][1] - x_lines[1][1]) < 1e-6 or abs(y_lines[0][1] - y_lines[1][1]) < 1e-6:
        raise ValueError("所选模型线在标准球场中不能重合")

    correspondence_order = (
        (x_lines[0], y_lines[0]),
        (x_lines[1], y_lines[0]),
        (x_lines[1], y_lines[1]),
        (x_lines[0], y_lines[1]),
    )
    court_points: list[tuple[float, float]] = []
    image_points: list[tuple[float, float]] = []
    for (x_name, x_value), (y_name, y_value) in correspondence_order:
        intersection = np.cross(coefficients[x_name], coefficients[y_name])
        if abs(float(intersection[2])) < 1e-8:
            raise ValueError(f"{MODEL_LINE_LABELS[x_name]}与{MODEL_LINE_LABELS[y_name]}在透视图中近似平行")
        point = intersection[:2] / intersection[2]
        if not np.all(np.isfinite(point)):
            raise ValueError("模型线交点无效")
        court_points.append((x_value, y_value))
        image_points.append((float(point[0]), float(point[1])))

    court_to_image = cv2.getPerspectiveTransform(
        np.asarray(court_points, dtype=np.float32),
        np.asarray(image_points, dtype=np.float32),
    )
    corners = cv2.perspectiveTransform(
        STANDARD_COURT.corners_array().reshape(1, 4, 2),
        court_to_image,
    )[0]
    if not np.all(np.isfinite(corners)):
        raise ValueError("标准球场模型投影无效")
    return corners.astype(np.float32)


def add_clicked_model_line(
    base_rgb: np.ndarray,
    clicked_points: list[tuple[float, float]] | None,
    point: tuple[float, float],
    line_names: list[str],
    view_kind: str,
) -> tuple[np.ndarray, list[tuple[float, float]], str, str]:
    """Add or move a semantic-line handle and preview the fitted court model."""

    if base_rgb is None:
        raise ValueError("请先生成场地预览")
    if len(line_names) != 4 or len(set(line_names)) != 4:
        raise ValueError("请选择四条不同的球场模型线")
    # Validate the two-by-two semantic arrangement before collecting points.
    axes = [_model_line_axis_and_coordinate(name)[0] for name in line_names]
    if axes.count("x") != 2 or axes.count("y") != 2:
        raise ValueError("模型标定需要两条纵线和两条横线")

    points = [(float(x), float(y)) for x, y in (clicked_points or [])]
    current = (float(point[0]), float(point[1]))
    if len(points) < 8:
        points.append(current)
    else:
        nearest = min(
            range(len(points)),
            key=lambda index: float(np.linalg.norm(np.asarray(points[index]) - np.asarray(current))),
        )
        points[nearest] = current

    frame = cv2.cvtColor(np.asarray(base_rgb), cv2.COLOR_RGB2BGR)
    preview = frame.copy()
    for line_index, name in enumerate(line_names):
        endpoints = points[line_index * 2 : line_index * 2 + 2]
        color = MODEL_LINE_COLORS[line_index]
        for endpoint_index, endpoint in enumerate(endpoints):
            pixel = tuple(np.rint(endpoint).astype(int))
            cv2.circle(preview, pixel, 7, color, -1, cv2.LINE_AA)
            cv2.putText(
                preview,
                f"L{line_index + 1}.{endpoint_index + 1}",
                (pixel[0] + 8, pixel[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        if len(endpoints) == 2:
            first = np.asarray(endpoints[0], dtype=np.float64)
            second = np.asarray(endpoints[1], dtype=np.float64)
            delta = second - first
            p1 = tuple(np.rint(first - delta * 20.0).astype(int))
            p2 = tuple(np.rint(second + delta * 20.0).astype(int))
            cv2.line(preview, p1, p2, color, 2, cv2.LINE_AA)
            cv2.putText(
                preview,
                MODEL_LINE_LABELS[name],
                tuple(np.rint(first + np.asarray([10.0, 18.0])).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )

    if len(points) < 8:
        next_index = len(points)
        next_name = line_names[next_index // 2]
        status = (
            f"模型线标注 {len(points)}/8；下一点：{MODEL_LINE_LABELS[next_name]} "
            f"第 {next_index % 2 + 1} 点。请沿同一条可见白线取两个相距较远的点。"
        )
        return cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), points, "", status

    image_lines = {
        name: points[index * 2 : index * 2 + 2]
        for index, name in enumerate(line_names)
    }
    corners = court_corners_from_model_lines(image_lines)
    height, width = frame.shape[:2]
    normalized = (corners / np.asarray([width, height], dtype=np.float32)).reshape(-1).tolist()
    candidate = CalibrationCandidate(
        corners=corners,
        source=CalibrationCandidateSource.MANUAL,
        frame_index=0,
        diagnostics={"submitted_by": "webui_model_lines", "model_lines": list(line_names)},
    )
    result = validate_calibration_candidate(frame, candidate, thresholds=_manual_thresholds(view_kind))
    preview = draw_preview(frame, result, stable_frames=[0] if result.accepted else [])
    outside = int(np.count_nonzero((corners[:, 0] < 0) | (corners[:, 0] >= width) | (corners[:, 1] < 0) | (corners[:, 1] >= height)))
    status = (
        f"标准球场模型已拟合；推导出 {outside} 个画外角点。"
        + (
            "校验通过，可点击“应用模型标定”。继续点击图像可移动最近的线控制点。"
            if result.accepted
            else "当前未通过校验：" + ", ".join(result.reasons) + "。继续点击可移动最近的控制点。"
        )
    )
    return cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), points, format_reference_points(normalized), status


def add_clicked_corner(
    base_rgb: np.ndarray,
    clicked_points: list[tuple[float, float]] | None,
    point: tuple[float, float],
) -> tuple[np.ndarray, list[tuple[float, float]], str, str]:
    if base_rgb is None:
        raise ValueError("请先生成场地预览")
    points = list(clicked_points or [])
    if len(points) >= 4:
        points = []
    points.append((float(point[0]), float(point[1])))
    preview = cv2.cvtColor(np.asarray(base_rgb), cv2.COLOR_RGB2BGR)
    for index, current in enumerate(points):
        pixel = tuple(np.rint(current).astype(int))
        cv2.circle(preview, pixel, 7, (0, 90, 255), -1, cv2.LINE_AA)
        cv2.putText(
            preview,
            CORNER_NAMES[index],
            (pixel[0] + 8, pixel[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    if len(points) > 1:
        cv2.polylines(
            preview,
            [np.rint(points).astype(np.int32)],
            len(points) == 4,
            (0, 220, 0),
            2,
            cv2.LINE_AA,
        )
    height, width = base_rgb.shape[:2]
    normalized = [(x / width, y / height) for x, y in points]
    text = format_reference_points(normalized) if len(points) == 4 else ""
    next_name = CORNER_NAMES[len(points)] if len(points) < 4 else "完成"
    status = f"已选择 {len(points)}/4；下一点：{next_name}。"
    if len(points) == 4:
        status += " 点击“应用手动角点”进行完整球场线验证。"
    return cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), points, text, status


def apply_manual_annotation(
    base_rgb: np.ndarray,
    points_text: str,
    view_kind: str,
) -> tuple[np.ndarray, list[float], str]:
    if base_rgb is None:
        raise ValueError("请先生成场地预览")
    flattened = parse_reference_points_text(points_text)
    frame = cv2.cvtColor(np.asarray(base_rgb), cv2.COLOR_RGB2BGR)
    corners = normalized_corners(
        frame.shape,
        flattened,
        allow_out_of_bounds=True,
    )
    result = validate_calibration_candidate(
        frame,
        CalibrationCandidate(
            corners=corners,
            source=CalibrationCandidateSource.MANUAL,
            frame_index=0,
            diagnostics={"submitted_by": "webui"},
        ),
        thresholds=_manual_thresholds(view_kind),
    )
    preview = draw_preview(frame, result, stable_frames=[0] if result.accepted else [])
    if not result.accepted:
        reasons = ", ".join(result.reasons) or "unknown"
        raise ValueError(f"手动角点未通过统一校准验证：{reasons}")
    return (
        cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
        flattened,
        (
            f"手动标定已锁定：质量 {result.quality.quality_score:.3f}。"
            "本次分析会将该归一化 Homography 应用于同一机位的所有有效回合。"
        ),
    )


def clear_manual_annotation(base_rgb: np.ndarray | None) -> tuple[Any, list[Any], None, str, str]:
    return base_rgb, [], None, "", "确认已清除；请接受自动建议或重新完成模型线标注。"

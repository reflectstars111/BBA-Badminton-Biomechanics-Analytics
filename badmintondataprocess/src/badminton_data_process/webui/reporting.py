from __future__ import annotations

import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from badminton_data_process.calibration.reference import COURT_LENGTH_M, NET_Y_M
from badminton_data_process.core.io import read_csv_rows, read_json, write_json

MAX_PLAYER_SPEED_M_S = 12.0


def _float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _int(value: object) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _true(value: object) -> bool:
    return str(value).lower() in {"1", "true", "yes"}


def _mean(values: Iterable[float]) -> float | None:
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else None


def _round(value: float | None, digits: int = 3) -> float | None:
    return round(value, digits) if value is not None and math.isfinite(value) else None


def _group_rows(
    rows: Iterable[dict[str, str]],
    fields: tuple[str, ...],
) -> dict[tuple[str, ...], list[dict[str, str]]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(field, "") for field in fields)].append(row)
    return dict(groups)


def _player_point(row: dict[str, str], player_id: str) -> tuple[int, float, float, float] | None:
    frame_id = _int(row.get("frame_id"))
    timestamp = _float(row.get("timestamp"))
    if _true(row.get("is_smoothed_valid")):
        x = _float(row.get("smoothed_court_x"))
        y = _float(row.get("smoothed_court_y"))
    else:
        x = _float(row.get("court_x"))
        y = _float(row.get("court_y"))
    if frame_id is None or timestamp is None or x is None or y is None:
        return None
    if not (0.0 <= x <= 6.10 and 0.0 <= y <= COURT_LENGTH_M):
        return None
    if player_id == "near" and y < NET_Y_M:
        return None
    if player_id == "far" and y > NET_Y_M:
        return None
    return frame_id, timestamp, x, y


def _body_center_height(row: dict[str, str]) -> tuple[float, float] | None:
    body_y = _float(row.get("smoothed_body_image_y") or row.get("body_image_y"))
    ground_y = _float(row.get("smoothed_ground_image_y") or row.get("ground_image_y"))
    bbox_top = _float(row.get("bbox_y1"))
    bbox_bottom = _float(row.get("bbox_y2"))
    if body_y is None or ground_y is None or bbox_top is None or bbox_bottom is None:
        return None
    bbox_height = bbox_bottom - bbox_top
    pixel_height = ground_y - body_y
    if bbox_height <= 1.0 or pixel_height < 0.0:
        return None
    return pixel_height, min(1.0, max(0.0, pixel_height / bbox_height))


def _player_metrics(
    player_id: str,
    rows: list[dict[str, str]],
    expected_frames: int,
) -> dict[str, Any]:
    points = sorted(
        (point for row in rows if (point := _player_point(row, player_id)) is not None),
        key=lambda point: point[0],
    )
    speeds: list[float] = []
    total_distance = 0.0
    movement_duration = 0.0
    for previous, current in zip(points, points[1:]):
        frame_gap = current[0] - previous[0]
        time_gap = current[1] - previous[1]
        if frame_gap != 1 or time_gap <= 0.0:
            continue
        distance = math.hypot(current[2] - previous[2], current[3] - previous[3])
        speed = distance / time_gap
        if speed > MAX_PLAYER_SPEED_M_S:
            continue
        speeds.append(speed)
        total_distance += distance
        movement_duration += time_gap

    heights = [height for row in rows if (height := _body_center_height(row)) is not None]
    pose_rows = sum(_true(row.get("pose_valid")) for row in rows)
    zone_counts = {"front": 0, "mid": 0, "back": 0}
    for _, _, _, court_y in points:
        depth = court_y - NET_Y_M if player_id == "near" else NET_Y_M - court_y
        if depth < 2.5:
            zone_counts["front"] += 1
        elif depth > 4.5:
            zone_counts["back"] += 1
        else:
            zone_counts["mid"] += 1
    zone_total = sum(zone_counts.values())
    return {
        "player_id": player_id,
        "tracked_rows": len(rows),
        "valid_frames": len(points),
        "expected_frames": expected_frames,
        "tracking_coverage_ratio": len(points) / expected_frames if expected_frames else None,
        "pose_valid_ratio": pose_rows / len(rows) if rows else None,
        "movement_duration_seconds": movement_duration,
        "total_distance_m": total_distance if speeds else None,
        "average_speed_m_s": total_distance / movement_duration if movement_duration else None,
        "maximum_speed_m_s": max(speeds) if speeds else None,
        "current_speed_m_s": speeds[-1] if speeds else None,
        "average_body_center_height_px": _mean(height[0] for height in heights),
        "average_body_center_height_ratio": _mean(height[1] for height in heights),
        "front_court_ratio": zone_counts["front"] / zone_total if zone_total else None,
        "mid_court_ratio": zone_counts["mid"] / zone_total if zone_total else None,
        "back_court_ratio": zone_counts["back"] / zone_total if zone_total else None,
        "_speed_samples": speeds,
        "_pose_valid_frames": pose_rows,
        "_height_count": len(heights),
        "_height_sum_px": sum(height[0] for height in heights),
        "_height_sum_ratio": sum(height[1] for height in heights),
    }


def _trim_speed_outliers(speeds: list[float]) -> list[float]:
    if len(speeds) < 5:
        return speeds
    ordered = sorted(speeds)
    median = statistics.median(ordered)
    deviations = [abs(value - median) for value in ordered]
    mad = statistics.median(deviations)
    percentile_99 = ordered[min(len(ordered) - 1, int((len(ordered) - 1) * 0.99))]
    robust_limit = median + 8.0 * mad if mad > 0.0 else median * 4.0
    limit = max(median * 3.0, min(percentile_99, robust_limit))
    return [value for value in speeds if value <= limit]


def _shuttle_metrics(rows: list[dict[str, str]], frame_diagonal: float | None) -> dict[str, Any]:
    points: list[tuple[int, float, float, float]] = []
    for row in rows:
        if not _true(row.get("is_smoothed_valid")) or _true(row.get("is_gap_filled")):
            continue
        frame_id = _int(row.get("frame_id"))
        timestamp = _float(row.get("timestamp"))
        x = _float(row.get("smoothed_x") or row.get("x"))
        y = _float(row.get("smoothed_y") or row.get("y"))
        if frame_id is not None and timestamp is not None and x is not None and y is not None:
            points.append((frame_id, timestamp, x, y))
    points.sort(key=lambda point: point[0])
    raw_speeds: list[float] = []
    for previous, current in zip(points, points[1:]):
        time_gap = current[1] - previous[1]
        if current[0] - previous[0] != 1 or time_gap <= 0.0:
            continue
        raw_speeds.append(
            math.hypot(current[2] - previous[2], current[3] - previous[3]) / time_gap
        )
    speeds = _trim_speed_outliers(raw_speeds)
    average_speed = _mean(speeds)
    return {
        "valid_observed_frames": len(points),
        "expected_frames": len(rows),
        "visibility_ratio": len(points) / len(rows) if rows else None,
        "average_image_speed_px_s": average_speed,
        "maximum_image_speed_px_s": max(speeds) if speeds else None,
        "current_image_speed_px_s": speeds[-1] if speeds else None,
        "average_screen_diagonals_s": (
            average_speed / frame_diagonal
            if average_speed is not None and frame_diagonal
            else None
        ),
        "_speed_samples": speeds,
    }


def _video_diagonal(path: Path) -> float | None:
    if not path.is_file():
        return None
    try:
        import cv2
    except ImportError:
        return None
    capture = cv2.VideoCapture(str(path))
    try:
        width = float(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0)
        height = float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0)
    finally:
        capture.release()
    return math.hypot(width, height) if width > 0.0 and height > 0.0 else None


def _public_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: _round(value) if isinstance(value, float) else value
        for key, value in metrics.items()
        if not key.startswith("_")
    }


def build_web_report(run_dir: Path) -> dict[str, Any]:
    """Create detailed, UI-neutral match statistics from completed artifacts."""

    run_dir = Path(run_dir).resolve()
    summary_path = run_dir / "analysis_summary.json"
    summary = read_json(summary_path) if summary_path.is_file() else {}
    player_rows = read_csv_rows(run_dir / "annotations" / "player_tracks_smoothed.csv")
    shuttle_rows = read_csv_rows(run_dir / "annotations" / "shuttle_tracks_smoothed.csv")
    rally_rows = read_csv_rows(run_dir / "rallies.csv")
    analysis_video = Path(summary.get("outputs", {}).get("analysis_video", ""))
    frame_diagonal = _video_diagonal(analysis_video)

    shuttle_by_rally = _group_rows(shuttle_rows, ("video_stem", "rally_id"))
    player_by_rally = _group_rows(player_rows, ("video_stem", "rally_id", "player_id"))
    rally_player_metrics: list[dict[str, Any]] = []
    internal_by_player: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (video_stem, rally_id, player_id), rows in sorted(player_by_rally.items()):
        expected = len(shuttle_by_rally.get((video_stem, rally_id), []))
        metrics = _player_metrics(player_id, rows, expected)
        metrics.update({"video_stem": video_stem, "rally_id": rally_id})
        internal_by_player[player_id].append(metrics)
        rally_player_metrics.append(_public_metrics(metrics))

    players: list[dict[str, Any]] = []
    for player_id, groups in sorted(internal_by_player.items()):
        movement_duration = sum(group["movement_duration_seconds"] for group in groups)
        distance_values = [group["total_distance_m"] for group in groups if group["total_distance_m"] is not None]
        total_distance = sum(distance_values) if distance_values else None
        speeds = [speed for group in groups for speed in group["_speed_samples"]]
        expected_frames = sum(group["expected_frames"] for group in groups)
        valid_frames = sum(group["valid_frames"] for group in groups)
        tracked_rows = sum(group["tracked_rows"] for group in groups)
        pose_valid_frames = sum(group["_pose_valid_frames"] for group in groups)
        height_count = sum(group["_height_count"] for group in groups)
        weighted_zone_total = valid_frames or 1
        players.append(
            _public_metrics(
                {
                    "player_id": player_id,
                    "rallies": len(groups),
                    "valid_frames": valid_frames,
                    "tracking_coverage_ratio": valid_frames / expected_frames if expected_frames else None,
                    "pose_valid_ratio": pose_valid_frames / tracked_rows if tracked_rows else None,
                    "movement_duration_seconds": movement_duration,
                    "total_distance_m": total_distance,
                    "average_speed_m_s": total_distance / movement_duration if total_distance is not None and movement_duration else None,
                    "maximum_speed_m_s": max(speeds) if speeds else None,
                    "current_speed_m_s": speeds[-1] if speeds else None,
                    "average_body_center_height_px": sum(group["_height_sum_px"] for group in groups) / height_count if height_count else None,
                    "average_body_center_height_ratio": sum(group["_height_sum_ratio"] for group in groups) / height_count if height_count else None,
                    "front_court_ratio": sum((group["front_court_ratio"] or 0.0) * group["valid_frames"] for group in groups) / weighted_zone_total,
                    "mid_court_ratio": sum((group["mid_court_ratio"] or 0.0) * group["valid_frames"] for group in groups) / weighted_zone_total,
                    "back_court_ratio": sum((group["back_court_ratio"] or 0.0) * group["valid_frames"] for group in groups) / weighted_zone_total,
                }
            )
        )

    shuttle_rallies: list[dict[str, Any]] = []
    all_shuttle_speeds: list[float] = []
    for (video_stem, rally_id), rows in sorted(shuttle_by_rally.items()):
        metrics = _shuttle_metrics(rows, frame_diagonal)
        all_shuttle_speeds.extend(metrics["_speed_samples"])
        metrics.update({"video_stem": video_stem, "rally_id": rally_id})
        shuttle_rallies.append(_public_metrics(metrics))
    shuttle_valid = sum(row["valid_observed_frames"] for row in shuttle_rallies)
    shuttle_expected = sum(row["expected_frames"] for row in shuttle_rallies)
    shuttle_average = _mean(all_shuttle_speeds)

    report = {
        "run_id": summary.get("run_id", run_dir.name),
        "status": summary.get("status", "unknown"),
        "run_dir": str(run_dir),
        "match": {
            "usable_rallies": len(rally_rows),
            "analyzed_duration_seconds": _round(
                sum(_float(row.get("duration_seconds")) or 0.0 for row in rally_rows)
            ),
            "analysis_video": str(analysis_video),
        },
        "players": players,
        "player_rallies": rally_player_metrics,
        "shuttle": _public_metrics(
            {
                "valid_observed_frames": shuttle_valid,
                "expected_frames": shuttle_expected,
                "visibility_ratio": shuttle_valid / shuttle_expected if shuttle_expected else None,
                "average_image_speed_px_s": shuttle_average,
                "maximum_image_speed_px_s": max(all_shuttle_speeds) if all_shuttle_speeds else None,
                "current_image_speed_px_s": all_shuttle_speeds[-1] if all_shuttle_speeds else None,
                "average_screen_diagonals_s": shuttle_average / frame_diagonal if shuttle_average is not None and frame_diagonal else None,
            }
        ),
        "shuttle_rallies": shuttle_rallies,
        "quality": {
            "metric_contract": "Player movement uses validated court-plane metres. Shuttle speed is image-plane only because an airborne shuttle cannot be projected to the ground plane as a physical speed.",
            "body_center_contract": "Body-center height is a 2-D pose estimate normalized by the person bounding-box height; it is not a 3-D centre-of-mass height in metres.",
            "dual_side_capability": "near/far are court-side roles; dual-side tactical conclusions remain experimental.",
        },
        "development": {
            "bone_action_detail": "正在开发中",
            "planned_items": ["击球动作分类", "挥拍阶段分解", "关节角度与稳定性", "步法与启动模式"],
        },
        "outputs": {
            **summary.get("outputs", {}),
            "web_report": str(run_dir / "webui_report.json"),
        },
    }
    write_json(run_dir / "webui_report.json", report)
    return report

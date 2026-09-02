from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from badminton_data_process.calibration.reference import (
    COURT_LENGTH_M,
    COURT_WIDTH_M,
    STANDARD_COURT,
)
from badminton_data_process.core.io import ensure_parent, read_csv_rows, read_json
from badminton_data_process.tracking.player.pose import (
    pose_keypoints_from_json,
    skeleton_segments,
)

try:
    import cv2
except ImportError:  # pragma: no cover - runtime dependency
    cv2 = None


PLAYER_COLORS = {"near": (64, 96, 255), "far": (255, 160, 48)}

# Regulation doubles outline with singles sidelines and both singles/doubles
# service markings. Coordinates use the same 6.10 m x 13.40 m court space as
# calibration and player Observation artifacts.
_OUTER_COURT_LINES = {
    "far_baseline",
    "right_doubles_sideline",
    "near_baseline",
    "left_doubles_sideline",
}
COURT_MARKINGS_M = {
    line.name.replace("left_singles", "singles_left").replace("right_singles", "singles_right"): (
        line.start,
        line.end,
    )
    for line in STANDARD_COURT.lines
    if line.name not in _OUTER_COURT_LINES
}


def require_opencv() -> None:
    if cv2 is None:
        raise RuntimeError("OpenCV is required to render a demo video")


def parse_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_int(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def group_rows_by_rally(
    rows: list[dict[str, str]],
) -> dict[tuple[str, str], list[dict[str, str]]]:
    """Group frame data by its real identity, not by the reusable frame id."""
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("video_stem", ""), row.get("rally_id", ""))].append(row)
    return dict(grouped)


def index_rows_by_frame(
    rows: list[dict[str, str]],
    nested_key: str | None = None,
) -> dict[int, Any]:
    indexed: dict[int, Any] = {}
    for row in rows:
        frame_id = parse_int(row.get("frame_id"))
        if frame_id is None:
            continue
        if nested_key is None:
            indexed[frame_id] = row
        else:
            indexed.setdefault(frame_id, {})[row.get(nested_key, "")] = row
    return indexed


def load_calibration(calibration_dir: Path, video_stem: str) -> dict[str, Any]:
    path = calibration_dir / f"{video_stem}.json"
    if not path.exists():
        return {}
    return read_json(path)


def preferred_point(row: dict[str, str], x_key: str, y_key: str) -> tuple[float, float] | None:
    smoothed_x = parse_float(row.get(f"smoothed_{x_key}"))
    smoothed_y = parse_float(row.get(f"smoothed_{y_key}"))
    if row.get("is_smoothed_valid") in {"1", "true", "True"} and smoothed_x is not None and smoothed_y is not None:
        return smoothed_x, smoothed_y
    x = parse_float(row.get(x_key))
    y = parse_float(row.get(y_key))
    return (x, y) if x is not None and y is not None else None


def player_image_point(row: dict[str, str]) -> tuple[float, float] | None:
    body_point = preferred_point(row, "body_image_x", "body_image_y")
    if body_point is not None and row.get("body_anchor_valid", "1") not in {"0", "false", "False"}:
        return body_point
    return preferred_point(row, "image_x", "image_y")


def player_court_point(row: dict[str, str]) -> tuple[float, float] | None:
    return preferred_point(row, "court_x", "court_y")


def shuttle_image_point(row: dict[str, str]) -> tuple[float, float] | None:
    if "is_smoothed_valid" in row and row.get("is_smoothed_valid") not in {"1", "true", "True"}:
        return None
    if "is_smoothed_valid" not in row and row.get("visibility", "1") in {"0", "false", "False"}:
        return None
    return preferred_point(row, "x", "y")


def image_to_court(
    point: tuple[float, float] | None,
    homography: np.ndarray | None,
) -> tuple[float, float] | None:
    if point is None or homography is None:
        return None
    source = np.array([point[0], point[1], 1.0], dtype=np.float64)
    projected = homography @ source
    if abs(float(projected[2])) < 1e-9:
        return None
    return float(projected[0] / projected[2]), float(projected[1] / projected[2])


def court_to_image(
    point: tuple[float, float],
    homography: np.ndarray | None,
) -> tuple[float, float] | None:
    if homography is None:
        return None
    try:
        inverse = np.linalg.inv(homography)
    except np.linalg.LinAlgError:
        return None
    return image_to_court(point, inverse)


def scale_point(
    point: tuple[float, float],
    scale_x: float,
    scale_y: float,
) -> tuple[float, float]:
    return point[0] * scale_x, point[1] * scale_y


def _draw_label(
    frame,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int] = (255, 255, 255),
    scale: float = 0.55,
) -> None:
    x, y = origin
    (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    cv2.rectangle(frame, (x - 4, y - height - 5), (x + width + 4, y + baseline + 3), (18, 18, 18), -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _draw_court_outline(
    frame,
    calibration: dict[str, Any],
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> np.ndarray | None:
    corners = calibration.get("image_points_tl_tr_br_bl")
    if corners:
        polygon = np.asarray(
            [scale_point((float(point[0]), float(point[1])), scale_x, scale_y) for point in corners],
            dtype=np.int32,
        )
        cv2.polylines(frame, [polygon], True, (0, 220, 255), 2, cv2.LINE_AA)
    matrix = calibration.get("homography_image_to_court")
    if not matrix:
        return None
    homography = np.asarray(matrix, dtype=np.float64)
    if homography.shape != (3, 3):
        return None
    display_to_source = np.asarray(
        [[1.0 / scale_x, 0.0, 0.0], [0.0, 1.0 / scale_y, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return homography @ display_to_source


def _draw_players(
    frame,
    rows: dict[str, dict[str, str]],
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> dict[str, tuple[float, float]]:
    positions: dict[str, tuple[float, float]] = {}
    for player_id, row in rows.items():
        color = PLAYER_COLORS.get(player_id, (240, 240, 240))
        pose_threshold = parse_float(row.get("pose_keypoint_threshold")) or 0.35
        pose_keypoints = pose_keypoints_from_json(row.get("pose_keypoints_json"))
        if row.get("pose_valid") in {"1", "true", "True"} and pose_keypoints:
            for start, end in skeleton_segments(pose_keypoints, pose_threshold):
                start_point = scale_point((start.x, start.y), scale_x, scale_y)
                end_point = scale_point((end.x, end.y), scale_x, scale_y)
                cv2.line(
                    frame,
                    (round(start_point[0]), round(start_point[1])),
                    (round(end_point[0]), round(end_point[1])),
                    color,
                    2,
                    cv2.LINE_AA,
                )
            for keypoint in pose_keypoints:
                if keypoint.confidence < pose_threshold:
                    continue
                point = scale_point((keypoint.x, keypoint.y), scale_x, scale_y)
                cv2.circle(
                    frame,
                    (round(point[0]), round(point[1])),
                    3,
                    (245, 245, 245),
                    -1,
                    cv2.LINE_AA,
                )
        values = [parse_float(row.get(key)) for key in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")]
        if all(value is not None for value in values):
            x1 = int(round(values[0] * scale_x))
            y1 = int(round(values[1] * scale_y))
            x2 = int(round(values[2] * scale_x))
            y2 = int(round(values[3] * scale_y))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            _draw_label(frame, player_id, (x1, max(18, y1 - 6)), color)
        point = player_image_point(row)
        if point is not None:
            point = scale_point(point, scale_x, scale_y)
            positions[player_id] = point
            cv2.circle(frame, (int(round(point[0])), int(round(point[1]))), 5, color, -1, cv2.LINE_AA)
    return positions


def _draw_shuttle(
    frame,
    row: dict[str, str] | None,
    trail: list[tuple[int, int]],
    trail_length: int,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> tuple[float, float] | None:
    point = shuttle_image_point(row) if row else None
    if point is None:
        trail.clear()
        return None
    point = scale_point(point, scale_x, scale_y)
    pixel = (int(round(point[0])), int(round(point[1])))
    interpolated = row.get("is_gap_filled") in {"1", "true", "True"} or row.get("is_interpolated") in {"1", "true", "True"}
    color = (0, 170, 255) if interpolated else (60, 255, 60)
    trail.append(pixel)
    del trail[:-trail_length]
    for index in range(1, len(trail)):
        intensity = index / max(1, len(trail) - 1)
        cv2.line(frame, trail[index - 1], trail[index], (0, int(110 + 145 * intensity), 0), max(1, int(3 * intensity)), cv2.LINE_AA)
    cv2.circle(frame, pixel, 7, color, 2, cv2.LINE_AA)
    return point


def _draw_events(
    frame,
    frame_id: int,
    events: list[dict[str, str]],
    homography: np.ndarray | None,
    event_hold_frames: int,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> None:
    active = []
    for event in events:
        event_frame = parse_int(event.get("frame_id"))
        if event_frame is not None and 0 <= frame_id - event_frame <= event_hold_frames:
            active.append(event)
    if not active:
        return
    event = active[-1]
    event_type = event.get("event_type", "event").upper()
    player_id = event.get("player_id", "")
    color = (0, 255, 255)
    image_x = parse_float(event.get("image_x"))
    image_y = parse_float(event.get("image_y"))
    if image_x is not None and image_y is not None:
        point = scale_point((image_x, image_y), scale_x, scale_y)
    else:
        court_x = parse_float(event.get("court_x"))
        court_y = parse_float(event.get("court_y"))
        point = court_to_image((court_x, court_y), homography) if court_x is not None and court_y is not None else None
    if point is not None:
        center = (int(round(point[0])), int(round(point[1])))
        cv2.circle(frame, center, 15, color, 3, cv2.LINE_AA)
    _draw_label(frame, f"{event_type}  {player_id}", (14, 58), color, 0.7)


def _draw_biomechanics(
    frame,
    frame_id: int,
    action_events: list[dict[str, str]],
    swing_phases: list[dict[str, str]],
    event_hold_frames: int,
) -> None:
    active_event = None
    for event in action_events:
        candidate = parse_int(event.get("candidate_frame"))
        if candidate is not None and abs(frame_id - candidate) <= event_hold_frames:
            active_event = event
    active_phase = None
    for phase in swing_phases:
        start = parse_int(phase.get("start_frame"))
        end = parse_int(phase.get("end_frame"))
        if (
            phase.get("phase_eligibility") == "eligible"
            and start is not None
            and end is not None
            and start <= frame_id < end
        ):
            active_phase = phase
            break
    if active_event is not None:
        player = active_event.get("player_id", "")
        score = parse_float(active_event.get("candidate_score"))
        stroke = (
            active_event.get("stroke_class", "")
            if active_event.get("classification_eligibility") == "eligible"
            else "stroke candidate"
        )
        score_text = f" {score:.2f}" if score is not None else ""
        _draw_label(
            frame,
            f"ACTION  {player}  {stroke}{score_text}",
            (14, 88),
            PLAYER_COLORS.get(player, (255, 255, 255)),
            0.55,
        )
    if active_phase is not None:
        side = active_phase.get("motion_side_candidate", "")
        _draw_label(
            frame,
            f"PHASE  {active_phase.get('phase', '')}  motion-side:{side}",
            (14, 116),
            (120, 255, 220),
            0.52,
        )


def _draw_stats(
    frame,
    stats: dict[str, dict[str, str]],
    rally_id: str,
    rally_index: int,
    rally_count: int,
) -> None:
    _draw_label(frame, f"Rally {rally_id}  {rally_index}/{rally_count}", (14, 28), (255, 255, 255), 0.65)
    _draw_label(frame, "Full-rally summary", (14, 58), (180, 180, 180), 0.5)
    y = 88
    for player_id in ("far", "near"):
        row = stats.get(player_id)
        if not row:
            continue
        color = PLAYER_COLORS[player_id]
        if row.get("movement_eligibility") == "not_eligible":
            movement_text = "movement N/A"
        else:
            distance = parse_float(row.get("total_distance_m")) or 0.0
            speed = parse_float(row.get("avg_speed_m_s")) or 0.0
            movement_text = f"{distance:.1f}m  {speed:.1f}m/s"
        if row.get("event_eligibility") == "not_eligible":
            event_text = "events N/A"
        else:
            candidates = parse_int(row.get("reversal_candidate_count")) or 0
            event_text = f"reversal candidates {candidates}"
        _draw_label(
            frame,
            f"{player_id}: {movement_text}  {event_text}",
            (14, y),
            color,
            0.5,
        )
        y += 27


def _topdown_pixel(
    point: tuple[float, float],
    rect: tuple[int, int, int, int],
) -> tuple[int, int] | None:
    if not (0.0 <= point[0] <= COURT_WIDTH_M and 0.0 <= point[1] <= COURT_LENGTH_M):
        return None
    left, top, width, height = rect
    x = left + int(round(point[0] / COURT_WIDTH_M * width))
    y = top + int(round(point[1] / COURT_LENGTH_M * height))
    return x, y


def _court_marking_segments(
    rect: tuple[int, int, int, int],
) -> dict[str, tuple[tuple[int, int], tuple[int, int]]]:
    """Map regulation court markings into the top-down panel rectangle."""
    segments: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {}
    for name, (start_m, end_m) in COURT_MARKINGS_M.items():
        start = _topdown_pixel(start_m, rect)
        end = _topdown_pixel(end_m, rect)
        if start is not None and end is not None:
            segments[name] = (start, end)
    return segments


def _draw_topdown(
    frame,
    player_rows: dict[str, dict[str, str]],
) -> None:
    frame_height, frame_width = frame.shape[:2]
    court_height = min(
        max(120, frame_height - 76),
        max(164, int(round(frame_height * 0.62))),
    )
    court_width = max(70, int(round(court_height * COURT_WIDTH_M / COURT_LENGTH_M)))
    left = max(8, frame_width - court_width - 20)
    top = 38
    rect = (left, top, court_width, court_height)
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (left - 8, top - 26),
        (left + court_width + 8, top + court_height + 8),
        (18, 45, 24),
        -1,
    )
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    cv2.rectangle(frame, (left, top), (left + court_width, top + court_height), (235, 235, 235), 2)
    for name, (start, end) in _court_marking_segments(rect).items():
        if name == "net":
            cv2.line(frame, start, end, (80, 220, 255), 2, cv2.LINE_AA)
        else:
            cv2.line(frame, start, end, (225, 225, 225), 1, cv2.LINE_AA)
    cv2.putText(
        frame,
        "COURT 6.10 x 13.40m",
        (left - 4, top - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.32,
        (225, 225, 225),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "FAR",
        (left + 5, top + 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        PLAYER_COLORS["far"],
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "NEAR",
        (left + 5, top + court_height - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        PLAYER_COLORS["near"],
        1,
        cv2.LINE_AA,
    )
    rejected = 0
    for player_id, row in player_rows.items():
        point = player_court_point(row)
        if point is not None:
            pixel = _topdown_pixel(point, rect)
            if pixel is None:
                rejected += 1
                continue
            color = PLAYER_COLORS.get(player_id, (255, 255, 255))
            cv2.circle(frame, pixel, 6, (20, 20, 20), -1, cv2.LINE_AA)
            cv2.circle(frame, pixel, 5, color, -1, cv2.LINE_AA)
            label_x = pixel[0] + 7 if pixel[0] < left + court_width - 34 else pixel[0] - 31
            label_y = min(top + court_height - 4, max(top + 12, pixel[1] - 5))
            cv2.putText(
                frame,
                player_id,
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                color,
                1,
                cv2.LINE_AA,
            )
    if rejected:
        _draw_label(frame, f"projection rejected: {rejected}", (left - 4, top + court_height + 28), (80, 80, 255), 0.42)


def should_show_full_rally_summary(frame_id: int, total_frames: int) -> bool:
    return total_frames > 0 and frame_id == total_frames - 1


def _rally_rows(
    grouped: dict[tuple[str, str], list[dict[str, str]]],
    video_stem: str,
    rally_id: str,
) -> list[dict[str, str]]:
    return grouped.get((video_stem, rally_id), [])


def render_demo(
    rallies_csv: Path,
    player_tracks_csv: Path,
    shuttle_tracks_csv: Path,
    calibration_dir: Path,
    output_video: Path,
    tactics_events_csv: Path | None = None,
    tactics_summary_csv: Path | None = None,
    action_events_csv: Path | None = None,
    swing_phases_csv: Path | None = None,
    max_rallies: int | None = None,
    trail_length: int = 18,
    event_hold_frames: int = 15,
    show_topdown: bool = True,
    show_stats: bool = True,
    codec: str = "mp4v",
) -> dict[str, object]:
    require_opencv()
    if max_rallies is not None and max_rallies <= 0:
        raise ValueError("max_rallies must be greater than zero")
    if trail_length <= 0:
        raise ValueError("trail_length must be greater than zero")
    if event_hold_frames < 0:
        raise ValueError("event_hold_frames must be non-negative")
    if len(codec) != 4:
        raise ValueError("codec must contain exactly four characters")

    rally_rows = read_csv_rows(rallies_csv)
    selected: list[tuple[dict[str, str], Path]] = []
    for row in rally_rows:
        video_path = Path(row.get("output_path", ""))
        if video_path.exists():
            selected.append((row, video_path))
        if max_rallies is not None and len(selected) >= max_rallies:
            break
    if not selected:
        raise RuntimeError(f"No readable rally videos found in {rallies_csv}")

    players = group_rows_by_rally(read_csv_rows(player_tracks_csv))
    shuttles = group_rows_by_rally(read_csv_rows(shuttle_tracks_csv))
    events = group_rows_by_rally(read_csv_rows(tactics_events_csv)) if tactics_events_csv else {}
    summaries = group_rows_by_rally(read_csv_rows(tactics_summary_csv)) if tactics_summary_csv else {}
    action_events = group_rows_by_rally(read_csv_rows(action_events_csv)) if action_events_csv else {}
    swing_phases = group_rows_by_rally(read_csv_rows(swing_phases_csv)) if swing_phases_csv else {}

    first_capture = cv2.VideoCapture(str(selected[0][1]))
    if not first_capture.isOpened():
        raise RuntimeError(f"Cannot open rally video: {selected[0][1]}")
    output_fps = float(first_capture.get(cv2.CAP_PROP_FPS) or 30.0)
    output_width = int(first_capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    output_height = int(first_capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    first_capture.release()
    if output_width <= 0 or output_height <= 0:
        raise RuntimeError(f"Invalid video dimensions: {selected[0][1]}")

    ensure_parent(output_video)
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*codec),
        output_fps,
        (output_width, output_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create demo video: {output_video}")

    rendered_frames = 0
    rendered_rallies = 0
    try:
        for rally_index, (rally, video_path) in enumerate(selected, start=1):
            video_stem = video_path.stem
            rally_id = rally.get("rally_id", "")
            player_frames = index_rows_by_frame(_rally_rows(players, video_stem, rally_id), "player_id")
            shuttle_frames = index_rows_by_frame(_rally_rows(shuttles, video_stem, rally_id))
            event_rows = _rally_rows(events, video_stem, rally_id)
            action_rows = _rally_rows(action_events, video_stem, rally_id)
            phase_rows = _rally_rows(swing_phases, video_stem, rally_id)
            stat_rows = {
                row.get("player_id", ""): row
                for row in _rally_rows(summaries, video_stem, rally_id)
            }
            calibration = load_calibration(calibration_dir, video_stem)

            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                continue
            source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if source_width <= 0 or source_height <= 0:
                capture.release()
                continue
            scale_x = output_width / source_width
            scale_y = output_height / source_height
            frame_id = 0
            trail: list[tuple[int, int]] = []
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame.shape[1] != output_width or frame.shape[0] != output_height:
                    frame = cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_AREA)
                homography = _draw_court_outline(frame, calibration, scale_x, scale_y)
                current_players = player_frames.get(frame_id, {})
                _draw_players(frame, current_players, scale_x, scale_y)
                _draw_shuttle(
                    frame,
                    shuttle_frames.get(frame_id),
                    trail,
                    trail_length,
                    scale_x,
                    scale_y,
                )
                _draw_events(
                    frame,
                    frame_id,
                    event_rows,
                    homography,
                    event_hold_frames,
                    scale_x,
                    scale_y,
                )
                _draw_biomechanics(
                    frame,
                    frame_id,
                    action_rows,
                    phase_rows,
                    event_hold_frames,
                )
                if show_stats and should_show_full_rally_summary(frame_id, total_frames):
                    _draw_stats(frame, stat_rows, rally_id, rally_index, len(selected))
                if show_topdown:
                    _draw_topdown(frame, current_players)
                writer.write(frame)
                frame_id += 1
                rendered_frames += 1
            capture.release()
            if frame_id:
                rendered_rallies += 1
    finally:
        writer.release()

    if rendered_frames == 0:
        output_video.unlink(missing_ok=True)
        raise RuntimeError("No frames were rendered from the selected rally videos")
    return {
        "output_video": str(output_video),
        "rallies": rendered_rallies,
        "frames": rendered_frames,
        "fps": output_fps,
        "width": output_width,
        "height": output_height,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a combined annotated demo from pipeline outputs.")
    parser.add_argument("--rallies-csv", type=Path, required=True)
    parser.add_argument("--player-tracks-csv", type=Path, required=True)
    parser.add_argument("--shuttle-tracks-csv", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--tactics-events-csv", type=Path, default=None)
    parser.add_argument("--tactics-summary-csv", type=Path, default=None)
    parser.add_argument("--action-events-csv", type=Path, default=None)
    parser.add_argument("--swing-phases-csv", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-rallies", type=int, default=None)
    parser.add_argument("--trail-length", type=int, default=18)
    parser.add_argument("--event-hold-frames", type=int, default=15)
    parser.add_argument("--no-topdown", action="store_true")
    parser.add_argument("--no-stats", action="store_true")
    parser.add_argument("--codec", default="mp4v")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = render_demo(
        rallies_csv=args.rallies_csv,
        player_tracks_csv=args.player_tracks_csv,
        shuttle_tracks_csv=args.shuttle_tracks_csv,
        calibration_dir=args.calibration_dir,
        tactics_events_csv=args.tactics_events_csv,
        tactics_summary_csv=args.tactics_summary_csv,
        action_events_csv=args.action_events_csv,
        swing_phases_csv=args.swing_phases_csv,
        output_video=args.output,
        max_rallies=args.max_rallies,
        trail_length=args.trail_length,
        event_hold_frames=args.event_hold_frames,
        show_topdown=not args.no_topdown,
        show_stats=not args.no_stats,
        codec=args.codec,
    )
    print(
        f"Rendered {result['rallies']} rallies / {result['frames']} frames "
        f"-> {result['output_video']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

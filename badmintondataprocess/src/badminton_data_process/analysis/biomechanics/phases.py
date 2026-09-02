from __future__ import annotations

import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping

from badminton_data_process.core.io import read_csv_rows, write_csv_rows
from badminton_data_process.tracking.player.pose import PoseKeypoint, pose_keypoints_from_json


SWING_PHASE_VERSION = "bba_2d_swing_phase_v1"
SWING_PHASE_FIELDS = [
    "video_stem",
    "rally_id",
    "event_id",
    "player_id",
    "phase",
    "phase_order",
    "start_frame",
    "end_frame",
    "frame_interval",
    "duration_seconds",
    "phase_eligibility",
    "phase_reject_reason",
    "phase_confidence",
    "motion_side_candidate",
    "boundary_evidence",
    "phase_version",
]


def _number(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _frame(row: Mapping[str, object]) -> int | None:
    value = _number(row.get("frame_id"))
    return int(value) if value is not None else None


def _valid_points(row: Mapping[str, object], threshold: float) -> dict[str, PoseKeypoint]:
    return {
        point.name: point
        for point in pose_keypoints_from_json(str(row.get("pose_keypoints_json") or ""))
        if point.confidence >= threshold
        and math.isfinite(point.x)
        and math.isfinite(point.y)
    }


def _relative_wrists(
    row: Mapping[str, object], threshold: float
) -> dict[str, tuple[float, float]]:
    points = _valid_points(row, threshold)
    required = {"left_shoulder", "right_shoulder", "left_hip", "right_hip"}
    if not required.issubset(points):
        return {}
    top = _number(row.get("bbox_y1"))
    bottom = _number(row.get("bbox_y2"))
    if top is None or bottom is None or bottom <= top:
        return {}
    torso_x = sum(points[name].x for name in required) / 4.0
    torso_y = sum(points[name].y for name in required) / 4.0
    height = bottom - top
    return {
        side: (
            (points[f"{side}_wrist"].x - torso_x) / height,
            (points[f"{side}_wrist"].y - torso_y) / height,
        )
        for side in ("left", "right")
        if f"{side}_wrist" in points
    }


def _side_motion(
    rows: list[Mapping[str, object]], threshold: float
) -> tuple[str | None, dict[int, float]]:
    observations: dict[str, list[tuple[int, float, tuple[float, float]]]] = defaultdict(list)
    for row in rows:
        frame = _frame(row)
        timestamp = _number(row.get("timestamp"))
        if frame is None or timestamp is None:
            continue
        for side, position in _relative_wrists(row, threshold).items():
            observations[side].append((frame, timestamp, position))

    path_lengths: dict[str, float] = {}
    speed_by_side: dict[str, dict[int, float]] = {}
    for side, samples in observations.items():
        samples.sort()
        path = 0.0
        speeds: dict[int, float] = {}
        for previous, current in zip(samples, samples[1:]):
            delta_time = current[1] - previous[1]
            if delta_time <= 0 or current[0] - previous[0] > 2:
                continue
            distance = math.hypot(
                current[2][0] - previous[2][0], current[2][1] - previous[2][1]
            )
            path += distance
            speeds[current[0]] = distance / delta_time
        path_lengths[side] = path
        speed_by_side[side] = speeds
    if not path_lengths:
        return None, {}
    side = max(path_lengths, key=path_lengths.get)
    return side, speed_by_side[side]


def _smooth(signal: dict[int, float], window: int) -> dict[int, float]:
    radius = window // 2
    return {
        frame: statistics.median(
            value
            for neighbor, value in signal.items()
            if abs(neighbor - frame) <= radius
        )
        for frame in signal
    }


def _longest_contiguous(frames: list[int]) -> int:
    longest = current = 0
    previous: int | None = None
    for frame in sorted(set(frames)):
        current = current + 1 if previous is not None and frame == previous + 1 else 1
        longest = max(longest, current)
        previous = frame
    return longest


def _phase_rows_for_rejection(
    event: Mapping[str, object], reason: str
) -> list[dict[str, object]]:
    return [
        {
            "video_stem": event.get("video_stem", ""),
            "rally_id": event.get("rally_id", ""),
            "event_id": event.get("event_id", ""),
            "player_id": event.get("player_id", ""),
            "phase": "",
            "phase_order": "",
            "start_frame": "",
            "end_frame": "",
            "frame_interval": "",
            "duration_seconds": "",
            "phase_eligibility": "not_eligible",
            "phase_reject_reason": reason,
            "phase_confidence": "",
            "motion_side_candidate": "",
            "boundary_evidence": "",
            "phase_version": SWING_PHASE_VERSION,
        }
    ]


def decompose_swing_phases(
    player_rows: Iterable[Mapping[str, object]],
    event_rows: Iterable[Mapping[str, object]],
    *,
    keypoint_threshold: float = 0.35,
    min_contiguous_frames: int = 5,
    smoothing_window: int = 5,
) -> list[dict[str, object]]:
    sequences: dict[tuple[str, str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in player_rows:
        sequences[
            (
                str(row.get("video_stem", "")),
                str(row.get("rally_id", "")),
                str(row.get("player_id", "")),
            )
        ].append(row)
    for sequence in sequences.values():
        sequence.sort(key=lambda row: _frame(row) or -1)

    output: list[dict[str, object]] = []
    for event in event_rows:
        key = (
            str(event.get("video_stem", "")),
            str(event.get("rally_id", "")),
            str(event.get("player_id", "")),
        )
        candidate_value = _number(event.get("candidate_frame"))
        start_value = _number(event.get("window_start_frame"))
        end_value = _number(event.get("window_end_frame"))
        if candidate_value is None or start_value is None or end_value is None:
            output.extend(_phase_rows_for_rejection(event, "invalid_event_window"))
            continue
        candidate, requested_start, requested_end = (
            int(candidate_value),
            int(start_value),
            int(end_value),
        )
        window_rows = [
            row
            for row in sequences.get(key, [])
            if (frame := _frame(row)) is not None
            and requested_start <= frame < requested_end
        ]
        frames = [_frame(row) for row in window_rows]
        valid_frames = [frame for frame in frames if frame is not None]
        if not valid_frames or _longest_contiguous(valid_frames) < min_contiguous_frames:
            output.extend(_phase_rows_for_rejection(event, "insufficient_contiguous_pose"))
            continue
        actual_start, actual_end = min(valid_frames), max(valid_frames) + 1
        if candidate - actual_start < 2 or actual_end - candidate < 3:
            output.extend(_phase_rows_for_rejection(event, "truncated_event_window"))
            continue
        side, raw_signal = _side_motion(window_rows, keypoint_threshold)
        if side is None or len(raw_signal) < min_contiguous_frames - 1:
            output.extend(_phase_rows_for_rejection(event, "insufficient_wrist_motion"))
            continue
        signal = _smooth(raw_signal, smoothing_window)
        peak = max(signal.values(), default=0.0)
        if peak <= 1e-9:
            output.extend(_phase_rows_for_rejection(event, "no_motion_peak"))
            continue
        threshold = peak * 0.35
        contact_start = max(actual_start + 1, candidate - 1)
        contact_end = min(actual_end - 1, candidate + 2)
        active_before = sorted(
            frame for frame, value in signal.items() if frame < candidate and value >= threshold
        )
        if not active_before:
            output.extend(_phase_rows_for_rejection(event, "no_acceleration_onset"))
            continue
        acceleration_start = active_before[0]
        acceleration_start = min(acceleration_start, contact_start - 1)
        recovery_start = actual_end
        post_frames = sorted(frame for frame in signal if frame >= contact_end)
        for first, second in zip(post_frames, post_frames[1:]):
            if second == first + 1 and signal[first] < threshold and signal[second] < threshold:
                recovery_start = first
                break
        recovery_start = max(contact_end + 1, recovery_start)
        boundaries = [
            ("preparation", actual_start, acceleration_start, "window_start->motion_onset"),
            ("acceleration", acceleration_start, contact_start, "motion_onset->contact_window"),
            ("contact_window", contact_start, contact_end, "candidate_frame±1"),
            ("follow_through", contact_end, min(recovery_start, actual_end), "contact_window->motion_decay"),
            ("recovery", min(recovery_start, actual_end), actual_end, "motion_decay->window_end"),
        ]
        timestamps = sorted(
            timestamp
            for row in window_rows
            if (timestamp := _number(row.get("timestamp"))) is not None
        )
        frame_periods = [
            b - a for a, b in zip(timestamps, timestamps[1:]) if b > a
        ]
        frame_period = statistics.median(frame_periods) if frame_periods else 0.0
        coverage = len(set(valid_frames)) / max(1, actual_end - actual_start)
        candidate_score = _number(event.get("candidate_score")) or 0.0
        confidence = round(_clamp(0.55 * coverage + 0.45 * candidate_score), 4)
        order = 0
        for phase, start, end, evidence in boundaries:
            if end <= start:
                continue
            order += 1
            output.append(
                {
                    "video_stem": key[0],
                    "rally_id": key[1],
                    "event_id": event.get("event_id", ""),
                    "player_id": key[2],
                    "phase": phase,
                    "phase_order": order,
                    "start_frame": start,
                    "end_frame": end,
                    "frame_interval": f"[{start},{end})",
                    "duration_seconds": round((end - start) * frame_period, 4),
                    "phase_eligibility": "eligible",
                    "phase_reject_reason": "",
                    "phase_confidence": confidence,
                    "motion_side_candidate": side,
                    "boundary_evidence": evidence,
                    "phase_version": SWING_PHASE_VERSION,
                }
            )
    return output


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, value))


def analyze_swing_phases_csv(
    player_tracks_csv: Path,
    action_events_csv: Path,
    output_csv: Path,
    **kwargs: object,
) -> dict[str, object]:
    rows = decompose_swing_phases(
        read_csv_rows(player_tracks_csv), read_csv_rows(action_events_csv), **kwargs
    )
    write_csv_rows(output_csv, SWING_PHASE_FIELDS, rows)
    return {
        "phase_rows": len(rows),
        "eligible_phase_rows": sum(row["phase_eligibility"] == "eligible" for row in rows),
        "output_csv": str(output_csv),
        "phase_version": SWING_PHASE_VERSION,
    }

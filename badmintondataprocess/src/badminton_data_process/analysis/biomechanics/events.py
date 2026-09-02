from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping

from badminton_data_process.core.io import read_csv_rows, write_csv_rows
from badminton_data_process.tracking.player.pose import (
    PoseKeypoint,
    pose_keypoints_from_json,
)


ACTION_EVENT_VERSION = "bba_stroke_candidate_v1"

ACTION_EVENT_FIELDS = [
    "video_path",
    "video_stem",
    "rally_id",
    "event_id",
    "candidate_frame",
    "candidate_timestamp",
    "player_id",
    "event_eligibility",
    "event_reject_reason",
    "candidate_score",
    "evidence_source",
    "evidence_count",
    "shuttle_turn_score",
    "shuttle_turn_angle_deg",
    "shuttle_proximity_score",
    "shuttle_wrist_distance_ratio",
    "wrist_motion_score",
    "wrist_speed_norm_s",
    "window_start_frame",
    "window_end_frame",
    "classification_eligibility",
    "classification_reject_reason",
    "stroke_class",
    "stroke_class_zh",
    "top2_json",
    "classification_confidence",
    "model_id",
    "stability_eligibility",
    "stability_reject_reason",
    "mean_support_width_ratio",
    "body_support_offset_rms",
    "trunk_sway_std_deg",
    "mean_knee_asymmetry_deg",
    "candidate_left_elbow_angle_deg",
    "candidate_right_elbow_angle_deg",
    "candidate_left_shoulder_angle_deg",
    "candidate_right_shoulder_angle_deg",
    "candidate_left_knee_angle_deg",
    "candidate_right_knee_angle_deg",
    "candidate_trunk_lean_deg",
    "wide_support_candidate",
    "footwork_eligibility",
    "footwork_reject_reason",
    "pre_contact_displacement_m",
    "post_contact_displacement_m",
    "event_path_distance_m",
    "max_footwork_speed_m_s",
    "recovery_time_s",
    "movement_start_frame",
    "braking_frame",
    "recovery_frame",
    "cross_step_candidate",
    "descriptor_version",
    "event_version",
]


def _float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _int(value: object) -> int | None:
    number = _float(value)
    return int(number) if number is not None else None


def _true(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


def _points(row: Mapping[str, object], threshold: float) -> dict[str, PoseKeypoint]:
    return {
        point.name: point
        for point in pose_keypoints_from_json(str(row.get("pose_keypoints_json") or ""))
        if point.confidence >= threshold
        and math.isfinite(point.x)
        and math.isfinite(point.y)
    }


def _midpoint(
    points: Mapping[str, PoseKeypoint], first: str, second: str
) -> tuple[float, float] | None:
    if first not in points or second not in points:
        return None
    return (
        (points[first].x + points[second].x) / 2.0,
        (points[first].y + points[second].y) / 2.0,
    )


def _bbox_height(row: Mapping[str, object]) -> float | None:
    top = _float(row.get("bbox_y1"))
    bottom = _float(row.get("bbox_y2"))
    if top is None or bottom is None or bottom <= top:
        return None
    return bottom - top


def _torso_center(points: Mapping[str, PoseKeypoint]) -> tuple[float, float] | None:
    shoulders = _midpoint(points, "left_shoulder", "right_shoulder")
    hips = _midpoint(points, "left_hip", "right_hip")
    if shoulders is None:
        return hips
    if hips is None:
        return shoulders
    return ((shoulders[0] + hips[0]) / 2.0, (shoulders[1] + hips[1]) / 2.0)


def _shuttle_is_observed(row: Mapping[str, object], confidence_threshold: float) -> bool:
    return (
        not _true(row.get("is_interpolated"))
        and _true(row.get("visibility"))
        and (_float(row.get("confidence")) or 0.0) >= confidence_threshold
        and _float(row.get("x")) is not None
        and _float(row.get("y")) is not None
    )


def _turn_evidence(
    previous: Mapping[str, object] | None,
    current: Mapping[str, object],
    following: Mapping[str, object] | None,
    *,
    max_observation_gap_frames: int,
    turn_threshold_deg: float,
) -> tuple[float, float | None]:
    if previous is None or following is None:
        return 0.0, None
    previous_frame = _int(previous.get("frame_id"))
    current_frame = _int(current.get("frame_id"))
    following_frame = _int(following.get("frame_id"))
    if None in (previous_frame, current_frame, following_frame):
        return 0.0, None
    assert previous_frame is not None and current_frame is not None and following_frame is not None
    if (
        current_frame - previous_frame > max_observation_gap_frames
        or following_frame - current_frame > max_observation_gap_frames
    ):
        return 0.0, None
    points = [
        (_float(row.get("x")), _float(row.get("y")))
        for row in (previous, current, following)
    ]
    if any(x is None or y is None for x, y in points):
        return 0.0, None
    (x0, y0), (x1, y1), (x2, y2) = points
    assert None not in (x0, y0, x1, y1, x2, y2)
    incoming = (x1 - x0, y1 - y0)
    outgoing = (x2 - x1, y2 - y1)
    norm_in = math.hypot(*incoming)
    norm_out = math.hypot(*outgoing)
    if norm_in <= 1e-9 or norm_out <= 1e-9:
        return 0.0, None
    cosine = (incoming[0] * outgoing[0] + incoming[1] * outgoing[1]) / (
        norm_in * norm_out
    )
    angle = math.degrees(math.acos(min(1.0, max(-1.0, cosine))))
    return _clamp01(angle / turn_threshold_deg), angle


def _proximity_evidence(
    player_row: Mapping[str, object],
    shuttle_row: Mapping[str, object],
    *,
    keypoint_threshold: float,
    proximity_ratio: float,
) -> tuple[float, float | None]:
    height = _bbox_height(player_row)
    shuttle_x = _float(shuttle_row.get("x"))
    shuttle_y = _float(shuttle_row.get("y"))
    if height is None or shuttle_x is None or shuttle_y is None:
        return 0.0, None
    points = _points(player_row, keypoint_threshold)
    targets = [
        (points[name].x, points[name].y)
        for name in ("left_wrist", "right_wrist")
        if name in points
    ]
    if not targets:
        body_x = _float(player_row.get("body_image_x"))
        body_y = _float(player_row.get("body_image_y"))
        if body_x is not None and body_y is not None:
            targets.append((body_x, body_y))
    if not targets:
        return 0.0, None
    distance_ratio = min(
        math.hypot(shuttle_x - target_x, shuttle_y - target_y) / height
        for target_x, target_y in targets
    )
    score = _clamp01(1.0 - distance_ratio / proximity_ratio)
    return score, distance_ratio


def _relative_wrist_positions(
    row: Mapping[str, object], threshold: float
) -> tuple[dict[str, tuple[float, float]], float | None]:
    points = _points(row, threshold)
    torso = _torso_center(points)
    height = _bbox_height(row)
    if torso is None or height is None:
        return {}, None
    return (
        {
            side: (
                (points[f"{side}_wrist"].x - torso[0]) / height,
                (points[f"{side}_wrist"].y - torso[1]) / height,
            )
            for side in ("left", "right")
            if f"{side}_wrist" in points
        },
        _float(row.get("timestamp")),
    )


def _wrist_motion_evidence(
    previous: Mapping[str, object] | None,
    current: Mapping[str, object],
    following: Mapping[str, object] | None,
    *,
    keypoint_threshold: float,
    speed_threshold_norm_s: float,
    max_pose_gap_frames: int,
) -> tuple[float, float | None]:
    if previous is None or following is None:
        return 0.0, None
    frame_ids = [_int(row.get("frame_id")) for row in (previous, current, following)]
    if any(frame is None for frame in frame_ids):
        return 0.0, None
    previous_frame, current_frame, following_frame = frame_ids
    assert previous_frame is not None and current_frame is not None and following_frame is not None
    if (
        current_frame - previous_frame > max_pose_gap_frames
        or following_frame - current_frame > max_pose_gap_frames
    ):
        return 0.0, None
    previous_wrist, previous_time = _relative_wrist_positions(previous, keypoint_threshold)
    following_wrist, following_time = _relative_wrist_positions(following, keypoint_threshold)
    if previous_time is None or following_time is None or following_time <= previous_time:
        return 0.0, None
    speeds = [
        math.hypot(
            following_wrist[side][0] - previous_wrist[side][0],
            following_wrist[side][1] - previous_wrist[side][1],
        )
        / (following_time - previous_time)
        for side in previous_wrist.keys() & following_wrist.keys()
    ]
    if not speeds:
        return 0.0, None
    speed = max(speeds)
    return _clamp01(speed / speed_threshold_norm_s), speed


def _neighbor_rows(
    rows: list[Mapping[str, object]], index: int
) -> tuple[Mapping[str, object] | None, Mapping[str, object] | None]:
    return (
        rows[index - 1] if index > 0 else None,
        rows[index + 1] if index + 1 < len(rows) else None,
    )


def _nms(candidates: Iterable[dict[str, object]], min_gap_frames: int) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for candidate in sorted(
        candidates,
        key=lambda row: (-float(row["candidate_score"]), int(row["candidate_frame"])),
    ):
        frame = int(candidate["candidate_frame"])
        if all(abs(frame - int(existing["candidate_frame"])) >= min_gap_frames for existing in selected):
            selected.append(candidate)
    return sorted(selected, key=lambda row: int(row["candidate_frame"]))


def detect_action_events(
    player_rows: Iterable[Mapping[str, object]],
    shuttle_rows: Iterable[Mapping[str, object]],
    *,
    keypoint_threshold: float = 0.35,
    shuttle_confidence_threshold: float = 0.15,
    shuttle_turn_angle_deg: float = 45.0,
    shuttle_proximity_ratio: float = 0.75,
    min_shuttle_proximity_score: float = 0.35,
    shuttle_turn_span_observations: int = 3,
    wrist_speed_threshold_norm_s: float = 1.0,
    min_event_gap_frames: int = 8,
    min_candidate_score: float = 0.60,
    event_pre_frames: int = 12,
    event_post_frames: int = 20,
    max_observation_gap_frames: int = 2,
) -> list[dict[str, object]]:
    players_by_group_frame: dict[
        tuple[str, str], dict[int, list[Mapping[str, object]]]
    ] = defaultdict(lambda: defaultdict(list))
    player_sequences: dict[
        tuple[str, str, str], list[Mapping[str, object]]
    ] = defaultdict(list)
    for row in player_rows:
        frame = _int(row.get("frame_id"))
        if frame is None:
            continue
        group = (str(row.get("video_stem", "")), str(row.get("rally_id", "")))
        players_by_group_frame[group][frame].append(row)
        player_sequences[(*group, str(row.get("player_id", "")))].append(row)

    player_neighbors: dict[
        tuple[str, str, str, int],
        tuple[Mapping[str, object] | None, Mapping[str, object] | None],
    ] = {}
    for sequence_key, sequence in player_sequences.items():
        sequence.sort(key=lambda row: _int(row.get("frame_id")) or -1)
        for index, row in enumerate(sequence):
            frame = _int(row.get("frame_id"))
            if frame is not None:
                player_neighbors[(*sequence_key, frame)] = _neighbor_rows(sequence, index)

    shuttles_by_group: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in shuttle_rows:
        if _shuttle_is_observed(row, shuttle_confidence_threshold):
            group = (str(row.get("video_stem", "")), str(row.get("rally_id", "")))
            shuttles_by_group[group].append(row)

    output: list[dict[str, object]] = []
    for group, shuttles in sorted(shuttles_by_group.items()):
        shuttles.sort(key=lambda row: _int(row.get("frame_id")) or -1)
        candidates: list[dict[str, object]] = []
        for shuttle_index, shuttle in enumerate(shuttles):
            frame = _int(shuttle.get("frame_id"))
            if frame is None:
                continue
            turn_span = min(
                shuttle_turn_span_observations,
                shuttle_index,
                len(shuttles) - shuttle_index - 1,
            )
            previous_shuttle = (
                shuttles[shuttle_index - turn_span] if turn_span > 0 else None
            )
            following_shuttle = (
                shuttles[shuttle_index + turn_span] if turn_span > 0 else None
            )
            turn_score, turn_angle = _turn_evidence(
                previous_shuttle,
                shuttle,
                following_shuttle,
                max_observation_gap_frames=max_observation_gap_frames * max(1, turn_span),
                turn_threshold_deg=shuttle_turn_angle_deg,
            )
            best: dict[str, object] | None = None
            for player in players_by_group_frame[group].get(frame, []):
                player_id = str(player.get("player_id", ""))
                previous_player, following_player = player_neighbors.get(
                    (*group, player_id, frame), (None, None)
                )
                proximity_score, distance_ratio = _proximity_evidence(
                    player,
                    shuttle,
                    keypoint_threshold=keypoint_threshold,
                    proximity_ratio=shuttle_proximity_ratio,
                )
                wrist_score, wrist_speed = _wrist_motion_evidence(
                    previous_player,
                    player,
                    following_player,
                    keypoint_threshold=keypoint_threshold,
                    speed_threshold_norm_s=wrist_speed_threshold_norm_s,
                    max_pose_gap_frames=max_observation_gap_frames,
                )
                evidence = []
                if turn_score >= 1.0:
                    evidence.append("shuttle_turn")
                if proximity_score >= min_shuttle_proximity_score:
                    evidence.append("shuttle_proximity")
                if wrist_score >= 1.0:
                    evidence.append("wrist_motion")
                if not {"shuttle_turn", "shuttle_proximity"}.issubset(evidence):
                    continue
                candidate_score = (
                    0.45 * turn_score + 0.35 * proximity_score + 0.20 * wrist_score
                )
                if candidate_score < min_candidate_score:
                    continue
                candidate: dict[str, object] = {
                    "video_path": shuttle.get("video_path", player.get("video_path", "")),
                    "video_stem": group[0],
                    "rally_id": group[1],
                    "event_id": "",
                    "candidate_frame": frame,
                    "candidate_timestamp": shuttle.get("timestamp", ""),
                    "player_id": player_id,
                    "event_eligibility": "eligible",
                    "event_reject_reason": "",
                    "candidate_score": round(candidate_score, 4),
                    "evidence_source": "|".join(evidence),
                    "evidence_count": len(evidence),
                    "shuttle_turn_score": round(turn_score, 4),
                    "shuttle_turn_angle_deg": "" if turn_angle is None else round(turn_angle, 4),
                    "shuttle_proximity_score": round(proximity_score, 4),
                    "shuttle_wrist_distance_ratio": "" if distance_ratio is None else round(distance_ratio, 4),
                    "wrist_motion_score": round(wrist_score, 4),
                    "wrist_speed_norm_s": "" if wrist_speed is None else round(wrist_speed, 4),
                    "window_start_frame": max(0, frame - event_pre_frames),
                    "window_end_frame": frame + event_post_frames + 1,
                    "classification_eligibility": "not_eligible",
                    "classification_reject_reason": "classifier_not_configured",
                    "stroke_class": "",
                    "stroke_class_zh": "",
                    "top2_json": "",
                    "classification_confidence": "",
                    "model_id": "",
                    "stability_eligibility": "pending",
                    "stability_reject_reason": "",
                    "mean_support_width_ratio": "",
                    "body_support_offset_rms": "",
                    "trunk_sway_std_deg": "",
                    "mean_knee_asymmetry_deg": "",
                    "candidate_left_elbow_angle_deg": "",
                    "candidate_right_elbow_angle_deg": "",
                    "candidate_left_shoulder_angle_deg": "",
                    "candidate_right_shoulder_angle_deg": "",
                    "candidate_left_knee_angle_deg": "",
                    "candidate_right_knee_angle_deg": "",
                    "candidate_trunk_lean_deg": "",
                    "wide_support_candidate": "",
                    "footwork_eligibility": "pending",
                    "footwork_reject_reason": "",
                    "pre_contact_displacement_m": "",
                    "post_contact_displacement_m": "",
                    "event_path_distance_m": "",
                    "max_footwork_speed_m_s": "",
                    "recovery_time_s": "",
                    "movement_start_frame": "",
                    "braking_frame": "",
                    "recovery_frame": "",
                    "cross_step_candidate": "",
                    "descriptor_version": "",
                    "event_version": ACTION_EVENT_VERSION,
                }
                if best is None or float(candidate["candidate_score"]) > float(best["candidate_score"]):
                    best = candidate
            if best is not None:
                candidates.append(best)

        selected = _nms(candidates, min_event_gap_frames)
        for index, candidate in enumerate(selected, start=1):
            candidate["event_id"] = f"{group[1] or group[0]}_E{index:03d}"
            output.append(candidate)
    return output


def analyze_action_events_csv(
    player_tracks_csv: Path,
    shuttle_tracks_csv: Path,
    output_csv: Path,
    **kwargs: object,
) -> dict[str, object]:
    rows = detect_action_events(
        read_csv_rows(player_tracks_csv),
        read_csv_rows(shuttle_tracks_csv),
        **kwargs,
    )
    write_csv_rows(output_csv, ACTION_EVENT_FIELDS, rows)
    return {
        "candidate_events": len(rows),
        "output_csv": str(output_csv),
        "event_version": ACTION_EVENT_VERSION,
    }

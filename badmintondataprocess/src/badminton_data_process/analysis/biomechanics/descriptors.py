from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping

from badminton_data_process.analysis.biomechanics.events import ACTION_EVENT_FIELDS
from badminton_data_process.core.io import read_csv_rows, write_csv_rows, write_json
from badminton_data_process.tracking.player.pose import pose_keypoints_from_json


DESCRIPTOR_VERSION = "bba_2d_descriptors_v1"
RALLY_SUMMARY_FIELDS = [
    "video_stem",
    "rally_id",
    "candidate_events",
    "stability_eligible_events",
    "footwork_eligible_events",
    "mean_candidate_score",
    "mean_support_width_ratio",
    "mean_body_support_offset_rms",
    "mean_event_path_distance_m",
    "mean_recovery_time_s",
    "reject_reasons_json",
    "descriptor_version",
]


def _float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _int(value: object) -> int | None:
    number = _float(value)
    return int(number) if number is not None else None


def _mean(values: Iterable[float | None]) -> float | None:
    usable = [value for value in values if value is not None]
    return statistics.fmean(usable) if usable else None


def _rounded(value: float | None) -> float | str:
    return round(value, 4) if value is not None and math.isfinite(value) else ""


def _rms(values: Iterable[float | None]) -> float | None:
    usable = [value for value in values if value is not None]
    return math.sqrt(statistics.fmean(value * value for value in usable)) if usable else None


def _path_distance(samples: list[tuple[int, float, float, float]]) -> float:
    return sum(
        math.hypot(current[2] - previous[2], current[3] - previous[3])
        for previous, current in zip(samples, samples[1:])
    )


def _position_nearest(
    samples: list[tuple[int, float, float, float]], frame: int
) -> tuple[int, float, float, float]:
    return min(samples, key=lambda sample: abs(sample[0] - frame))


def _cross_step_candidate(rows: list[Mapping[str, object]], threshold: float) -> bool | None:
    signs: list[int] = []
    for row in rows:
        points = {
            point.name: point
            for point in pose_keypoints_from_json(str(row.get("pose_keypoints_json") or ""))
            if point.confidence >= threshold
        }
        if "left_ankle" not in points or "right_ankle" not in points:
            continue
        difference = points["left_ankle"].x - points["right_ankle"].x
        if abs(difference) > 1e-6:
            signs.append(1 if difference > 0 else -1)
    if len(signs) < 3:
        return None
    return any(first != second for first, second in zip(signs, signs[1:]))


def _stability_descriptors(
    rows: list[Mapping[str, object]], min_rows: int
) -> dict[str, object]:
    eligible = [row for row in rows if row.get("kinematics_eligibility") == "eligible"]
    if len(eligible) < min_rows:
        return {
            "stability_eligibility": "not_eligible",
            "stability_reject_reason": "insufficient_kinematics_rows",
        }
    support = [_float(row.get("support_width_ratio")) for row in eligible]
    offsets = [_float(row.get("body_support_offset_ratio")) for row in eligible]
    trunk = [
        value
        for row in eligible
        if (value := _float(row.get("trunk_lean_deg"))) is not None
    ]
    knee_asymmetry = []
    for row in eligible:
        left = _float(row.get("left_knee_angle_deg"))
        right = _float(row.get("right_knee_angle_deg"))
        if left is not None and right is not None:
            knee_asymmetry.append(abs(left - right))
    usable_support = [value for value in support if value is not None]
    return {
        "stability_eligibility": "eligible",
        "stability_reject_reason": "",
        "mean_support_width_ratio": _rounded(_mean(support)),
        "body_support_offset_rms": _rounded(_rms(offsets)),
        "trunk_sway_std_deg": _rounded(statistics.pstdev(trunk) if len(trunk) >= 2 else None),
        "mean_knee_asymmetry_deg": _rounded(_mean(knee_asymmetry)),
        "wide_support_candidate": (
            int(max(usable_support) >= 0.35) if usable_support else ""
        ),
    }


def _candidate_kinematics_snapshot(
    rows: list[Mapping[str, object]], candidate_frame: int
) -> dict[str, object]:
    eligible = [
        row
        for row in rows
        if row.get("kinematics_eligibility") == "eligible"
        and _int(row.get("frame_id")) is not None
    ]
    if not eligible:
        return {}
    snapshot = min(
        eligible,
        key=lambda row: abs((_int(row.get("frame_id")) or 0) - candidate_frame),
    )
    return {
        f"candidate_{field}": _rounded(_float(snapshot.get(field)))
        for field in (
            "left_elbow_angle_deg",
            "right_elbow_angle_deg",
            "left_shoulder_angle_deg",
            "right_shoulder_angle_deg",
            "left_knee_angle_deg",
            "right_knee_angle_deg",
            "trunk_lean_deg",
        )
    }


def _footwork_descriptors(
    rows: list[Mapping[str, object]],
    candidate_frame: int,
    *,
    min_rows: int,
    keypoint_threshold: float,
    allow_player: bool,
) -> dict[str, object]:
    if not allow_player:
        return {
            "footwork_eligibility": "not_eligible",
            "footwork_reject_reason": "far_player_analysis_disabled",
        }
    samples = []
    for row in rows:
        frame = _int(row.get("frame_id"))
        timestamp = _float(row.get("timestamp"))
        x = _float(row.get("court_x"))
        y = _float(row.get("court_y"))
        if None not in (frame, timestamp, x, y):
            samples.append((int(frame), float(timestamp), float(x), float(y)))
    samples.sort()
    if len(samples) < min_rows:
        return {
            "footwork_eligibility": "not_eligible",
            "footwork_reject_reason": "insufficient_valid_court_positions",
        }
    before = [sample for sample in samples if sample[0] <= candidate_frame]
    after = [sample for sample in samples if sample[0] >= candidate_frame]
    if len(before) < 2 or len(after) < 2:
        return {
            "footwork_eligibility": "not_eligible",
            "footwork_reject_reason": "truncated_footwork_window",
        }
    contact = _position_nearest(samples, candidate_frame)
    pre_reference = before[0]
    speeds: list[tuple[int, float]] = []
    for previous, current in zip(samples, samples[1:]):
        delta_time = current[1] - previous[1]
        if delta_time <= 0 or current[0] - previous[0] > 2:
            continue
        speeds.append(
            (
                current[0],
                math.hypot(current[2] - previous[2], current[3] - previous[3])
                / delta_time,
            )
        )
    movement_start = next((frame for frame, speed in speeds if speed >= 0.5), None)
    post_speeds = [(frame, speed) for frame, speed in speeds if frame > candidate_frame]
    braking = next((frame for frame, speed in post_speeds if speed < 0.5), None)
    recovery = next(
        (
            sample
            for sample in after[1:]
            if math.hypot(sample[2] - pre_reference[2], sample[3] - pre_reference[3]) <= 0.35
        ),
        None,
    )
    return {
        "footwork_eligibility": "eligible",
        "footwork_reject_reason": "",
        "pre_contact_displacement_m": _rounded(_path_distance(before)),
        "post_contact_displacement_m": _rounded(_path_distance(after)),
        "event_path_distance_m": _rounded(_path_distance(samples)),
        "max_footwork_speed_m_s": _rounded(max((speed for _, speed in speeds), default=None)),
        "recovery_time_s": _rounded(recovery[1] - contact[1] if recovery else None),
        "movement_start_frame": movement_start if movement_start is not None else "",
        "braking_frame": braking if braking is not None else "",
        "recovery_frame": recovery[0] if recovery else "",
        "cross_step_candidate": (
            "" if (cross := _cross_step_candidate(rows, keypoint_threshold)) is None else int(cross)
        ),
    }


def enrich_action_events(
    event_rows: Iterable[Mapping[str, object]],
    player_rows: Iterable[Mapping[str, object]],
    kinematics_rows: Iterable[Mapping[str, object]],
    *,
    keypoint_threshold: float = 0.35,
    min_contiguous_frames: int = 5,
    enable_far_player: bool = True,
) -> list[dict[str, object]]:
    players: dict[tuple[str, str, str], list[Mapping[str, object]]] = defaultdict(list)
    kinematics: dict[tuple[str, str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in player_rows:
        players[(str(row.get("video_stem", "")), str(row.get("rally_id", "")), str(row.get("player_id", "")))].append(row)
    for row in kinematics_rows:
        kinematics[(str(row.get("video_stem", "")), str(row.get("rally_id", "")), str(row.get("player_id", "")))].append(row)
    output = []
    for source in event_rows:
        event = {field: source.get(field, "") for field in ACTION_EVENT_FIELDS}
        key = (str(event["video_stem"]), str(event["rally_id"]), str(event["player_id"]))
        start = _int(event.get("window_start_frame"))
        end = _int(event.get("window_end_frame"))
        candidate = _int(event.get("candidate_frame"))
        if None in (start, end, candidate):
            event.update(
                {
                    "stability_eligibility": "not_eligible",
                    "stability_reject_reason": "invalid_event_window",
                    "footwork_eligibility": "not_eligible",
                    "footwork_reject_reason": "invalid_event_window",
                }
            )
        else:
            assert start is not None and end is not None and candidate is not None
            player_window = [row for row in players.get(key, []) if (frame := _int(row.get("frame_id"))) is not None and start <= frame < end]
            kinematics_window = [row for row in kinematics.get(key, []) if (frame := _int(row.get("frame_id"))) is not None and start <= frame < end]
            event.update(_stability_descriptors(kinematics_window, min_contiguous_frames))
            event.update(_candidate_kinematics_snapshot(kinematics_window, candidate))
            event.update(
                _footwork_descriptors(
                    player_window,
                    candidate,
                    min_rows=min_contiguous_frames,
                    keypoint_threshold=keypoint_threshold,
                    allow_player=enable_far_player or key[2] != "far",
                )
            )
        event["descriptor_version"] = DESCRIPTOR_VERSION
        output.append(event)
    return output


def _summary_rows(events: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for event in events:
        groups[(str(event["video_stem"]), str(event["rally_id"]))].append(event)
    output = []
    for (video_stem, rally_id), rows in sorted(groups.items()):
        reasons = Counter(
            str(reason)
            for row in rows
            for reason in (
                row.get("stability_reject_reason", ""),
                row.get("footwork_reject_reason", ""),
            )
            if reason
        )
        output.append(
            {
                "video_stem": video_stem,
                "rally_id": rally_id,
                "candidate_events": len(rows),
                "stability_eligible_events": sum(row["stability_eligibility"] == "eligible" for row in rows),
                "footwork_eligible_events": sum(row["footwork_eligibility"] == "eligible" for row in rows),
                "mean_candidate_score": _rounded(_mean(_float(row["candidate_score"]) for row in rows)),
                "mean_support_width_ratio": _rounded(_mean(_float(row["mean_support_width_ratio"]) for row in rows)),
                "mean_body_support_offset_rms": _rounded(_mean(_float(row["body_support_offset_rms"]) for row in rows)),
                "mean_event_path_distance_m": _rounded(_mean(_float(row["event_path_distance_m"]) for row in rows)),
                "mean_recovery_time_s": _rounded(_mean(_float(row["recovery_time_s"]) for row in rows)),
                "reject_reasons_json": json.dumps(reasons, ensure_ascii=False, sort_keys=True),
                "descriptor_version": DESCRIPTOR_VERSION,
            }
        )
    return output


def analyze_event_descriptors(
    action_events_csv: Path,
    player_tracks_csv: Path,
    kinematics_frames_csv: Path,
    rally_summary_csv: Path,
    match_summary_json: Path,
    **kwargs: object,
) -> dict[str, object]:
    events = enrich_action_events(
        read_csv_rows(action_events_csv),
        read_csv_rows(player_tracks_csv),
        read_csv_rows(kinematics_frames_csv),
        **kwargs,
    )
    write_csv_rows(action_events_csv, ACTION_EVENT_FIELDS, events)
    rallies = _summary_rows(events)
    write_csv_rows(rally_summary_csv, RALLY_SUMMARY_FIELDS, rallies)
    match_summary = {
        "schema_version": DESCRIPTOR_VERSION,
        "candidate_events": len(events),
        "rallies_with_candidates": len(rallies),
        "stability_eligible_events": sum(event["stability_eligibility"] == "eligible" for event in events),
        "footwork_eligible_events": sum(event["footwork_eligibility"] == "eligible" for event in events),
        "capability_statement": "二维描述性运动学，不是医学诊断或三维生物力学测量。",
    }
    write_json(match_summary_json, match_summary)
    return match_summary

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping

from badminton_data_process.analysis.biomechanics.evaluation import GROUND_TRUTH_FIELDS
from badminton_data_process.core.io import ensure_dir, read_csv_rows, write_csv_rows
from badminton_data_process.tracking.player.pose import (
    pose_keypoints_from_json,
    skeleton_segments,
)


REVIEW_DRAFT_FIELDS = GROUND_TRUTH_FIELDS[:-1] + [
    "candidate_score",
    "predicted_stroke_class",
    "predicted_confidence",
    "notes",
]


def _integer(value: object) -> int | None:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def _safe_name(value: object) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return name[:120] or "event"


def _select_events(
    rows: Iterable[Mapping[str, object]], max_events: int
) -> list[dict[str, object]]:
    eligible = [
        dict(row)
        for row in rows
        if str(row.get("event_eligibility", "eligible")) == "eligible"
        and _integer(row.get("candidate_frame")) is not None
    ]
    eligible.sort(
        key=lambda row: (
            str(row.get("video_stem", "")),
            str(row.get("rally_id", "")),
            _integer(row.get("candidate_frame")) or 0,
        )
    )
    if max_events <= 0 or len(eligible) <= max_events:
        return eligible

    # First balance classifier outcomes and near/far roles. Within every stratum,
    # round-robin across rallies so a long or failure-heavy rally cannot dominate
    # the frozen review cohort.
    strata: dict[tuple[str, str], dict[tuple[str, str], list[dict[str, object]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in eligible:
        classification_status = str(row.get("classification_eligibility", "")).strip()
        outcome = (
            "eligible"
            if classification_status == "eligible"
            else str(row.get("classification_reject_reason") or "not_classified")
        )
        stratum = (str(row.get("player_id") or "unknown"), outcome)
        rally = (str(row.get("video_stem", "")), str(row.get("rally_id", "")))
        strata[stratum][rally].append(row)

    stratum_queues: list[list[dict[str, object]]] = []
    for stratum in sorted(strata):
        rally_groups = list(strata[stratum].values())
        queue: list[dict[str, object]] = []
        offset = 0
        while True:
            added = False
            for group in rally_groups:
                if offset < len(group):
                    queue.append(group[offset])
                    added = True
            if not added:
                break
            offset += 1
        stratum_queues.append(queue)

    selected: list[dict[str, object]] = []
    offset = 0
    while len(selected) < max_events:
        added = False
        for queue in stratum_queues:
            if offset < len(queue):
                selected.append(queue[offset])
                added = True
                if len(selected) == max_events:
                    break
        if not added:
            break
        offset += 1
    return selected


def _classification_outcome(row: Mapping[str, object]) -> str:
    if str(row.get("classification_eligibility", "")) == "eligible":
        return "eligible"
    return str(row.get("classification_reject_reason") or "not_classified")


def _review_diagnostics(
    rows: Iterable[Mapping[str, object]],
    selected: Iterable[Mapping[str, object]],
) -> dict[str, object]:
    candidates = [
        dict(row)
        for row in rows
        if str(row.get("event_eligibility", "eligible")) == "eligible"
        and _integer(row.get("candidate_frame")) is not None
    ]
    selected_rows = [dict(row) for row in selected]
    eligible_count = sum(
        str(row.get("classification_eligibility", "")) == "eligible"
        for row in candidates
    )
    outcomes = Counter(_classification_outcome(row) for row in candidates)
    selected_outcomes = Counter(_classification_outcome(row) for row in selected_rows)
    by_player: dict[str, dict[str, object]] = {}
    for player_id in sorted({str(row.get("player_id") or "unknown") for row in candidates}):
        player_rows = [row for row in candidates if str(row.get("player_id") or "unknown") == player_id]
        player_eligible = sum(
            str(row.get("classification_eligibility", "")) == "eligible"
            for row in player_rows
        )
        by_player[player_id] = {
            "candidate_events": len(player_rows),
            "classification_eligible_events": player_eligible,
            "classification_eligible_ratio": (
                round(player_eligible / len(player_rows), 4) if player_rows else None
            ),
        }
    return {
        "candidate_events": len(candidates),
        "classification_eligible_events": eligible_count,
        "classification_eligible_ratio": (
            round(eligible_count / len(candidates), 4) if candidates else None
        ),
        "classification_outcomes": dict(sorted(outcomes.items())),
        "selected_classification_outcomes": dict(sorted(selected_outcomes.items())),
        "players": by_player,
    }


def _draw_tracks(
    frame: object,
    rows: Iterable[Mapping[str, object]],
    *,
    target_player: str,
    keypoint_threshold: float,
) -> object:
    import cv2

    colors = {"near": (64, 96, 255), "far": (255, 170, 32)}
    for row in rows:
        role = str(row.get("player_id", ""))
        color = colors.get(role, (220, 220, 220))
        thickness = 4 if role == target_player else 2
        points = pose_keypoints_from_json(str(row.get("pose_keypoints_json") or ""))
        for start, end in skeleton_segments(points, keypoint_threshold):
            cv2.line(
                frame,
                (int(round(start.x)), int(round(start.y))),
                (int(round(end.x)), int(round(end.y))),
                color,
                thickness,
                cv2.LINE_AA,
            )
        for point in points:
            if point.confidence >= keypoint_threshold:
                cv2.circle(
                    frame,
                    (int(round(point.x)), int(round(point.y))),
                    3 if role == target_player else 2,
                    color,
                    -1,
                    cv2.LINE_AA,
                )
        coordinates = [_integer(row.get(field)) for field in (
            "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"
        )]
        if all(value is not None for value in coordinates):
            x1, y1, x2, y2 = (int(value) for value in coordinates)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(
                frame,
                role,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
                cv2.LINE_AA,
            )
    return frame


def _read_frame(capture: object, frame_id: int) -> object | None:
    import cv2

    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_id))
    ok, frame = capture.read()
    return frame if ok else None


def _write_image(path: Path, image: object) -> bool:
    import cv2

    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        return False
    encoded.tofile(str(path))
    return True


def _review_montage(
    event: Mapping[str, object],
    track_index: Mapping[tuple[str, str, int], list[dict[str, str]]],
    *,
    frame_radius: int,
    panel_width: int,
    keypoint_threshold: float,
) -> object | None:
    import cv2

    video_path = Path(str(event.get("video_path", "")))
    candidate = _integer(event.get("candidate_frame"))
    if candidate is None or not video_path.is_file():
        return None
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        return None
    frame_ids = [max(0, candidate - frame_radius), candidate, candidate + frame_radius]
    panels = []
    for frame_id in frame_ids:
        frame = _read_frame(capture, frame_id)
        if frame is None:
            continue
        group = (
            str(event.get("video_stem", "")),
            str(event.get("rally_id", "")),
            frame_id,
        )
        _draw_tracks(
            frame,
            track_index.get(group, []),
            target_player=str(event.get("player_id", "")),
            keypoint_threshold=keypoint_threshold,
        )
        scale = min(1.0, panel_width / max(1, frame.shape[1]))
        if scale < 1.0:
            frame = cv2.resize(
                frame,
                (int(round(frame.shape[1] * scale)), int(round(frame.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        title = f"frame {frame_id}" + ("  CANDIDATE" if frame_id == candidate else "")
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 38), (12, 16, 24), -1)
        cv2.putText(
            frame,
            title,
            (12, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        panels.append(frame)
    capture.release()
    if len(panels) != 3:
        return None
    target_height = min(panel.shape[0] for panel in panels)
    panels = [
        cv2.resize(
            panel,
            (int(round(panel.shape[1] * target_height / panel.shape[0])), target_height),
            interpolation=cv2.INTER_AREA,
        )
        for panel in panels
    ]
    return cv2.hconcat(panels)


def export_action_event_review(
    action_events_csv: Path,
    output_dir: Path,
    *,
    player_tracks_csv: Path | None = None,
    max_events: int = 0,
    frame_radius: int = 2,
    panel_width: int = 640,
    keypoint_threshold: float = 0.35,
) -> dict[str, object]:
    output_dir = ensure_dir(output_dir)
    image_dir = ensure_dir(output_dir / "images")
    all_events = read_csv_rows(action_events_csv)
    events = _select_events(all_events, max_events)
    tracks = read_csv_rows(player_tracks_csv) if player_tracks_csv else []
    track_index: dict[tuple[str, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in tracks:
        frame_id = _integer(row.get("frame_id"))
        if frame_id is not None:
            track_index[(row.get("video_stem", ""), row.get("rally_id", ""), frame_id)].append(row)

    draft_rows: list[dict[str, object]] = []
    rendered = 0
    for index, event in enumerate(events, start=1):
        image_name = f"{index:04d}_{_safe_name(event.get('event_id'))}.jpg"
        montage = _review_montage(
            event,
            track_index,
            frame_radius=frame_radius,
            panel_width=panel_width,
            keypoint_threshold=keypoint_threshold,
        )
        review_image = ""
        if montage is not None and _write_image(image_dir / image_name, montage):
            rendered += 1
            review_image = f"images/{image_name}"
        draft_rows.append(
            {
                "video_stem": event.get("video_stem", ""),
                "rally_id": event.get("rally_id", ""),
                "event_id": event.get("event_id", ""),
                "reference_frame": event.get("candidate_frame", ""),
                "player_id": event.get("player_id", ""),
                "stroke_class": "",
                "annotation_scope": "prediction_seeded",
                "review_status": "pending",
                "review_image": review_image,
                "candidate_score": event.get("candidate_score", ""),
                "predicted_stroke_class": event.get("stroke_class", ""),
                "predicted_confidence": event.get("classification_confidence", ""),
                "notes": "",
            }
        )
    draft_csv = output_dir / "biomechanics_ground_truth_draft.csv"
    write_csv_rows(draft_csv, REVIEW_DRAFT_FIELDS, draft_rows)
    result = {
        "schema_version": "bba_biomechanics_review_v2",
        "status": "empty" if not events else ("success" if rendered == len(events) else "partial"),
        "selected_events": len(events),
        "rendered_montages": rendered,
        "missing_montages": len(events) - rendered,
        "annotation_scope": "prediction_seeded",
        "draft_csv": str(draft_csv),
        "diagnostics": _review_diagnostics(all_events, events),
    }
    (output_dir / "review_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export BBA action-event contact montages and a human-review CSV draft."
    )
    parser.add_argument("action_events", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--player-tracks", type=Path)
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument("--frame-radius", type=int, default=2)
    parser.add_argument("--panel-width", type=int, default=640)
    parser.add_argument("--keypoint-threshold", type=float, default=0.35)
    args = parser.parse_args(argv)
    if args.max_events < 0:
        parser.error("--max-events must be non-negative")
    if args.frame_radius < 1:
        parser.error("--frame-radius must be at least 1")
    if args.panel_width < 160:
        parser.error("--panel-width must be at least 160")
    result = export_action_event_review(
        args.action_events,
        args.output_dir,
        player_tracks_csv=args.player_tracks,
        max_events=args.max_events,
        frame_radius=args.frame_radius,
        panel_width=args.panel_width,
        keypoint_threshold=args.keypoint_threshold,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
from pathlib import Path

from badminton_data_process.core.config import load_config
from badminton_data_process.core.config_schema import parse_config
from badminton_data_process.core.io import ensure_dir, write_csv_rows, write_json
from badminton_data_process.core.paths import RunLayout, discover_project_root
from badminton_data_process.core.run import make_run_id
from badminton_data_process.core.schemas import (
    MAIN_VIEW_FRAME_FIELDS,
    MAIN_VIEW_QUALITY_FIELDS,
    MAIN_VIEW_SEGMENT_FIELDS,
    MainViewLabel,
    REJECTED_SEGMENT_FIELDS,
)
from badminton_data_process.main_view.scoring import FrameScore, require_opencv, score_frame

try:
    import cv2
except ImportError:  # pragma: no cover - runtime dependency
    cv2 = None


def contiguous_segments(
    scores: list[FrameScore],
    fps: float,
    sample_every: int,
    min_segment_seconds: float,
    max_gap_seconds: float,
) -> list[dict[str, object]]:
    segments: list[list[FrameScore]] = []
    current: list[FrameScore] = []
    max_gap_frames = int(max_gap_seconds * fps)
    for score in scores:
        if not score.is_main_view:
            continue
        if not current:
            current = [score]
            continue
        if score.sample_frame - current[-1].sample_frame <= max_gap_frames + sample_every:
            current.append(score)
        else:
            segments.append(current)
            current = [score]
    if current:
        segments.append(current)

    rows: list[dict[str, object]] = []
    min_frames = int(min_segment_seconds * fps)
    for index, items in enumerate(segments, start=1):
        start = max(0, items[0].sample_frame)
        end = items[-1].sample_frame + sample_every
        if end - start < min_frames:
            continue
        row = {
            "segment_id": f"{index:03d}",
            "start_frame": start,
            "end_frame": end,
            "start_time": round(start / fps, 3),
            "end_time": round(end / fps, 3),
            "duration_seconds": round((end - start) / fps, 3),
            "label": MainViewLabel.MAIN_VIEW.value,
            "main_view_score": round(sum(s.main_view_score for s in items) / len(items), 4),
            "court_score": round(sum(s.court_score for s in items) / len(items), 4),
            "geometry_score": round(sum(s.geometry_score for s in items) / len(items), 4),
            "layout_score": round(sum(s.layout_score for s in items) / len(items), 4),
            "stability_score": round(sum(s.stability_score for s in items) / len(items), 4),
            "reject_score": round(sum(s.reject_score for s in items) / len(items), 4),
            "frame_count": len(items),
        }
        rows.append(row)
    for index, row in enumerate(rows, start=1):
        row["segment_id"] = f"{index:03d}"
    return rows


def rejected_segments(scores: list[FrameScore], fps: float, sample_every: int, min_score: float = 0.45) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    current: list[FrameScore] = []
    for score in scores:
        if not score.is_main_view and score.main_view_score >= min_score:
            current.append(score)
        elif current:
            rows.append(_rejected_row(current, fps, sample_every, len(rows) + 1))
            current = []
    if current:
        rows.append(_rejected_row(current, fps, sample_every, len(rows) + 1))
    return rows


def _rejected_row(items: list[FrameScore], fps: float, sample_every: int, index: int) -> dict[str, object]:
    reasons = [item.reject_reason for item in items if item.reject_reason]
    reason = max(set(reasons), key=reasons.count) if reasons else "low_main_view_score"
    start = items[0].sample_frame
    end = items[-1].sample_frame + sample_every
    return {
        "segment_id": f"R{index:03d}",
        "rally_id": "",
        "start_frame": start,
        "end_frame": end,
        "start_time": round(start / fps, 3),
        "end_time": round(end / fps, 3),
        "duration_seconds": round((end - start) / fps, 3),
        "reject_reason": reason,
        "score": round(sum(item.main_view_score for item in items) / len(items), 4),
    }


def timeline_payload(video_path: Path, fps: float, frame_count: int, segments: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "segment_id": row["segment_id"],
            "start_frame": row["start_frame"],
            "end_frame": row["end_frame"],
            "start_time": row["start_time"],
            "end_time": row["end_time"],
            "duration_seconds": row["duration_seconds"],
            "label": row["label"],
            "main_view_score": row["main_view_score"],
            "source_video": str(video_path),
            "fps": fps,
            "source_frame_count": frame_count,
        }
        for row in segments
    ]


def analyze_main_view(
    input_video: Path,
    output_dir: Path,
    sample_every: int = 30,
    threshold: float = 0.75,
    max_reject_score: float = 0.4,
    min_segment_seconds: float = 3.0,
    max_gap_seconds: float = 2.0,
) -> int:
    require_opencv()
    capture = cv2.VideoCapture(str(input_video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {input_video}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    reported_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    output_dir = ensure_dir(output_dir)
    keyframe_dir = ensure_dir(output_dir / "keyframes")
    ensure_dir(output_dir / "rejected")

    scores: list[FrameScore] = []
    previous_corners = None
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % sample_every != 0:
            frame_index += 1
            continue
        score, corners = score_frame(
            frame,
            frame_index,
            fps,
            previous_corners,
            threshold=threshold,
            max_reject_score=max_reject_score,
        )
        if corners is not None and score.geometry_score >= 0.45:
            previous_corners = corners
        scores.append(score)
        if score.is_main_view and len(scores) % 10 == 0:
            cv2.imwrite(str(keyframe_dir / f"frame_{frame_index:08d}.jpg"), frame)
        frame_index += 1
    capture.release()
    frame_count = frame_index if frame_index > 0 else reported_frame_count

    frame_rows = [score.to_row() for score in scores]
    segment_rows = contiguous_segments(scores, fps, sample_every, min_segment_seconds, max_gap_seconds)
    for row in segment_rows:
        row["end_frame"] = min(int(row["end_frame"]), frame_count)
        row["end_time"] = round(int(row["end_frame"]) / fps, 3)
        row["duration_seconds"] = round(
            (int(row["end_frame"]) - int(row["start_frame"])) / fps,
            3,
        )
    min_segment_frames = int(min_segment_seconds * fps)
    segment_rows = [
        row
        for row in segment_rows
        if int(row["end_frame"]) - int(row["start_frame"]) >= min_segment_frames
    ]
    for index, row in enumerate(segment_rows, start=1):
        row["segment_id"] = f"{index:03d}"
    reject_rows = rejected_segments(scores, fps, sample_every)
    quality_rows = [
        {
            **{field: row.get(field, "") for field in MAIN_VIEW_QUALITY_FIELDS},
            "segment_id": row["segment_id"],
            "rally_id": "",
            "start_frame": row["start_frame"],
            "end_frame": row["end_frame"],
            "start_time": row["start_time"],
            "end_time": row["end_time"],
            "duration_seconds": row["duration_seconds"],
            "quality_score": row["main_view_score"],
            "main_view_score": row["main_view_score"],
            "court_score": row["court_score"],
            "geometry_score": row["geometry_score"],
            "layout_score": row["layout_score"],
            "projection_outlier_ratio": "",
            "boundary_stuck_ratio": "",
            "absurd_y_ratio": "",
            "accepted": 1,
            "reject_reason": "",
        }
        for row in segment_rows
    ]

    write_csv_rows(output_dir / "main_view_frame_scores.csv", MAIN_VIEW_FRAME_FIELDS, frame_rows)
    write_csv_rows(output_dir / "main_view_segments.csv", MAIN_VIEW_SEGMENT_FIELDS, segment_rows)
    write_csv_rows(output_dir / "main_view_quality.csv", MAIN_VIEW_QUALITY_FIELDS, quality_rows)
    write_csv_rows(output_dir / "rejected_segments.csv", REJECTED_SEGMENT_FIELDS, reject_rows)
    write_json(output_dir / "main_view_timeline.json", timeline_payload(input_video, fps, frame_count, segment_rows))
    print(f"Main-view frame scores: {len(frame_rows)}")
    print(f"Accepted main-view segments: {len(segment_rows)}")
    print(f"Rejected candidate segments: {len(reject_rows)}")
    print(f"Main-view output dir: {output_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze strict birdseye main-view timeline.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--runs-dir", type=Path, default=None)
    parser.add_argument("--sample-every", type=int, default=30)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--max-reject-score", type=float, default=0.4)
    parser.add_argument("--min-segment-seconds", type=float, default=3.0)
    parser.add_argument("--max-gap-seconds", type=float, default=2.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = discover_project_root()
    run_id = args.run_id or make_run_id("main_view")
    cfg = parse_config(load_config(args.config, root=root))
    layout = RunLayout.create(
        root,
        run_id,
        args.runs_dir if args.runs_dir is not None else cfg.data.runs_dir,
    )
    output_dir = args.output_dir or layout.main_view_dir
    return analyze_main_view(
        input_video=args.input,
        output_dir=output_dir,
        sample_every=args.sample_every,
        threshold=args.threshold,
        max_reject_score=args.max_reject_score,
        min_segment_seconds=args.min_segment_seconds,
        max_gap_seconds=args.max_gap_seconds,
    )

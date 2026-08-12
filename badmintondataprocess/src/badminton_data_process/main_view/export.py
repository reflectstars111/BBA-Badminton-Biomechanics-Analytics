from __future__ import annotations

import argparse
from pathlib import Path

from badminton_data_process.core.io import ensure_parent, read_json, write_csv_rows
from badminton_data_process.core.schemas import FRAME_INDEX_FIELDS

try:
    import cv2
except ImportError:  # pragma: no cover - runtime dependency
    cv2 = None


def require_opencv() -> None:
    if cv2 is None:
        raise RuntimeError("OpenCV is required for main-view export.")


def export_main_view(
    input_video: Path,
    timeline_path: Path,
    output_video: Path,
    frame_index_csv: Path,
) -> int:
    require_opencv()
    timeline = read_json(timeline_path)
    segments = [
        (str(item["segment_id"]), int(item["start_frame"]), int(item["end_frame"]))
        for item in timeline
    ]
    if not segments:
        raise RuntimeError(f"Timeline has no segments: {timeline_path}")

    capture = cv2.VideoCapture(str(input_video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {input_video}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video dimensions: {input_video}")

    ensure_parent(output_video)
    ensure_parent(frame_index_csv)
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    frame_index_rows: list[dict[str, object]] = []
    clean_frame_id = 0
    for segment_id, start, end in segments:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start)
        frame_id = start
        while frame_id < end:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(frame)
            frame_index_rows.append(
                {
                    "clean_frame_id": clean_frame_id,
                    "original_time": round(frame_id / fps, 3),
                    "original_frame_id": frame_id,
                    "segment_id": segment_id,
                }
            )
            clean_frame_id += 1
            frame_id += 1
    writer.release()
    capture.release()
    write_csv_rows(frame_index_csv, FRAME_INDEX_FIELDS, frame_index_rows)
    print(f"Clean main-view video: {output_video}")
    print(f"Frame index CSV: {frame_index_csv}")
    print(f"Clean frames: {clean_frame_id}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export clean main-view video from timeline.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--timeline", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--frame-index", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    timeline_dir = args.timeline.parent
    output_video = args.out or timeline_dir / "clean_main_view.mp4"
    frame_index_csv = args.frame_index or timeline_dir / "frame_index.csv"
    return export_main_view(args.input, args.timeline, output_video, frame_index_csv)


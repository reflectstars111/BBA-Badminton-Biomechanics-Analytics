from __future__ import annotations

import argparse
import math
from pathlib import Path

from badminton_data_process.core.io import ensure_dir, read_json, write_csv_rows
from badminton_data_process.core.schemas import RALLY_FIELDS
from badminton_data_process.legacy import load_legacy_module, run_legacy_main


def _module():
    return load_legacy_module("rally_segmentation.py")


def frame_metrics(*args, **kwargs):
    return _module().frame_metrics(*args, **kwargs)


def analyze_video(*args, **kwargs):
    return _module().analyze_video(*args, **kwargs)


def merge_candidate_segments(*args, **kwargs):
    return _module().merge_candidate_segments(*args, **kwargs)


def select_live_rallies(*args, **kwargs):
    return _module().select_live_rallies(*args, **kwargs)


def segment_rallies(*args, **kwargs):
    return _module().segment_rallies(*args, **kwargs)


def _validate_args(args: argparse.Namespace) -> None:
    if args.sample_every <= 0:
        raise SystemExit("--sample-every must be > 0")
    if args.min_rally_seconds <= 0:
        raise SystemExit("--min-rally-seconds must be > 0")
    if args.max_rally_seconds <= args.min_rally_seconds:
        raise SystemExit("--max-rally-seconds must be > --min-rally-seconds")
    if args.max_gap_seconds < 0:
        raise SystemExit("--max-gap-seconds must be >= 0")
    if not math.isfinite(args.min_motion_score) or args.min_motion_score < 0:
        raise SystemExit("--min-motion-score must be a finite value >= 0")
    if not math.isfinite(args.max_motion_score) or args.max_motion_score <= args.min_motion_score:
        raise SystemExit("--max-motion-score must be > --min-motion-score")


def _timeline_intervals(timeline_path: Path) -> list[tuple[int, int, str]]:
    payload = read_json(timeline_path)
    intervals: list[tuple[int, int, str]] = []
    for index, item in enumerate(payload, start=1):
        start = int(item["start_frame"])
        end = int(item["end_frame"])
        if end <= start:
            continue
        intervals.append((start, end, str(item.get("segment_id", f"{index:03d}"))))
    return intervals


def segment_rallies_with_timeline(
    input_path: Path,
    timeline_path: Path,
    output_dir: Path,
    metadata_csv: Path,
    sample_every: int,
    min_rally_seconds: float,
    max_rally_seconds: float,
    max_gap_seconds: float,
    min_motion_score: float,
    max_motion_score: float,
    min_center_green_ratio: float,
    min_bottom_green_ratio: float,
    min_line_ratio: float,
    min_top_green_ratio: float,
    min_middle_green_ratio: float,
    max_left_right_green_diff: float,
    min_top_dark_ratio: float,
    min_middle_edge_ratio: float,
    pad_before_seconds: float,
    pad_after_seconds: float,
    max_pre_context_seconds: float,
    max_post_context_seconds: float,
    allowed_context_drop_samples: int,
    overwrite: bool,
) -> int:
    module = _module()
    intervals = _timeline_intervals(timeline_path)
    if not intervals:
        raise RuntimeError(f"Timeline contains no valid intervals: {timeline_path}")

    analysis_rows, fps, _, _ = module.analyze_video(
        input_path,
        sample_every,
        min_motion_score,
        max_motion_score,
        min_center_green_ratio,
        min_bottom_green_ratio,
        min_line_ratio,
        min_top_green_ratio,
        min_middle_green_ratio,
        max_left_right_green_diff,
        min_top_dark_ratio,
        min_middle_edge_ratio,
    )

    all_segments: list[tuple[int, int]] = []
    timeline_filtered_rows: list[dict[str, float]] = []
    for start_frame, end_frame, _segment_id in intervals:
        rows = [
            row
            for row in analysis_rows
            if start_frame <= int(row["sample_frame"]) <= end_frame
        ]
        timeline_filtered_rows.extend(rows)
        segments = module.merge_candidate_segments(
            analysis_rows=rows,
            sample_every=sample_every,
            min_rally_seconds=min_rally_seconds,
            max_rally_seconds=max_rally_seconds,
            max_gap_seconds=max_gap_seconds,
            pad_before_seconds=pad_before_seconds,
            pad_after_seconds=pad_after_seconds,
            fps=fps,
            max_pre_context_seconds=max_pre_context_seconds,
            max_post_context_seconds=max_post_context_seconds,
            allowed_context_drop_samples=allowed_context_drop_samples,
        )
        for segment_start, segment_end in segments:
            all_segments.append((max(start_frame, segment_start), min(end_frame, segment_end)))

    all_segments = sorted(set(all_segments))
    match_id = module.extract_match_id(input_path)
    output_dir = ensure_dir(output_dir)
    if overwrite:
        removed = module.remove_previous_outputs(output_dir, match_id)
        if removed:
            print(f"Removed previous clips: {removed}")
    rally_rows = module.write_rally_clips(input_path, output_dir, all_segments, fps, match_id)
    for row in rally_rows:
        row["notes"] = "timeline-constrained main-view rally candidate"
    module.write_analysis_csv(metadata_csv, timeline_filtered_rows)
    write_csv_rows(metadata_csv, RALLY_FIELDS, rally_rows)

    print(f"Video: {input_path}")
    print(f"Timeline: {timeline_path}")
    print(f"FPS: {fps:.3f}")
    print(f"Timeline intervals: {len(intervals)}")
    print(f"Sampled frames in timeline: {len(timeline_filtered_rows)}")
    print(f"Candidate rallies: {len(rally_rows)}")
    print(f"Rally metadata saved to: {metadata_csv}")
    return 0


def build_timeline_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Segment rally clips, optionally constrained by main-view timeline.")
    parser.add_argument("input", type=Path, help="Path to the full match video.")
    parser.add_argument("--timeline", type=Path, default=None, help="Main-view timeline JSON.")
    parser.add_argument("--output-dir", type=Path, default=Path("rallies"))
    parser.add_argument("--metadata-csv", type=Path, default=Path("metadata/rallies.csv"))
    parser.add_argument("--sample-every", type=int, default=15)
    parser.add_argument("--min-rally-seconds", type=float, default=4.0)
    parser.add_argument("--max-rally-seconds", type=float, default=45.0)
    parser.add_argument("--max-gap-seconds", type=float, default=3.0)
    parser.add_argument("--min-motion-score", type=float, default=0.01)
    parser.add_argument("--max-motion-score", type=float, default=0.16)
    parser.add_argument("--min-center-green-ratio", type=float, default=0.22)
    parser.add_argument("--min-bottom-green-ratio", type=float, default=0.36)
    parser.add_argument("--min-line-ratio", type=float, default=0.09)
    parser.add_argument("--min-top-green-ratio", type=float, default=0.05)
    parser.add_argument("--min-middle-green-ratio", type=float, default=0.20)
    parser.add_argument("--max-left-right-green-diff", type=float, default=0.16)
    parser.add_argument("--min-top-dark-ratio", type=float, default=0.15)
    parser.add_argument("--min-middle-edge-ratio", type=float, default=0.16)
    parser.add_argument("--pad-before-seconds", type=float, default=0.4)
    parser.add_argument("--pad-after-seconds", type=float, default=0.6)
    parser.add_argument("--max-pre-context-seconds", type=float, default=2.2)
    parser.add_argument("--max-post-context-seconds", type=float, default=1.4)
    parser.add_argument("--allowed-context-drop-samples", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    if argv and "--timeline" in argv:
        args = build_timeline_parser().parse_args(argv)
        _validate_args(args)
        return segment_rallies_with_timeline(
            input_path=args.input,
            timeline_path=args.timeline,
            output_dir=args.output_dir,
            metadata_csv=args.metadata_csv,
            sample_every=args.sample_every,
            min_rally_seconds=args.min_rally_seconds,
            max_rally_seconds=args.max_rally_seconds,
            max_gap_seconds=args.max_gap_seconds,
            min_motion_score=args.min_motion_score,
            max_motion_score=args.max_motion_score,
            min_center_green_ratio=args.min_center_green_ratio,
            min_bottom_green_ratio=args.min_bottom_green_ratio,
            min_line_ratio=args.min_line_ratio,
            min_top_green_ratio=args.min_top_green_ratio,
            min_middle_green_ratio=args.min_middle_green_ratio,
            max_left_right_green_diff=args.max_left_right_green_diff,
            min_top_dark_ratio=args.min_top_dark_ratio,
            min_middle_edge_ratio=args.min_middle_edge_ratio,
            pad_before_seconds=args.pad_before_seconds,
            pad_after_seconds=args.pad_after_seconds,
            max_pre_context_seconds=args.max_pre_context_seconds,
            max_post_context_seconds=args.max_post_context_seconds,
            allowed_context_drop_samples=args.allowed_context_drop_samples,
            overwrite=args.overwrite,
        )
    return run_legacy_main("rally_segmentation.py", argv)

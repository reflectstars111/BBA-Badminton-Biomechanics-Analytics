from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from common import ensure_dir, write_csv_rows

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover - runtime dependency
    cv2 = None
    np = None


def require_opencv() -> None:
    if cv2 is None or np is None:
        raise RuntimeError(
            'OpenCV and NumPy are required. Install them with: pip install opencv-python numpy'
        )


def extract_match_id(input_path: Path) -> str:
    return input_path.stem.replace(' ', '_')


def frame_metrics(frame: np.ndarray) -> dict[str, float]:
    resized = cv2.resize(frame, (320, 180))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)

    edges = cv2.Canny(gray, 50, 150)
    line_ratio = float(np.count_nonzero(edges)) / float(edges.size)

    green_mask = (
        (hue >= 35)
        & (hue <= 95)
        & (saturation >= 40)
        & (value >= 35)
    )
    center_green_ratio = float(np.mean(green_mask[40:175, 25:295]))
    bottom_green_ratio = float(np.mean(green_mask[70:179, 40:280]))
    top_green_ratio = float(np.mean(green_mask[20:90, 40:280]))
    middle_green_ratio = float(np.mean(green_mask[60:130, 40:280]))
    left_green_ratio = float(np.mean(green_mask[:, :160]))
    right_green_ratio = float(np.mean(green_mask[:, 160:]))
    top_dark_ratio = float(np.mean(gray[:50, :] < 40))
    middle_edge_ratio = float(np.mean(edges[60:120, :] > 0))
    green_ratio = float(np.mean(green_mask))

    return {
        'line_ratio': line_ratio,
        'green_ratio': green_ratio,
        'center_green_ratio': center_green_ratio,
        'bottom_green_ratio': bottom_green_ratio,
        'top_green_ratio': top_green_ratio,
        'middle_green_ratio': middle_green_ratio,
        'left_green_ratio': left_green_ratio,
        'right_green_ratio': right_green_ratio,
        'top_dark_ratio': top_dark_ratio,
        'middle_edge_ratio': middle_edge_ratio,
    }


def analyze_video(
    input_path: Path,
    sample_every: int,
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
) -> tuple[list[dict[str, float]], float, int, int]:
    require_opencv()
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f'Cannot open video: {input_path}')

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    analysis_rows: list[dict[str, float]] = []
    previous_gray: np.ndarray | None = None
    frame_index = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        if frame_index % sample_every != 0:
            frame_index += 1
            continue

        resized = cv2.resize(frame, (320, 180))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        if previous_gray is None:
            motion_score = 0.0
        else:
            frame_diff = cv2.absdiff(gray, previous_gray)
            motion_score = float(np.mean(frame_diff)) / 255.0

        metrics = frame_metrics(frame)
        is_court_view = (
            metrics['center_green_ratio'] >= min_center_green_ratio
            and metrics['bottom_green_ratio'] >= min_bottom_green_ratio
            and metrics['line_ratio'] >= min_line_ratio
            and metrics['top_green_ratio'] >= min_top_green_ratio
            and metrics['middle_green_ratio'] >= min_middle_green_ratio
            and abs(metrics['left_green_ratio'] - metrics['right_green_ratio'])
            <= max_left_right_green_diff
            and metrics['top_dark_ratio'] >= min_top_dark_ratio
            and metrics['middle_edge_ratio'] >= min_middle_edge_ratio
        )
        is_candidate = (
            is_court_view
            and motion_score >= min_motion_score
            and motion_score <= max_motion_score
        )
        analysis_rows.append(
            {
                'sample_frame': float(frame_index),
                'timestamp': frame_index / fps,
                'motion_score': motion_score,
                'line_ratio': metrics['line_ratio'],
                'green_ratio': metrics['green_ratio'],
                'center_green_ratio': metrics['center_green_ratio'],
                'bottom_green_ratio': metrics['bottom_green_ratio'],
                'top_green_ratio': metrics['top_green_ratio'],
                'middle_green_ratio': metrics['middle_green_ratio'],
                'left_green_ratio': metrics['left_green_ratio'],
                'right_green_ratio': metrics['right_green_ratio'],
                'top_dark_ratio': metrics['top_dark_ratio'],
                'middle_edge_ratio': metrics['middle_edge_ratio'],
                'is_court_view': float(is_court_view),
                'is_candidate': float(is_candidate),
            }
        )

        previous_gray = gray
        frame_index += 1

    capture.release()
    if width <= 0 or height <= 0:
        raise RuntimeError(f'Invalid video shape for: {input_path}')

    return analysis_rows, fps, frame_count, width * height


def merge_candidate_segments(
    analysis_rows: list[dict[str, float]],
    sample_every: int,
    min_rally_seconds: float,
    max_rally_seconds: float,
    max_gap_seconds: float,
    pad_before_seconds: float,
    pad_after_seconds: float,
    fps: float,
    max_pre_context_seconds: float,
    max_post_context_seconds: float,
    allowed_context_drop_samples: int,
) -> list[tuple[int, int]]:
    sample_segments: list[tuple[int, int]] = []
    start_frame: int | None = None
    end_frame: int | None = None

    for row in analysis_rows:
        frame_id = int(row['sample_frame'])
        is_candidate = bool(row['is_candidate'])
        if is_candidate:
            if start_frame is None:
                start_frame = frame_id
            end_frame = frame_id
        elif start_frame is not None and end_frame is not None:
            sample_segments.append((start_frame, end_frame))
            start_frame = None
            end_frame = None

    if start_frame is not None and end_frame is not None:
        sample_segments.append((start_frame, end_frame))

    merged_segments: list[tuple[int, int]] = []
    max_gap_frames = int(max_gap_seconds * fps)
    min_rally_frames = int(min_rally_seconds * fps)
    max_rally_frames = int(max_rally_seconds * fps)
    pad_before_frames = int(pad_before_seconds * fps)
    pad_after_frames = int(pad_after_seconds * fps)

    for start, end in sample_segments:
        actual_start = start
        actual_end = end + sample_every
        if not merged_segments:
            merged_segments.append((actual_start, actual_end))
            continue

        previous_start, previous_end = merged_segments[-1]
        if actual_start - previous_end <= max_gap_frames:
            merged_segments[-1] = (previous_start, actual_end)
        else:
            merged_segments.append((actual_start, actual_end))

    if not merged_segments:
        return []

    sample_frames = [int(row['sample_frame']) for row in analysis_rows]
    court_flags = [bool(row['is_court_view']) for row in analysis_rows]
    max_pre_context_frames = int(max_pre_context_seconds * fps)
    max_post_context_frames = int(max_post_context_seconds * fps)

    expanded_segments: list[tuple[int, int]] = []
    for start, end in merged_segments:
        start_index = 0
        end_index = len(sample_frames) - 1
        for index, frame_id in enumerate(sample_frames):
            if frame_id <= start:
                start_index = index
            if frame_id <= end:
                end_index = index

        expanded_start = start
        expanded_end = end

        drop_count = 0
        index = start_index - 1
        while index >= 0:
            frame_id = sample_frames[index]
            if start - frame_id > max_pre_context_frames:
                break
            if court_flags[index]:
                expanded_start = frame_id
                drop_count = 0
            else:
                drop_count += 1
                if drop_count > allowed_context_drop_samples:
                    break
            index -= 1

        drop_count = 0
        index = end_index + 1
        while index < len(sample_frames):
            frame_id = sample_frames[index]
            if frame_id - end > max_post_context_frames:
                break
            if court_flags[index]:
                expanded_end = frame_id + sample_every
                drop_count = 0
            else:
                drop_count += 1
                if drop_count > allowed_context_drop_samples:
                    break
            index += 1

        expanded_start = max(0, expanded_start - pad_before_frames)
        expanded_end = expanded_end + pad_after_frames
        if not expanded_segments:
            expanded_segments.append((expanded_start, expanded_end))
            continue

        previous_start, previous_end = expanded_segments[-1]
        if expanded_start - previous_end <= max_gap_frames:
            expanded_segments[-1] = (previous_start, max(previous_end, expanded_end))
        else:
            expanded_segments.append((expanded_start, expanded_end))

    return [
        (start, end)
        for start, end in expanded_segments
        if min_rally_frames <= end - start <= max_rally_frames
    ]


def write_analysis_csv(metadata_csv: Path, analysis_rows: list[dict[str, float]]) -> Path:
    analysis_csv = metadata_csv.with_name(f'{metadata_csv.stem}_analysis.csv')
    fieldnames = [
        'sample_frame',
        'timestamp',
        'motion_score',
        'line_ratio',
        'green_ratio',
        'center_green_ratio',
        'bottom_green_ratio',
        'top_green_ratio',
        'middle_green_ratio',
        'left_green_ratio',
        'right_green_ratio',
        'top_dark_ratio',
        'middle_edge_ratio',
        'is_court_view',
        'is_candidate',
    ]
    with analysis_csv.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(analysis_rows)
    return analysis_csv


def remove_previous_outputs(output_dir: Path, match_id: str) -> int:
    removed = 0
    for path in output_dir.glob(f'{match_id}_rally_*.mp4'):
        path.unlink()
        removed += 1
    return removed


def write_rally_clips(
    input_path: Path,
    output_dir: Path,
    segments: list[tuple[int, int]],
    fps: float,
    match_id: str,
) -> list[dict[str, object]]:
    require_opencv()
    if not segments:
        return []

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f'Cannot open video: {input_path}')

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    rally_rows: list[dict[str, object]] = []
    for rally_index, (start_frame, end_frame) in enumerate(segments, start=1):
        rally_id = f'{rally_index:03d}'
        output_path = output_dir / f'{match_id}_rally_{rally_id}.mp4'
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current_frame = start_frame
        while current_frame <= end_frame:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(frame)
            current_frame += 1

        writer.release()
        rally_rows.append(
            {
                'match_id': match_id,
                'rally_id': rally_id,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'start_time': round(start_frame / fps, 3),
                'end_time': round(end_frame / fps, 3),
                'duration_seconds': round((end_frame - start_frame) / fps, 3),
                'output_path': str(output_path),
                'notes': 'auto-generated candidate segment',
            }
        )

    capture.release()
    return rally_rows


def segment_rallies(
    input_path: Path,
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
    analysis_rows, fps, _, _ = analyze_video(
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
    segments = merge_candidate_segments(
        analysis_rows=analysis_rows,
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

    match_id = extract_match_id(input_path)
    output_dir = ensure_dir(output_dir)
    if overwrite:
        removed = remove_previous_outputs(output_dir, match_id)
        if removed:
            print(f'Removed previous clips: {removed}')
    rally_rows = write_rally_clips(input_path, output_dir, segments, fps, match_id)
    analysis_csv = write_analysis_csv(metadata_csv, analysis_rows)
    fieldnames = [
        'match_id',
        'rally_id',
        'start_frame',
        'end_frame',
        'start_time',
        'end_time',
        'duration_seconds',
        'output_path',
        'notes',
    ]
    write_csv_rows(metadata_csv, fieldnames, rally_rows)

    print(f'Video: {input_path}')
    print(f'FPS: {fps:.3f}')
    print(f'Sampled frames: {len(analysis_rows)}')
    print(f'Candidate rallies: {len(rally_rows)}')
    print(f'Rally metadata saved to: {metadata_csv}')
    print(f'Analysis metrics saved to: {analysis_csv}')
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Segment rally clips from a full match video.')
    parser.add_argument('input', type=Path, help='Path to the full match video.')
    parser.add_argument('--output-dir', type=Path, default=Path('rallies'))
    parser.add_argument(
        '--metadata-csv',
        type=Path,
        default=Path('metadata/rallies.csv'),
        help='Output CSV for generated rally metadata.',
    )
    parser.add_argument(
        '--sample-every',
        type=int,
        default=15,
        help='Analyze one frame every N frames.',
    )
    parser.add_argument(
        '--min-rally-seconds',
        type=float,
        default=4.0,
        help='Minimum duration for a candidate rally.',
    )
    parser.add_argument(
        '--max-rally-seconds',
        type=float,
        default=45.0,
        help='Maximum duration for a candidate rally.',
    )
    parser.add_argument(
        '--max-gap-seconds',
        type=float,
        default=3.0,
        help='Maximum gap to merge adjacent candidate segments.',
    )
    parser.add_argument(
        '--min-motion-score',
        type=float,
        default=0.01,
        help='Minimum motion score required for active play in a full-court view.',
    )
    parser.add_argument(
        '--max-motion-score',
        type=float,
        default=0.16,
        help='Maximum motion score allowed for active play in a full-court view.',
    )
    parser.add_argument(
        '--min-center-green-ratio',
        type=float,
        default=0.22,
        help='Minimum green ratio in the main court area.',
    )
    parser.add_argument(
        '--min-bottom-green-ratio',
        type=float,
        default=0.36,
        help='Minimum green ratio in the lower court area.',
    )
    parser.add_argument(
        '--min-line-ratio',
        type=float,
        default=0.09,
        help='Minimum edge ratio required for a full-court broadcast view.',
    )
    parser.add_argument(
        '--min-top-green-ratio',
        type=float,
        default=0.05,
        help='Minimum green ratio required in the upper part of the court.',
    )
    parser.add_argument(
        '--min-middle-green-ratio',
        type=float,
        default=0.20,
        help='Minimum green ratio required in the middle part of the court.',
    )
    parser.add_argument(
        '--max-left-right-green-diff',
        type=float,
        default=0.16,
        help='Maximum allowed difference between left and right green occupancy.',
    )
    parser.add_argument(
        '--min-top-dark-ratio',
        type=float,
        default=0.80,
        help='Minimum dark-pixel ratio in the upper area to prefer broadcast wide shots.',
    )
    parser.add_argument(
        '--min-middle-edge-ratio',
        type=float,
        default=0.16,
        help='Minimum edge ratio in the middle region to preserve clear court structure.',
    )
    parser.add_argument(
        '--pad-before-seconds',
        type=float,
        default=0.4,
        help='Context padding added before each detected rally.',
    )
    parser.add_argument(
        '--pad-after-seconds',
        type=float,
        default=0.6,
        help='Context padding added after each detected rally.',
    )
    parser.add_argument(
        '--max-pre-context-seconds',
        type=float,
        default=2.2,
        help='Maximum main-view context to extend before an active rally segment.',
    )
    parser.add_argument(
        '--max-post-context-seconds',
        type=float,
        default=1.4,
        help='Maximum main-view context to extend after an active rally segment.',
    )
    parser.add_argument(
        '--allowed-context-drop-samples',
        type=int,
        default=1,
        help='How many non-court sampled frames are tolerated while extending context.',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Delete previously generated clips for the same match before writing new ones.',
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.sample_every <= 0:
        raise SystemExit('--sample-every must be > 0')
    if args.min_rally_seconds <= 0:
        raise SystemExit('--min-rally-seconds must be > 0')
    if args.max_rally_seconds <= args.min_rally_seconds:
        raise SystemExit('--max-rally-seconds must be > --min-rally-seconds')
    if args.max_gap_seconds < 0:
        raise SystemExit('--max-gap-seconds must be >= 0')
    if not math.isfinite(args.min_motion_score) or args.min_motion_score < 0:
        raise SystemExit('--min-motion-score must be a finite value >= 0')
    if not math.isfinite(args.max_motion_score) or args.max_motion_score <= args.min_motion_score:
        raise SystemExit('--max-motion-score must be > --min-motion-score')

    return segment_rallies(
        input_path=args.input,
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


if __name__ == '__main__':
    raise SystemExit(main())

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


def frame_motion_scores(
    current_gray: np.ndarray,
    previous_gray: np.ndarray,
) -> tuple[float, float]:
    """Return full-frame and play-area motion scores.

    Players and the shuttle occupy too few pixels in a wide broadcast frame
    for the full-frame mean to be a reliable hard gate. The normalized play
    area excludes most spectators, scoreboards and side carpets while keeping
    both halves of the court across the supported main-view compositions.
    """
    if current_gray.shape != previous_gray.shape:
        raise ValueError('motion frames must have identical shapes')
    frame_diff = cv2.absdiff(current_gray, previous_gray)
    height, width = frame_diff.shape[:2]
    x0 = int(round(width * 0.0875))
    x1 = int(round(width * 0.9125))
    y0 = int(round(height * 0.1556))
    y1 = int(round(height * 0.9889))
    play_area = frame_diff[y0:y1, x0:x1]
    if play_area.size == 0:
        play_area = frame_diff
    return (
        float(np.mean(frame_diff)) / 255.0,
        float(np.mean(play_area)) / 255.0,
    )


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
            global_motion_score = 0.0
            play_area_motion_score = 0.0
        else:
            global_motion_score, play_area_motion_score = frame_motion_scores(
                gray,
                previous_gray,
            )

        # Keep ``motion_score`` as the activity score consumed by the public
        # segmentation contract. The two component scores remain in the audit
        # CSV so threshold decisions can be reproduced.
        motion_score = play_area_motion_score

        metrics = frame_metrics(frame)
        # Rally boundaries follow the stable full-court broadcast view. Edge
        # density is useful as a confidence signal, but is too brittle to be
        # a boundary signal when small, fast players blur the court lines.
        is_rally_view = (
            metrics['center_green_ratio'] >= min_center_green_ratio
            and metrics['bottom_green_ratio'] >= min_bottom_green_ratio
            and metrics['top_green_ratio'] >= min_top_green_ratio
            and metrics['middle_green_ratio'] >= min_middle_green_ratio
            and abs(metrics['left_green_ratio'] - metrics['right_green_ratio'])
            <= max_left_right_green_diff
            and metrics['top_dark_ratio'] >= min_top_dark_ratio
        )
        is_court_view = (
            is_rally_view
            and metrics['line_ratio'] >= min_line_ratio
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
                'global_motion_score': global_motion_score,
                'play_area_motion_score': play_area_motion_score,
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
                'is_rally_view': float(is_rally_view),
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
    if not analysis_rows:
        return []

    # A rally is bounded by the continuous full-court broadcast shot. Motion
    # is deliberately not used for boundaries: players occupy too few pixels
    # in a wide shot, so whole-frame differences go quiet during real play.
    view_segments: list[tuple[int, int]] = []
    start_frame: int | None = None
    last_view_frame: int | None = None
    drop_count = 0

    for row in analysis_rows:
        frame_id = int(row['sample_frame'])
        is_rally_view = bool(row.get('is_rally_view', row['is_court_view']))
        if is_rally_view:
            if start_frame is None:
                start_frame = frame_id
            last_view_frame = frame_id
            drop_count = 0
        elif start_frame is not None and last_view_frame is not None:
            drop_count += 1
            if drop_count > allowed_context_drop_samples:
                view_segments.append((start_frame, last_view_frame + sample_every))
                start_frame = None
                last_view_frame = None
                drop_count = 0

    if start_frame is not None and last_view_frame is not None:
        view_segments.append((start_frame, last_view_frame + sample_every))

    min_rally_frames = int(min_rally_seconds * fps)
    max_rally_frames = int(max_rally_seconds * fps)
    pad_before_frames = int(pad_before_seconds * fps)
    pad_after_frames = int(pad_after_seconds * fps)

    return [
        (max(0, start - pad_before_frames), end + pad_after_frames)
        for start, end in view_segments
        if min_rally_frames <= end - start <= max_rally_frames
    ]


def _crop_normalized(frame: np.ndarray, roi: tuple[float, float, float, float]) -> np.ndarray:
    height, width = frame.shape[:2]
    x, y, roi_width, roi_height = roi
    x1 = max(0, min(width - 1, int(round(x * width))))
    y1 = max(0, min(height - 1, int(round(y * height))))
    x2 = max(x1 + 1, min(width, int(round((x + roi_width) * width))))
    y2 = max(y1 + 1, min(height, int(round((y + roi_height) * height))))
    return frame[y1:y2, x1:x2]


def detect_score_changes(
    input_path: Path,
    score_roi: tuple[float, float, float, float],
    context_roi: tuple[float, float, float, float],
) -> list[int]:
    """Detect persistent scoreboard-state changes without OCR."""
    require_opencv()
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f'Cannot open video: {input_path}')

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    sample_every = max(1, int(round(fps / 2.0)))
    observations: list[tuple[int, np.ndarray]] = []
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % sample_every == 0:
            context = _crop_normalized(frame, context_roi)
            hsv = cv2.cvtColor(context, cv2.COLOR_BGR2HSV)
            presence = float(np.mean((hsv[..., 1] < 90) & (hsv[..., 2] > 75)))
            if 0.35 <= presence <= 0.43:
                score = _crop_normalized(frame, score_roi)
                gray = cv2.cvtColor(score, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (54, 82), interpolation=cv2.INTER_AREA)
                observations.append((frame_index, (resized > 120).astype(np.uint8)))
        frame_index += 1
    capture.release()

    reference: np.ndarray | None = None
    recent: list[tuple[int, np.ndarray]] = []
    changes: list[int] = []
    for frame_id, fingerprint in observations:
        recent.append((frame_id, fingerprint))
        recent = recent[-3:]
        if len(recent) < 3 or recent[-1][0] - recent[0][0] > int(2.0 * fps):
            continue
        candidate = (
            np.median(np.stack([item[1] for item in recent]), axis=0) >= 0.5
        ).astype(np.uint8)
        stable_difference = max(
            float(np.mean(candidate != item[1])) for item in recent
        )
        if stable_difference > 0.008:
            continue
        if reference is None:
            reference = candidate
            continue
        change = float(np.mean(reference != candidate))
        # A normal one-point update changes only a small digit region. Larger
        # jumps are layout/final-result transitions: adopt the new layout as
        # the baseline without emitting a point.
        if change >= 0.05:
            reference = candidate
            recent = []
        elif change > 0.014:
            reference = candidate
            changes.append(recent[1][0])
            recent = []
    return changes


def select_live_rallies(
    view_segments: list[tuple[int, int]],
    score_change_frames: list[int],
    fps: float,
    max_score_lag_seconds: float = 8.0,
) -> list[tuple[int, int]]:
    """Keep the final full-court shot before each score update."""
    max_lag_frames = int(max_score_lag_seconds * fps)
    end_tolerance = int(2.0 * fps)
    selected: list[tuple[int, int]] = []
    for score_frame in score_change_frames:
        candidates = [
            segment
            for segment in view_segments
            if segment[0] <= score_frame
            and segment[1] <= score_frame + end_tolerance
            and score_frame - segment[1] <= max_lag_frames
        ]
        if not candidates:
            continue
        segment = max(candidates, key=lambda item: item[1])
        if not selected or segment != selected[-1]:
            selected.append(segment)
    return selected


def parse_normalized_roi(value: str) -> tuple[float, float, float, float]:
    parts = tuple(float(item.strip()) for item in value.split(','))
    if len(parts) != 4 or any(item < 0.0 or item > 1.0 for item in parts):
        raise argparse.ArgumentTypeError('ROI must be x,y,width,height with values from 0 to 1')
    if parts[2] <= 0.0 or parts[3] <= 0.0 or parts[0] + parts[2] > 1.0 or parts[1] + parts[3] > 1.0:
        raise argparse.ArgumentTypeError('ROI must fit inside the normalized frame')
    return parts


def write_analysis_csv(metadata_csv: Path, analysis_rows: list[dict[str, float]]) -> Path:
    analysis_csv = metadata_csv.with_name(f'{metadata_csv.stem}_analysis.csv')
    fieldnames = [
        'sample_frame',
        'timestamp',
        'motion_score',
        'global_motion_score',
        'play_area_motion_score',
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
        'is_rally_view',
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
    notes: str = 'auto-generated candidate segment',
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
        if not writer.isOpened():
            capture.release()
            raise RuntimeError(f'Cannot create rally video: {output_path}')

        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current_frame = start_frame
        written_frames = 0
        while current_frame < end_frame:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(frame)
            current_frame += 1
            written_frames += 1

        writer.release()
        expected_frames = end_frame - start_frame
        if written_frames != expected_frames:
            output_path.unlink(missing_ok=True)
            capture.release()
            raise RuntimeError(
                f'Rally interval [{start_frame}, {end_frame}) expected '
                f'{expected_frames} frames but decoded {written_frames}'
            )
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
                'notes': notes,
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
    scoreboard_score_roi: tuple[float, float, float, float] | None = None,
    scoreboard_context_roi: tuple[float, float, float, float] | None = None,
    scoreboard_max_lag_seconds: float = 8.0,
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
    notes = 'full-court view candidate'
    score_changes: list[int] = []
    if scoreboard_score_roi is not None and scoreboard_context_roi is not None:
        score_changes = detect_score_changes(
            input_path,
            scoreboard_score_roi,
            scoreboard_context_roi,
        )
        segments = select_live_rallies(
            segments,
            score_changes,
            fps,
            max_score_lag_seconds=scoreboard_max_lag_seconds,
        )
        notes = 'live rally selected before scoreboard update'

    match_id = extract_match_id(input_path)
    output_dir = ensure_dir(output_dir)
    if overwrite:
        removed = remove_previous_outputs(output_dir, match_id)
        if removed:
            print(f'Removed previous clips: {removed}')
    rally_rows = write_rally_clips(input_path, output_dir, segments, fps, match_id, notes=notes)
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
    if score_changes:
        print(f'Score changes: {len(score_changes)}')
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
        default=1.0,
        help='Minimum duration for a candidate rally.',
    )
    parser.add_argument(
        '--max-rally-seconds',
        type=float,
        default=120.0,
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
        default=0.15,
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
    parser.add_argument(
        '--scoreboard-score-roi',
        type=parse_normalized_roi,
        default=None,
        help='Normalized x,y,width,height for the current-game score cells.',
    )
    parser.add_argument(
        '--scoreboard-context-roi',
        type=parse_normalized_roi,
        default=None,
        help='Normalized x,y,width,height for scoreboard presence detection.',
    )
    parser.add_argument('--scoreboard-max-lag-seconds', type=float, default=8.0)
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
        scoreboard_score_roi=args.scoreboard_score_roi,
        scoreboard_context_roi=args.scoreboard_context_roi,
        scoreboard_max_lag_seconds=args.scoreboard_max_lag_seconds,
    )


if __name__ == '__main__':
    raise SystemExit(main())

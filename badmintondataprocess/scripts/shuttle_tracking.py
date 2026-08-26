from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from common import ensure_dir, read_csv_rows, write_csv_rows

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover - runtime dependency
    cv2 = None
    np = None


TRACK_FIELDNAMES = [
    'video_path',
    'video_stem',
    'rally_id',
    'frame_id',
    'timestamp',
    'x',
    'y',
    'confidence',
    'is_interpolated',
    'visibility',
]

SUMMARY_FIELDNAMES = [
    'video_path',
    'video_stem',
    'status',
    'track_rows',
    'visible_rows',
    'interpolated_rows',
    'debug_video',
    'message',
]


def require_opencv() -> None:
    if cv2 is None or np is None:
        raise RuntimeError(
            'OpenCV and NumPy are required. Install them with: pip install opencv-python numpy'
        )


def iter_video_paths(input_path: Path) -> list[Path]:
    if input_path.is_file() and input_path.suffix.lower() == '.csv':
        return [Path(row['output_path']) for row in read_csv_rows(input_path) if row.get('output_path')]
    if input_path.is_dir():
        return sorted(input_path.glob('*.mp4'))
    return [input_path]


def rally_id_from_stem(video_stem: str) -> str:
    match = re.search(r'_rally_(\d+)$', video_stem)
    return match.group(1) if match else '000'


def load_calibration(calibration_path: Path) -> np.ndarray | None:
    if not calibration_path.exists():
        return None
    payload = json.loads(calibration_path.read_text(encoding='utf-8'))
    return np.array(payload['image_points_tl_tr_br_bl'], dtype=np.float32)


def build_search_mask(frame_shape: tuple[int, int, int], corners: np.ndarray | None) -> np.ndarray:
    mask = np.ones(frame_shape[:2], dtype=np.uint8) * 255
    if corners is None:
        return mask

    base = np.zeros(frame_shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(base, corners.astype(np.int32), 255)
    kernel = np.ones((31, 31), np.uint8)
    dilated = cv2.dilate(base, kernel, iterations=2)
    return dilated


def candidate_points(
    diff_mask: np.ndarray,
    gray_frame: np.ndarray,
    search_mask: np.ndarray,
    min_brightness: int,
    min_candidate_area: float,
    max_candidate_area: float,
    max_candidate_size: int,
) -> list[dict[str, float]]:
    mask = cv2.bitwise_and(diff_mask, search_mask)
    brightness_mask = (gray_frame >= min_brightness).astype(np.uint8) * 255
    mask = cv2.bitwise_and(mask, brightness_mask)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[dict[str, float]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_candidate_area or area > max_candidate_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w > max_candidate_size or h > max_candidate_size:
            continue
        roi = gray_frame[y:y + h, x:x + w]
        brightness = float(np.mean(roi)) if roi.size else 0.0
        motion = float(np.mean(diff_mask[y:y + h, x:x + w])) / 255.0 if roi.size else 0.0
        cx = x + w / 2.0
        cy = y + h / 2.0
        candidates.append(
            {
                'x': float(cx),
                'y': float(cy),
                'brightness': brightness,
                'area': float(area),
                'motion': motion,
            }
        )
    return candidates


def choose_candidate(
    candidates: list[dict[str, float]],
    prev_point: tuple[float, float] | None,
    prev_velocity: tuple[float, float] | None,
    max_jump: float,
    direction_weight: float,
    speed_weight: float,
) -> dict[str, float] | None:
    if not candidates:
        return None

    predicted = None
    if prev_point is not None:
        predicted = prev_point
        if prev_velocity is not None:
            predicted = (prev_point[0] + prev_velocity[0], prev_point[1] + prev_velocity[1])

    best_score = -1e9
    best_candidate: dict[str, float] | None = None
    for candidate in candidates:
        score = candidate['brightness'] + candidate['area'] * 6.0 + candidate['motion'] * 30.0
        if predicted is not None:
            distance = math.hypot(candidate['x'] - predicted[0], candidate['y'] - predicted[1])
            if distance > max_jump:
                continue
            score -= distance * 1.2
        if prev_point is not None and prev_velocity is not None:
            speed = math.hypot(prev_velocity[0], prev_velocity[1])
            candidate_velocity = (
                candidate['x'] - prev_point[0],
                candidate['y'] - prev_point[1],
            )
            candidate_speed = math.hypot(candidate_velocity[0], candidate_velocity[1])
            if speed > 1.0 and candidate_speed > 1.0:
                direction = (
                    prev_velocity[0] * candidate_velocity[0]
                    + prev_velocity[1] * candidate_velocity[1]
                ) / (speed * candidate_speed)
                score += direction * direction_weight
                score -= abs(candidate_speed - speed) * speed_weight
        if score > best_score:
            best_score = score
            best_candidate = candidate
    return best_candidate


def point_is_in_mask(point: tuple[float, float], mask: np.ndarray) -> bool:
    x = min(mask.shape[1] - 1, max(0, int(round(point[0]))))
    y = min(mask.shape[0] - 1, max(0, int(round(point[1]))))
    return bool(mask[y, x] > 0)


def draw_debug(
    frame: np.ndarray,
    corners: np.ndarray | None,
    point: tuple[float, float] | None,
    confidence: float,
    interpolated: bool,
) -> np.ndarray:
    debug = frame.copy()
    if corners is not None:
        cv2.polylines(debug, [corners.astype(np.int32)], True, (0, 255, 255), 2)
    if point is not None:
        color = (0, 255, 0) if not interpolated else (0, 128, 255)
        cv2.circle(debug, (int(round(point[0])), int(round(point[1]))), 6, color, 2)
        label = f'shuttle {confidence:.2f}'
        if interpolated:
            label += ' interp'
        cv2.putText(
            debug,
            label,
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    return debug


def process_video_tracknet(
    video_path: Path,
    calibration_dir: Path,
    debug_dir: Path,
    tracknet_weights: str,
    max_missing_frames: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Track the shuttle using a TrackNet multi-frame detector."""
    from badminton_data_process.tracking.shuttle.tracknet import TrackNetDetector

    require_opencv()
    detector = TrackNetDetector(tracknet_weights)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return [], {
            'video_path': str(video_path),
            'video_stem': video_path.stem,
            'status': 'failed',
            'track_rows': 0,
            'visible_rows': 0,
            'interpolated_rows': 0,
            'debug_video': '',
            'message': 'cannot open video',
        }

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    corners = load_calibration(calibration_dir / f'{video_path.stem}.json')
    search_mask = build_search_mask((height, width, 3), corners)

    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()

    detections = detector.detect_sequence(frames, {'search_mask': search_mask})

    debug_dir = ensure_dir(debug_dir)
    debug_video_path = debug_dir / f'{video_path.stem}.mp4'
    writer = cv2.VideoWriter(
        str(debug_video_path),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height),
    )

    rows: list[dict[str, object]] = []
    rally_id = rally_id_from_stem(video_path.stem)
    visible_rows = 0
    interpolated_rows = 0
    prev_point: tuple[float, float] | None = None
    prev_velocity: tuple[float, float] | None = None
    missing_count = 0

    for frame_id, detection in enumerate(detections):
        point: tuple[float, float] | None = None
        confidence = 0.0
        interpolated = False
        visibility = 0

        if detection['visibility'] and detection['x'] is not None:
            point = (float(detection['x']), float(detection['y']))
            confidence = float(detection['confidence'])
            visibility = 1
            visible_rows += 1
            if prev_point is not None:
                prev_velocity = (point[0] - prev_point[0], point[1] - prev_point[1])
            prev_point = point
            missing_count = 0
        elif prev_point is not None and prev_velocity is not None and missing_count < max_missing_frames:
            decayed_velocity = (prev_velocity[0] * 0.85, prev_velocity[1] * 0.85)
            point = (prev_point[0] + decayed_velocity[0], prev_point[1] + decayed_velocity[1])
            point = (
                float(min(max(point[0], 0.0), width - 1.0)),
                float(min(max(point[1], 0.0), height - 1.0)),
            )
            if point_is_in_mask(point, search_mask):
                confidence = 0.15
                interpolated = True
                visibility = 0
                prev_point = point
                prev_velocity = decayed_velocity
                missing_count += 1
                interpolated_rows += 1
            else:
                prev_point = None
                prev_velocity = None
                missing_count += 1
        else:
            missing_count += 1
            if missing_count > max_missing_frames:
                prev_point = None
                prev_velocity = None

        rows.append(
            {
                'video_path': str(video_path),
                'video_stem': video_path.stem,
                'rally_id': rally_id,
                'frame_id': frame_id,
                'timestamp': round(frame_id / fps, 3),
                'x': '' if point is None else round(point[0], 2),
                'y': '' if point is None else round(point[1], 2),
                'confidence': round(confidence, 3),
                'is_interpolated': int(interpolated),
                'visibility': visibility,
            }
        )
        frame = frames[frame_id] if frame_id < len(frames) else frames[-1]
        writer.write(draw_debug(frame, corners, point, confidence, interpolated))

    writer.release()
    return rows, {
        'video_path': str(video_path),
        'video_stem': video_path.stem,
        'status': 'success',
        'track_rows': len(rows),
        'visible_rows': visible_rows,
        'interpolated_rows': interpolated_rows,
        'debug_video': str(debug_video_path),
        'message': 'ok',
    }


def process_video(
    video_path: Path,
    calibration_dir: Path,
    debug_dir: Path,
    diff_threshold: int,
    max_jump: float,
    max_missing_frames: int,
    min_brightness: int,
    min_candidate_area: float,
    max_candidate_area: float,
    max_candidate_size: int,
    direction_weight: float,
    speed_weight: float,
    model: str = 'motion_bright_baseline',
    tracknet_weights: str = '',
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if model == 'tracknet':
        return process_video_tracknet(
            video_path=video_path,
            calibration_dir=calibration_dir,
            debug_dir=debug_dir,
            tracknet_weights=tracknet_weights,
            max_missing_frames=max_missing_frames,
        )
    require_opencv()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return [], {
            'video_path': str(video_path),
            'video_stem': video_path.stem,
            'status': 'failed',
            'track_rows': 0,
            'visible_rows': 0,
            'interpolated_rows': 0,
            'debug_video': '',
            'message': 'cannot open video',
        }

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    corners = load_calibration(calibration_dir / f'{video_path.stem}.json')
    search_mask = build_search_mask((height, width, 3), corners)

    debug_dir = ensure_dir(debug_dir)
    debug_video_path = debug_dir / f'{video_path.stem}.mp4'
    writer = cv2.VideoWriter(
        str(debug_video_path),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height),
    )

    rows: list[dict[str, object]] = []
    previous_gray: np.ndarray | None = None
    prev_point: tuple[float, float] | None = None
    prev_velocity: tuple[float, float] | None = None
    missing_count = 0
    frame_id = 0
    rally_id = rally_id_from_stem(video_path.stem)
    visible_rows = 0
    interpolated_rows = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        filtered_gray = cv2.GaussianBlur(gray, (3, 3), 0)
        point: tuple[float, float] | None = None
        confidence = 0.0
        interpolated = False
        visibility = 0

        if previous_gray is not None:
            diff = cv2.absdiff(filtered_gray, previous_gray)
            _, diff_mask = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)
            candidates = candidate_points(
                diff_mask,
                gray,
                search_mask,
                min_brightness=min_brightness,
                min_candidate_area=min_candidate_area,
                max_candidate_area=max_candidate_area,
                max_candidate_size=max_candidate_size,
            )
            chosen = choose_candidate(
                candidates,
                prev_point,
                prev_velocity,
                max_jump,
                direction_weight=direction_weight,
                speed_weight=speed_weight,
            )
            if chosen is not None:
                point = (chosen['x'], chosen['y'])
                confidence = min(
                    0.98,
                    max(0.2, chosen['brightness'] / 255.0 * 0.7 + chosen['motion'] * 0.3),
                )
                visibility = 1
                visible_rows += 1
                if prev_point is not None:
                    prev_velocity = (point[0] - prev_point[0], point[1] - prev_point[1])
                prev_point = point
                missing_count = 0
            elif prev_point is not None and prev_velocity is not None and missing_count < max_missing_frames:
                decayed_velocity = (prev_velocity[0] * 0.85, prev_velocity[1] * 0.85)
                point = (prev_point[0] + decayed_velocity[0], prev_point[1] + decayed_velocity[1])
                point = (
                    float(min(max(point[0], 0.0), width - 1.0)),
                    float(min(max(point[1], 0.0), height - 1.0)),
                )
                if point_is_in_mask(point, search_mask):
                    confidence = 0.15
                    interpolated = True
                    visibility = 0
                    prev_point = point
                    prev_velocity = decayed_velocity
                    missing_count += 1
                    interpolated_rows += 1
                else:
                    prev_point = None
                    prev_velocity = None
                    missing_count += 1
            else:
                missing_count += 1
                if missing_count > max_missing_frames:
                    prev_point = None
                    prev_velocity = None
        previous_gray = filtered_gray

        rows.append(
            {
                'video_path': str(video_path),
                'video_stem': video_path.stem,
                'rally_id': rally_id,
                'frame_id': frame_id,
                'timestamp': round(frame_id / fps, 3),
                'x': '' if point is None else round(point[0], 2),
                'y': '' if point is None else round(point[1], 2),
                'confidence': round(confidence, 3),
                'is_interpolated': int(interpolated),
                'visibility': visibility,
            }
        )

        writer.write(draw_debug(frame, corners, point, confidence, interpolated))
        frame_id += 1

    capture.release()
    writer.release()
    return rows, {
        'video_path': str(video_path),
        'video_stem': video_path.stem,
        'status': 'success',
        'track_rows': len(rows),
        'visible_rows': visible_rows,
        'interpolated_rows': interpolated_rows,
        'debug_video': str(debug_video_path),
        'message': 'ok',
    }


def track_shuttle(
    input_path: Path,
    calibration_dir: Path,
    output_csv: Path,
    summary_csv: Path,
    debug_dir: Path,
    diff_threshold: int,
    max_jump: float,
    max_missing_frames: int,
    min_brightness: int,
    min_candidate_area: float,
    max_candidate_area: float,
    max_candidate_size: int,
    direction_weight: float,
    speed_weight: float,
    model: str = 'motion_bright_baseline',
    tracknet_weights: str = '',
) -> int:
    videos = iter_video_paths(input_path)
    all_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []

    for video_path in videos:
        rows, summary = process_video(
            video_path=video_path,
            calibration_dir=calibration_dir,
            debug_dir=debug_dir,
            diff_threshold=diff_threshold,
            max_jump=max_jump,
            max_missing_frames=max_missing_frames,
            min_brightness=min_brightness,
            min_candidate_area=min_candidate_area,
            max_candidate_area=max_candidate_area,
            max_candidate_size=max_candidate_size,
            direction_weight=direction_weight,
            speed_weight=speed_weight,
            model=model,
            tracknet_weights=tracknet_weights,
        )
        all_rows.extend(rows)
        summaries.append(summary)
        print(
            f"{video_path.name}: {summary['status']} "
            f"(visible={summary['visible_rows']}, interpolated={summary['interpolated_rows']})"
        )

    write_csv_rows(output_csv, TRACK_FIELDNAMES, all_rows)
    write_csv_rows(summary_csv, SUMMARY_FIELDNAMES, summaries)
    success_count = sum(row['status'] == 'success' for row in summaries)
    print(f'Videos processed: {len(summaries)}')
    print(f'Successful shuttle runs: {success_count}')
    print(f'Shuttle tracks CSV: {output_csv}')
    print(f'Summary CSV: {summary_csv}')
    return 0 if success_count == len(summaries) else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Track shuttle trajectory in rally clips.')
    parser.add_argument(
        'input',
        type=Path,
        help='Rally video, directory, or rally metadata CSV containing output_path.',
    )
    parser.add_argument(
        '--calibration-dir',
        type=Path,
        default=Path('annotations/court_calibration'),
        help='Directory containing per-rally court calibration JSON files.',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('annotations/shuttle_tracks.csv'),
        help='Output shuttle track CSV.',
    )
    parser.add_argument(
        '--summary-csv',
        type=Path,
        default=Path('annotations/shuttle_tracking_summary.csv'),
        help='Summary CSV for per-video shuttle tracking results.',
    )
    parser.add_argument(
        '--debug-dir',
        type=Path,
        default=Path('outputs/shuttle_tracking_debug'),
        help='Directory for shuttle tracking debug videos.',
    )
    parser.add_argument(
        '--diff-threshold',
        type=int,
        default=18,
        help='Frame difference threshold for moving bright shuttle candidates.',
    )
    parser.add_argument(
        '--max-jump',
        type=float,
        default=80.0,
        help='Maximum allowed shuttle jump between consecutive frames.',
    )
    parser.add_argument(
        '--max-missing-frames',
        type=int,
        default=3,
        help='Maximum consecutive frames to interpolate from velocity.',
    )
    parser.add_argument(
        '--min-brightness',
        type=int,
        default=165,
        help='Minimum grayscale brightness for shuttle candidates.',
    )
    parser.add_argument(
        '--min-candidate-area',
        type=float,
        default=1.0,
        help='Minimum contour area for shuttle candidates.',
    )
    parser.add_argument(
        '--max-candidate-area',
        type=float,
        default=55.0,
        help='Maximum contour area for shuttle candidates.',
    )
    parser.add_argument(
        '--max-candidate-size',
        type=int,
        default=14,
        help='Maximum width or height for shuttle candidate boxes.',
    )
    parser.add_argument(
        '--direction-weight',
        type=float,
        default=24.0,
        help='Score bonus for candidates aligned with previous velocity.',
    )
    parser.add_argument(
        '--speed-weight',
        type=float,
        default=0.35,
        help='Penalty weight for candidates whose speed changes too abruptly.',
    )
    parser.add_argument(
        '--model',
        choices=['motion_bright_baseline', 'tracknet'],
        default='motion_bright_baseline',
        help='Shuttle detection model. "tracknet" uses the multi-frame TrackNet detector.',
    )
    parser.add_argument(
        '--tracknet-weights',
        type=str,
        default='',
        help='Path to the TrackNet checkpoint (required when --model tracknet).',
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return track_shuttle(
        input_path=args.input,
        calibration_dir=args.calibration_dir,
        output_csv=args.output,
        summary_csv=args.summary_csv,
        debug_dir=args.debug_dir,
        diff_threshold=args.diff_threshold,
        max_jump=args.max_jump,
        max_missing_frames=args.max_missing_frames,
        min_brightness=args.min_brightness,
        min_candidate_area=args.min_candidate_area,
        max_candidate_area=args.max_candidate_area,
        max_candidate_size=args.max_candidate_size,
        direction_weight=args.direction_weight,
        speed_weight=args.speed_weight,
        model=args.model,
        tracknet_weights=args.tracknet_weights,
    )


if __name__ == '__main__':
    raise SystemExit(main())

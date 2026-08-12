from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import ensure_dir, read_csv_rows, write_csv_rows

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover - runtime dependency
    cv2 = None
    np = None


COURT_FIELDNAMES = [
    'video_path',
    'video_stem',
    'status',
    'frame_index',
    'json_path',
    'preview_path',
    'message',
]

COURT_POINTS = np.array(
    [
        [0.0, 0.0],
        [6.10, 0.0],
        [6.10, 13.40],
        [0.0, 13.40],
    ],
    dtype=np.float32,
) if np is not None else None


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


def representative_frame_indices(frame_count: int) -> list[int]:
    if frame_count <= 0:
        return [0]
    ratios = [0.5, 0.35, 0.65, 0.2, 0.8]
    indices: list[int] = []
    for ratio in ratios:
        index = max(0, min(frame_count - 1, int(frame_count * ratio)))
        if index not in indices:
            indices.append(index)
    return indices


def read_frame_at(capture: cv2.VideoCapture, frame_index: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f'Failed to read frame {frame_index}')
    return frame


def representative_frame(capture: cv2.VideoCapture) -> tuple[np.ndarray, int]:
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    best_frame: np.ndarray | None = None
    best_index = 0
    best_score = -1.0
    for frame_index in representative_frame_indices(frame_count):
        frame = read_frame_at(capture, frame_index)
        mask = court_mask(frame)
        score = float(np.mean(mask > 0))
        if score > best_score:
            best_score = score
            best_frame = frame
            best_index = frame_index
    if best_frame is None:
        raise RuntimeError('Failed to read representative frame')
    return best_frame, best_index


def court_mask(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    mask = (
        (hue >= 35)
        & (hue <= 95)
        & (saturation >= 35)
        & (value >= 30)
    ).astype(np.uint8) * 255
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def band_points(points: np.ndarray, y_min: float, y_max: float) -> np.ndarray:
    selected = points[(points[:, 1] >= y_min) & (points[:, 1] <= y_max)]
    if len(selected) >= 4:
        return selected
    return points


def detect_court_corners(frame: np.ndarray) -> np.ndarray:
    mask = court_mask(frame)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError('No court contour detected')

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < frame.shape[0] * frame.shape[1] * 0.08:
        raise RuntimeError('Detected court area is too small')

    points = contour.reshape(-1, 2)
    _, y, _, h = cv2.boundingRect(contour)
    top_band = band_points(points, y, y + h * 0.38)
    bottom_band = band_points(points, y + h * 0.58, y + h)

    top_left = top_band[np.argmin(top_band[:, 0])]
    top_right = top_band[np.argmax(top_band[:, 0])]
    bottom_left = bottom_band[np.argmin(bottom_band[:, 0])]
    bottom_right = bottom_band[np.argmax(bottom_band[:, 0])]

    corners = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
    return corners


def draw_preview(frame: np.ndarray, corners: np.ndarray) -> np.ndarray:
    preview = frame.copy()
    cv2.polylines(preview, [corners.astype(np.int32)], True, (0, 255, 255), 3)
    for name, point in zip(['TL', 'TR', 'BR', 'BL'], corners):
        cv2.circle(preview, tuple(point.astype(int)), 6, (0, 0, 255), -1)
        cv2.putText(
            preview,
            name,
            tuple((point + np.array([6, -6])).astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return preview


def save_calibration(
    video_path: Path,
    frame_index: int,
    frame_shape: tuple[int, int, int],
    corners: np.ndarray,
    output_json: Path,
    output_preview: Path,
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_preview.parent.mkdir(parents=True, exist_ok=True)

    homography = cv2.getPerspectiveTransform(corners.astype(np.float32), COURT_POINTS)
    payload = {
        'video_path': str(video_path),
        'frame_index': frame_index,
        'image_size': {'width': int(frame_shape[1]), 'height': int(frame_shape[0])},
        'image_points_tl_tr_br_bl': corners.astype(float).tolist(),
        'court_points_tl_tr_br_bl': COURT_POINTS.astype(float).tolist(),
        'homography_image_to_court': homography.astype(float).tolist(),
    }
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def calibrate_video(video_path: Path, output_dir: Path, preview_dir: Path) -> dict[str, object]:
    require_opencv()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return {
            'video_path': str(video_path),
            'video_stem': video_path.stem,
            'status': 'failed',
            'frame_index': -1,
            'json_path': '',
            'preview_path': '',
            'message': 'cannot open video',
        }

    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        candidate_indices = representative_frame_indices(frame_count)
        last_error: str | None = None
        frame = None
        frame_index = -1
        corners = None
        for candidate_index in candidate_indices:
            try:
                candidate_frame = read_frame_at(capture, candidate_index)
                candidate_corners = detect_court_corners(candidate_frame)
                frame = candidate_frame
                frame_index = candidate_index
                corners = candidate_corners
                break
            except Exception as exc:  # pragma: no cover - runtime dependent
                last_error = str(exc)
                continue
        if frame is None or corners is None:
            raise RuntimeError(last_error or 'Failed to detect court corners')
        output_json = output_dir / f'{video_path.stem}.json'
        output_preview = preview_dir / f'{video_path.stem}.png'
        save_calibration(video_path, frame_index, frame.shape, corners, output_json, output_preview)
        preview = draw_preview(frame, corners)
        cv2.imwrite(str(output_preview), preview)
        return {
            'video_path': str(video_path),
            'video_stem': video_path.stem,
            'status': 'success',
            'frame_index': frame_index,
            'json_path': str(output_json),
            'preview_path': str(output_preview),
            'message': 'ok',
        }
    except Exception as exc:  # pragma: no cover - runtime dependent
        return {
            'video_path': str(video_path),
            'video_stem': video_path.stem,
            'status': 'failed',
            'frame_index': -1,
            'json_path': '',
            'preview_path': '',
            'message': str(exc),
        }
    finally:
        capture.release()


def calibrate_courts(input_path: Path, output_dir: Path, preview_dir: Path, summary_csv: Path) -> int:
    videos = iter_video_paths(input_path)
    output_dir = ensure_dir(output_dir)
    preview_dir = ensure_dir(preview_dir)

    rows = [calibrate_video(video_path, output_dir, preview_dir) for video_path in videos]
    write_csv_rows(summary_csv, COURT_FIELDNAMES, rows)

    success_count = sum(row['status'] == 'success' for row in rows)
    print(f'Videos processed: {len(rows)}')
    print(f'Successful calibrations: {success_count}')
    print(f'Summary CSV: {summary_csv}')
    return 0 if success_count == len(rows) else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Calibrate badminton court homography.')
    parser.add_argument(
        'input',
        type=Path,
        help='Video file, rally directory, or rally metadata CSV containing output_path.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('annotations/court_calibration'),
        help='Directory for per-video calibration JSON files.',
    )
    parser.add_argument(
        '--preview-dir',
        type=Path,
        default=Path('outputs/court_calibration_debug'),
        help='Directory for preview images with detected court corners.',
    )
    parser.add_argument(
        '--summary-csv',
        type=Path,
        default=Path('annotations/court_calibration_summary.csv'),
        help='Summary CSV describing calibration success for each video.',
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return calibrate_courts(
        input_path=args.input,
        output_dir=args.output_dir,
        preview_dir=args.preview_dir,
        summary_csv=args.summary_csv,
    )


if __name__ == '__main__':
    raise SystemExit(main())

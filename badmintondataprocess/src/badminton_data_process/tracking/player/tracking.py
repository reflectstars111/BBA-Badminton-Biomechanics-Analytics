from __future__ import annotations

import argparse
import json
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from badminton_data_process.core.io import ensure_dir, read_csv_rows, write_csv_rows

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover - runtime dependency
    cv2 = None
    np = None

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - optional runtime dependency
    YOLO = None


__all__ = [
    'TRACK_FIELDNAMES',
    'SUMMARY_FIELDNAMES',
    'require_opencv',
    'iter_video_paths',
    'rally_id_from_stem',
    'load_calibration',
    'build_court_mask',
    'green_mask',
    'contour_candidates',
    'detect_players_heuristic',
    'load_yolo_model',
    'bbox_bottom_center',
    'shift_bbox',
    'clamp_bbox_to_frame',
    'iou',
    'point_distance',
    'detect_players_yolo',
    'init_track_states',
    'candidate_role_score',
    'predict_track_bbox',
    'pick_player_boxes',
    'project_point',
    'draw_debug_frame',
    'detect_players',
    'process_video',
    'track_players',
    'build_parser',
    'main',
]

TRACK_FIELDNAMES = [
    'video_path',
    'video_stem',
    'rally_id',
    'frame_id',
    'timestamp',
    'player_id',
    'bbox_x1',
    'bbox_y1',
    'bbox_x2',
    'bbox_y2',
    'image_x',
    'image_y',
    'court_x',
    'court_y',
    'confidence',
    'is_interpolated',
    'detector',
]

SUMMARY_FIELDNAMES = [
    'video_path',
    'video_stem',
    'status',
    'track_rows',
    'debug_video',
    'detector',
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


def load_calibration(calibration_path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(calibration_path.read_text(encoding='utf-8'))
    corners = np.array(payload['image_points_tl_tr_br_bl'], dtype=np.float32)
    homography = np.array(payload['homography_image_to_court'], dtype=np.float32)
    return corners, homography


def build_court_mask(frame_shape: tuple[int, int, int], corners: np.ndarray) -> np.ndarray:
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, corners.astype(np.int32), 255)
    return mask


def green_mask(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    return (
        (hue >= 35)
        & (hue <= 95)
        & (saturation >= 35)
        & (value >= 30)
    ).astype(np.uint8) * 255


def contour_candidates(
    frame: np.ndarray,
    fg_mask: np.ndarray,
    court_mask: np.ndarray,
) -> list[dict[str, object]]:
    non_green = cv2.bitwise_not(green_mask(frame))
    mask = cv2.bitwise_and(fg_mask, court_mask)
    mask = cv2.bitwise_and(mask, non_green)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[dict[str, object]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 120 or area > 15000:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if h < 22 or w < 8:
            continue
        bottom_center = (x + w / 2.0, y + h)
        candidates.append(
            {
                'bbox': (x, y, x + w, y + h),
                'score': min(0.8, 0.2 + area / 2000.0),
                'bottom_center': bottom_center,
            }
        )
    return candidates


def detect_players_heuristic(
    frame: np.ndarray,
    fg_mask: np.ndarray,
    court_mask: np.ndarray,
) -> list[dict[str, object]]:
    return contour_candidates(frame, fg_mask, court_mask)


@lru_cache(maxsize=4)
def load_yolo_model(model_name: str) -> Any:
    if YOLO is None:
        raise RuntimeError(
            'ultralytics is not installed. Run: .venv/bin/pip install ultralytics scipy'
        )
    return YOLO(model_name)


def bbox_bottom_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, _, x2, y2 = bbox
    return (float(x1 + x2) / 2.0, float(y2))


def shift_bbox(
    bbox: tuple[int, int, int, int],
    dx: float,
    dy: float,
    frame_shape: tuple[int, int, int],
) -> tuple[int, int, int, int]:
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    shifted = (
        int(round(x1 + dx)),
        int(round(y1 + dy)),
        int(round(x2 + dx)),
        int(round(y2 + dy)),
    )
    sx1 = max(0, min(width - 2, shifted[0]))
    sy1 = max(0, min(height - 2, shifted[1]))
    sx2 = max(sx1 + 1, min(width - 1, shifted[2]))
    sy2 = max(sy1 + 1, min(height - 1, shifted[3]))
    return (sx1, sy1, sx2, sy2)


def clamp_bbox_to_frame(
    bbox: tuple[int, int, int, int],
    frame_shape: tuple[int, int, int],
) -> tuple[int, int, int, int]:
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(width - 2, x1))
    x2 = max(x1 + 1, min(width - 1, x2))
    y1 = max(0, min(height - 2, y1))
    y2 = max(y1 + 1, min(height - 1, y2))
    return (x1, y1, x2, y2)


def iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = float(inter_w * inter_h)
    if inter_area <= 0.0:
        return 0.0
    area_a = float((ax2 - ax1) * (ay2 - ay1))
    area_b = float((bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0.0 else 0.0


def point_distance(point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
    return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])


def detect_players_yolo(
    frame: np.ndarray,
    court_mask: np.ndarray,
    model_name: str,
    confidence_threshold: float,
    image_size: int,
) -> list[dict[str, object]]:
    model = load_yolo_model(model_name)
    results = model.predict(
        source=frame,
        conf=confidence_threshold,
        classes=[0],
        verbose=False,
        imgsz=image_size,
        max_det=10,
    )
    candidates: list[dict[str, object]] = []
    if not results:
        return candidates

    result = results[0]
    boxes = getattr(result, 'boxes', None)
    if boxes is None:
        return candidates

    for xyxy, conf in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), strict=True):
        x1, y1, x2, y2 = [int(round(value)) for value in xyxy.tolist()]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1] - 1, x2)
        y2 = min(frame.shape[0] - 1, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        bottom_center = bbox_bottom_center((x1, y1, x2, y2))
        bx = min(frame.shape[1] - 1, max(0, int(round(bottom_center[0]))))
        by = min(frame.shape[0] - 1, max(0, int(round(bottom_center[1]))))
        if court_mask[by, bx] == 0:
            continue

        area = float((x2 - x1) * (y2 - y1))
        candidates.append(
            {
                'bbox': (x1, y1, x2, y2),
                'score': float(conf),
                'bottom_center': bottom_center,
                'area': area,
            }
        )
    return candidates


def init_track_states() -> dict[str, dict[str, object]]:
    return {
        'far': {
            'bbox': None,
            'velocity': (0.0, 0.0),
            'miss_count': 0,
        },
        'near': {
            'bbox': None,
            'velocity': (0.0, 0.0),
            'miss_count': 0,
        },
    }


def candidate_role_score(
    candidate: dict[str, object],
    player_id: str,
    near_threshold_y: float,
    predicted_bbox: tuple[int, int, int, int] | None,
    role_half_tolerance: float,
) -> float:
    bbox = candidate['bbox']
    score = float(candidate['score']) * 120.0
    bottom_x, bottom_y = candidate['bottom_center']
    area = float(candidate.get('area', 0.0))
    is_near = player_id == 'near'
    y_delta = bottom_y - near_threshold_y
    if is_near:
        if y_delta >= 0:
            score += min(35.0, y_delta * 0.22)
        else:
            score -= abs(y_delta) * 4.2
    else:
        if y_delta <= 0:
            score += min(30.0, abs(y_delta) * 0.22)
        else:
            score -= y_delta * 4.2

    if is_near:
        score += min(18.0, area / 1200.0)
    else:
        score -= abs(area - 5000.0) / 1400.0

    if predicted_bbox is not None:
        predicted_bottom = bbox_bottom_center(predicted_bbox)
        center_distance = point_distance((bottom_x, bottom_y), predicted_bottom)
        score -= center_distance * (0.35 if is_near else 0.22)
        score += iou(bbox, predicted_bbox) * 50.0

    if player_id == 'far':
        top_margin = max(0.0, role_half_tolerance - max(0.0, bottom_y - near_threshold_y))
        score += top_margin * 0.18
    else:
        bottom_margin = max(0.0, role_half_tolerance - max(0.0, near_threshold_y - bottom_y))
        score += bottom_margin * 0.18
    return score


def predict_track_bbox(
    track_state: dict[str, object],
    frame_shape: tuple[int, int, int],
) -> tuple[int, int, int, int] | None:
    bbox = track_state.get('bbox')
    if bbox is None:
        return None
    dx, dy = track_state.get('velocity', (0.0, 0.0))
    miss_count = int(track_state.get('miss_count', 0))
    decay = max(0.25, 0.82 ** miss_count)
    return shift_bbox(bbox, dx * decay, dy * decay, frame_shape)


def pick_player_boxes(
    candidates: list[dict[str, object]],
    near_threshold_y: float,
    track_states: dict[str, dict[str, object]],
    frame_shape: tuple[int, int, int],
    near_max_track_distance: float,
    far_max_track_distance: float,
    near_max_missing_frames: int,
    far_max_missing_frames: int,
    role_half_tolerance: float,
) -> dict[str, tuple[tuple[int, int, int, int] | None, float, bool]]:
    selected: dict[str, tuple[tuple[int, int, int, int] | None, float, bool]] = {}
    used_candidate_indices: set[int] = set()
    for player_id in ['near', 'far']:
        predicted_bbox = predict_track_bbox(track_states[player_id], frame_shape)
        if predicted_bbox is not None:
            predicted_bbox = clamp_bbox_to_frame(
                predicted_bbox,
                frame_shape=frame_shape,
            )
        predicted_bottom = (
            bbox_bottom_center(predicted_bbox) if predicted_bbox is not None else None
        )
        max_track_distance = far_max_track_distance if player_id == 'far' else near_max_track_distance
        role_candidates: list[tuple[float, int, dict[str, object]]] = []
        for candidate_index, candidate in enumerate(candidates):
            if candidate_index in used_candidate_indices:
                continue
            candidate_bottom = candidate['bottom_center']
            if player_id == 'far' and candidate_bottom[1] > near_threshold_y + role_half_tolerance:
                continue
            if player_id == 'near' and candidate_bottom[1] < near_threshold_y - role_half_tolerance:
                continue
            if player_id == 'far' and candidate_bottom[1] > near_threshold_y:
                continue
            if player_id == 'near' and candidate_bottom[1] < near_threshold_y:
                continue
            if predicted_bottom is not None:
                distance = point_distance(candidate_bottom, predicted_bottom)
                if distance > max_track_distance:
                    continue
            score = candidate_role_score(
                candidate=candidate,
                player_id=player_id,
                near_threshold_y=near_threshold_y,
                predicted_bbox=predicted_bbox,
                role_half_tolerance=role_half_tolerance,
            )
            role_candidates.append((score, candidate_index, candidate))

        if role_candidates:
            _, selected_index, best = max(role_candidates, key=lambda item: item[0])
            used_candidate_indices.add(selected_index)
            bbox = clamp_bbox_to_frame(
                best['bbox'],
                frame_shape=frame_shape,
            )
            previous_bbox = track_states[player_id].get('bbox')
            if previous_bbox is not None:
                prev_bottom = bbox_bottom_center(previous_bbox)
                curr_bottom = bbox_bottom_center(bbox)
                track_states[player_id]['velocity'] = (
                    curr_bottom[0] - prev_bottom[0],
                    curr_bottom[1] - prev_bottom[1],
                )
            track_states[player_id]['bbox'] = bbox
            track_states[player_id]['miss_count'] = 0
            selected[player_id] = (bbox, float(best['score']), False)
            continue

        miss_count = int(track_states[player_id].get('miss_count', 0)) + 1
        track_states[player_id]['miss_count'] = miss_count
        max_missing_frames = far_max_missing_frames if player_id == 'far' else near_max_missing_frames
        if predicted_bbox is not None and miss_count <= max_missing_frames:
            predicted_bbox = clamp_bbox_to_frame(
                predicted_bbox,
                frame_shape=frame_shape,
            )
            track_states[player_id]['bbox'] = predicted_bbox
            dx, dy = track_states[player_id].get('velocity', (0.0, 0.0))
            track_states[player_id]['velocity'] = (dx * 0.82, dy * 0.82)
            selected[player_id] = (predicted_bbox, 0.18 if player_id == 'far' else 0.16, True)
        else:
            track_states[player_id]['bbox'] = None
            track_states[player_id]['velocity'] = (0.0, 0.0)
            selected[player_id] = (None, 0.0, True)
    return selected


def project_point(homography: np.ndarray, point: tuple[float, float]) -> tuple[float, float]:
    src = np.array([[[point[0], point[1]]]], dtype=np.float32)
    projected = cv2.perspectiveTransform(src, homography)[0, 0]
    return float(projected[0]), float(projected[1])


def draw_debug_frame(
    frame: np.ndarray,
    corners: np.ndarray,
    player_boxes: dict[str, tuple[tuple[int, int, int, int] | None, float, bool]],
    detector: str,
) -> np.ndarray:
    debug = frame.copy()
    cv2.polylines(debug, [corners.astype(np.int32)], True, (0, 255, 255), 2)
    colors = {'near': (0, 255, 0), 'far': (255, 128, 0)}
    cv2.putText(
        debug,
        f'detector: {detector}',
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    for player_id, (bbox, confidence, interpolated) in player_boxes.items():
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        color = colors[player_id]
        cv2.rectangle(debug, (x1, y1), (x2, y2), color, 2)
        label = f'{player_id} {confidence:.2f}'
        if interpolated:
            label += ' interp'
        cv2.putText(
            debug,
            label,
            (x1, max(40, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )
    return debug


def detect_players(
    frame: np.ndarray,
    detector: str,
    fg_mask: np.ndarray | None,
    court_mask: np.ndarray,
    yolo_model_name: str,
    yolo_confidence: float,
    yolo_image_size: int,
) -> list[dict[str, object]]:
    if detector == 'yolo':
        return detect_players_yolo(
            frame=frame,
            court_mask=court_mask,
            model_name=yolo_model_name,
            confidence_threshold=yolo_confidence,
            image_size=yolo_image_size,
        )
    if fg_mask is None:
        raise RuntimeError('heuristic detector requires foreground mask')
    return detect_players_heuristic(frame, fg_mask, court_mask)


def process_video(
    video_path: Path,
    calibration_dir: Path,
    debug_dir: Path,
    detector: str,
    yolo_model_name: str,
    yolo_confidence: float,
    yolo_image_size: int,
    near_max_track_distance: float,
    far_max_track_distance: float,
    near_max_missing_frames: int,
    far_max_missing_frames: int,
    role_half_tolerance: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    require_opencv()
    calibration_path = calibration_dir / f'{video_path.stem}.json'
    if not calibration_path.exists():
        return [], {
            'video_path': str(video_path),
            'video_stem': video_path.stem,
            'status': 'failed',
            'track_rows': 0,
            'debug_video': '',
            'detector': detector,
            'message': f'missing calibration: {calibration_path}',
        }

    corners, homography = load_calibration(calibration_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return [], {
            'video_path': str(video_path),
            'video_stem': video_path.stem,
            'status': 'failed',
            'track_rows': 0,
            'debug_video': '',
            'detector': detector,
            'message': 'cannot open video',
        }

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    court_mask = build_court_mask((height, width, 3), corners)
    near_threshold_y = float((corners[0, 1] + corners[1, 1] + corners[2, 1] + corners[3, 1]) / 4.0)
    background_subtractor = None
    if detector == 'heuristic':
        background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=120,
            varThreshold=25,
            detectShadows=False,
        )

    debug_dir = ensure_dir(debug_dir)
    debug_video_path = debug_dir / f'{video_path.stem}.mp4'
    writer = cv2.VideoWriter(
        str(debug_video_path),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height),
    )

    rows: list[dict[str, object]] = []
    track_states = init_track_states()
    frame_id = 0
    rally_id = rally_id_from_stem(video_path.stem)

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            fg_mask = None
            if background_subtractor is not None:
                fg_mask = background_subtractor.apply(frame)
                _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

            candidates = detect_players(
                frame=frame,
                detector=detector,
                fg_mask=fg_mask,
                court_mask=court_mask,
                yolo_model_name=yolo_model_name,
                yolo_confidence=yolo_confidence,
                yolo_image_size=yolo_image_size,
            )
            player_boxes = pick_player_boxes(
                candidates=candidates,
                near_threshold_y=near_threshold_y,
                track_states=track_states,
                frame_shape=frame.shape,
                near_max_track_distance=near_max_track_distance,
                far_max_track_distance=far_max_track_distance,
                near_max_missing_frames=near_max_missing_frames,
                far_max_missing_frames=far_max_missing_frames,
                role_half_tolerance=role_half_tolerance,
            )

            for player_id, (bbox, confidence, interpolated) in player_boxes.items():
                if bbox is None:
                    continue
                x1, y1, x2, y2 = bbox
                image_x = (x1 + x2) / 2.0
                image_y = float(y2)
                court_x, court_y = project_point(homography, (image_x, image_y))
                rows.append(
                    {
                        'video_path': str(video_path),
                        'video_stem': video_path.stem,
                        'rally_id': rally_id,
                        'frame_id': frame_id,
                        'timestamp': round(frame_id / fps, 3),
                        'player_id': player_id,
                        'bbox_x1': x1,
                        'bbox_y1': y1,
                        'bbox_x2': x2,
                        'bbox_y2': y2,
                        'image_x': round(image_x, 2),
                        'image_y': round(image_y, 2),
                        'court_x': round(court_x, 3),
                        'court_y': round(court_y, 3),
                        'confidence': round(confidence, 3),
                        'is_interpolated': int(interpolated),
                        'detector': detector,
                    }
                )

            debug_frame = draw_debug_frame(frame, corners, player_boxes, detector)
            writer.write(debug_frame)
            frame_id += 1
    except Exception as exc:  # pragma: no cover - runtime dependent
        capture.release()
        writer.release()
        return rows, {
            'video_path': str(video_path),
            'video_stem': video_path.stem,
            'status': 'failed',
            'track_rows': len(rows),
            'debug_video': str(debug_video_path),
            'detector': detector,
            'message': str(exc),
        }

    capture.release()
    writer.release()
    return rows, {
        'video_path': str(video_path),
        'video_stem': video_path.stem,
        'status': 'success',
        'track_rows': len(rows),
        'debug_video': str(debug_video_path),
        'detector': detector,
        'message': 'ok',
    }


def track_players(
    input_path: Path,
    calibration_dir: Path,
    output_csv: Path,
    summary_csv: Path,
    debug_dir: Path,
    detector: str,
    yolo_model_name: str,
    yolo_confidence: float,
    yolo_image_size: int,
    near_max_track_distance: float,
    far_max_track_distance: float,
    near_max_missing_frames: int,
    far_max_missing_frames: int,
    role_half_tolerance: float,
) -> int:
    videos = iter_video_paths(input_path)
    all_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for video_path in videos:
        rows, summary = process_video(
            video_path=video_path,
            calibration_dir=calibration_dir,
            debug_dir=debug_dir,
            detector=detector,
            yolo_model_name=yolo_model_name,
            yolo_confidence=yolo_confidence,
            yolo_image_size=yolo_image_size,
            near_max_track_distance=near_max_track_distance,
            far_max_track_distance=far_max_track_distance,
            near_max_missing_frames=near_max_missing_frames,
            far_max_missing_frames=far_max_missing_frames,
            role_half_tolerance=role_half_tolerance,
        )
        all_rows.extend(rows)
        summaries.append(summary)
        print(f"{video_path.name}: {summary['status']} ({summary['track_rows']} rows)")

    write_csv_rows(output_csv, TRACK_FIELDNAMES, all_rows)
    write_csv_rows(summary_csv, SUMMARY_FIELDNAMES, summaries)
    success_count = sum(row['status'] == 'success' for row in summaries)
    print(f'Videos processed: {len(summaries)}')
    print(f'Successful tracking runs: {success_count}')
    print(f'Player tracks CSV: {output_csv}')
    print(f'Summary CSV: {summary_csv}')
    return 0 if success_count == len(summaries) else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Track badminton players in rally clips.')
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
        default=Path('annotations/player_tracks.csv'),
        help='Combined player track CSV.',
    )
    parser.add_argument(
        '--summary-csv',
        type=Path,
        default=Path('annotations/player_tracking_summary.csv'),
        help='Summary CSV for per-video tracking results.',
    )
    parser.add_argument(
        '--debug-dir',
        type=Path,
        default=Path('outputs/player_tracking_debug'),
        help='Directory for debug videos with heuristic tracks.',
    )
    parser.add_argument(
        '--detector',
        choices=['heuristic', 'yolo'],
        default='heuristic',
        help='Player detector backend.',
    )
    parser.add_argument(
        '--yolo-model',
        type=str,
        default='yolov8n.pt',
        help='YOLO model name or path for person detection.',
    )
    parser.add_argument(
        '--yolo-confidence',
        type=float,
        default=0.12,
        help='Confidence threshold used by YOLO person detection.',
    )
    parser.add_argument(
        '--yolo-image-size',
        type=int,
        default=1280,
        help='Inference image size used by YOLO.',
    )
    parser.add_argument(
        '--near-max-track-distance',
        type=float,
        default=120.0,
        help='Maximum allowed center jump for near-player association.',
    )
    parser.add_argument(
        '--far-max-track-distance',
        type=float,
        default=170.0,
        help='Maximum allowed center jump for far-player association.',
    )
    parser.add_argument(
        '--near-max-missing-frames',
        type=int,
        default=4,
        help='Maximum interpolated frames for near-player tracks.',
    )
    parser.add_argument(
        '--far-max-missing-frames',
        type=int,
        default=10,
        help='Maximum interpolated frames for far-player tracks.',
    )
    parser.add_argument(
        '--role-half-tolerance',
        type=float,
        default=48.0,
        help='Vertical tolerance around the mid court split when assigning far/near roles.',
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return track_players(
        input_path=args.input,
        calibration_dir=args.calibration_dir,
        output_csv=args.output,
        summary_csv=args.summary_csv,
        debug_dir=args.debug_dir,
        detector=args.detector,
        yolo_model_name=args.yolo_model,
        yolo_confidence=args.yolo_confidence,
        yolo_image_size=args.yolo_image_size,
        near_max_track_distance=args.near_max_track_distance,
        far_max_track_distance=args.far_max_track_distance,
        near_max_missing_frames=args.near_max_missing_frames,
        far_max_missing_frames=args.far_max_missing_frames,
        role_half_tolerance=args.role_half_tolerance,
    )


if __name__ == '__main__':
    raise SystemExit(main())

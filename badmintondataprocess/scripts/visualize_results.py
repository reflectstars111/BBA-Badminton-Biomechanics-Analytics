from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import ensure_dir, read_csv_rows

try:
    import cv2
except ImportError:  # pragma: no cover - runtime dependency
    cv2 = None

import numpy as np

PLAYER_COLORS = {'near': (0, 0, 255), 'far': (255, 0, 0)}


def load_shuttle_by_frame(rows: list[dict[str, str]]) -> dict[int, dict[str, str]]:
    by_frame: dict[int, dict[str, str]] = {}
    for row in rows:
        try:
            by_frame[int(row['frame_id'])] = row
        except (ValueError, KeyError):
            continue
    return by_frame


def load_players_by_frame(rows: list[dict[str, str]]) -> dict[int, dict[str, dict[str, str]]]:
    by_frame: dict[int, dict[str, dict[str, str]]] = {}
    for row in rows:
        try:
            frame_id = int(row['frame_id'])
            player_id = row['player_id']
        except (ValueError, KeyError):
            continue
        by_frame.setdefault(frame_id, {})[player_id] = row
    return by_frame


def parse_float(value: str | None) -> float | None:
    if value in (None, ''):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_corners(calibration_dir: Path, video_stem: str) -> list[tuple[float, float]] | None:
    calibration_path = calibration_dir / f'{video_stem}.json'
    if not calibration_path.exists():
        return None
    payload = json.loads(calibration_path.read_text(encoding='utf-8'))
    points = payload.get('image_points_tl_tr_br_bl')
    if not points:
        return None
    return [(float(point[0]), float(point[1])) for point in points]


def draw_overlay(
    frame,
    shuttle_rows: dict[int, dict[str, str]],
    player_rows: dict[int, dict[str, dict[str, str]]],
    corners: list[tuple[float, float]] | None,
    frame_id: int,
    trail: list[tuple[float, float]],
) -> None:
    if corners is not None:
        poly = np.array([(int(x), int(y)) for x, y in corners], dtype=np.int32)
        cv2.polylines(frame, [poly], True, (0, 255, 255), 2)

    shuttle = shuttle_rows.get(frame_id)
    if shuttle:
        x = parse_float(shuttle.get('x'))
        y = parse_float(shuttle.get('y'))
        if x is not None and y is not None:
            point = (int(round(x)), int(round(y)))
            is_interp = shuttle.get('is_interpolated') == '1'
            color = (0, 128, 255) if is_interp else (0, 255, 0)
            cv2.circle(frame, point, 6, color, 2)
            trail.append(point)
            label = 'shuttle'
            if is_interp:
                label += ' interp'
            cv2.putText(frame, label, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    if len(trail) > 12:
        trail.pop(0)
    for i in range(1, len(trail)):
        alpha = i / len(trail)
        thickness = max(1, int(3 * alpha))
        cv2.line(frame, trail[i - 1], trail[i], (0, 255, 0), thickness)

    for player_id, row in player_rows.get(frame_id, {}).items():
        x1 = parse_float(row.get('bbox_x1'))
        y1 = parse_float(row.get('bbox_y1'))
        x2 = parse_float(row.get('bbox_x2'))
        y2 = parse_float(row.get('bbox_y2'))
        if x1 is None or y1 is None or x2 is None or y2 is None:
            continue
        color = PLAYER_COLORS.get(player_id, (255, 255, 255))
        cv2.rectangle(
            frame,
            (int(round(x1)), int(round(y1))),
            (int(round(x2)), int(round(y2))),
            color,
            2,
        )
        cv2.putText(
            frame,
            player_id,
            (int(round(x1)), int(round(y1)) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )


def visualize(
    input_video: Path,
    output_video: Path,
    shuttle_track_csv: Path | None = None,
    player_track_csv: Path | None = None,
    calibration_dir: Path | None = None,
) -> None:
    if cv2 is None:
        raise RuntimeError('OpenCV is required. Install it with: pip install opencv-python')
    output_video.parent.mkdir(parents=True, exist_ok=True)

    video_stem = input_video.stem
    shuttle_rows = (
        load_shuttle_by_frame(read_csv_rows(shuttle_track_csv))
        if shuttle_track_csv is not None
        else {}
    )
    player_rows = (
        load_players_by_frame(read_csv_rows(player_track_csv))
        if player_track_csv is not None
        else {}
    )
    corners = load_corners(calibration_dir, video_stem) if calibration_dir is not None else None

    capture = cv2.VideoCapture(str(input_video))
    if not capture.isOpened():
        raise RuntimeError(f'Cannot open video: {input_video}')
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height),
    )

    trail: list[tuple[float, float]] = []
    frame_id = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        draw_overlay(frame, shuttle_rows, player_rows, corners, frame_id, trail)
        writer.write(frame)
        frame_id += 1
    capture.release()
    writer.release()
    print(f'Rendered {frame_id} frames -> {output_video}')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Visualize tracking and tactical analysis outputs.')
    parser.add_argument('input', type=Path, help='Input rally video.')
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('outputs/trajectory_videos/visualization.mp4'),
        help='Output visualization video path.',
    )
    parser.add_argument('--shuttle-track-csv', type=Path, default=None)
    parser.add_argument('--player-track-csv', type=Path, default=None)
    parser.add_argument('--calibration-dir', type=Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    visualize(
        args.input,
        args.output,
        shuttle_track_csv=args.shuttle_track_csv,
        player_track_csv=args.player_track_csv,
        calibration_dir=args.calibration_dir,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

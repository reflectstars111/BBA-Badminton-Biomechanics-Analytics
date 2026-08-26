from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from common import ensure_dir, read_csv_rows, write_csv_rows

COURT_WIDTH_M = 6.1
COURT_LENGTH_M = 13.4
NET_Y_M = 6.7
FRONT_DEPTH_M = 2.5
BACK_DEPTH_M = 4.5

TACTICS_SUMMARY_FIELDS = [
    'video_path',
    'video_stem',
    'rally_id',
    'player_id',
    'frames_valid',
    'total_distance_m',
    'avg_speed_m_s',
    'coverage_area_m2',
    'mean_court_x',
    'mean_court_y',
    'front_court_ratio',
    'mid_court_ratio',
    'back_court_ratio',
    'hit_count',
    'landing_count',
]

TACTICS_EVENT_FIELDS = [
    'video_path',
    'video_stem',
    'rally_id',
    'frame_id',
    'timestamp',
    'event_type',
    'player_id',
    'court_x',
    'court_y',
]


def parse_float(value: str | None) -> float | None:
    if value in (None, ''):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_homography(calibration_dir: Path, video_stem: str) -> list[list[float]] | None:
    path = Path(calibration_dir) / f'{video_stem}.json'
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding='utf-8'))
    matrix = payload.get('homography_image_to_court')
    if not matrix:
        return None
    return [[float(value) for value in row] for row in matrix]


def image_to_court(x: float, y: float, h: list[list[float]]) -> tuple[float, float] | None:
    w = h[0][0] * x + h[0][1] * y + h[0][2]
    v = h[1][0] * x + h[1][1] * y + h[1][2]
    s = h[2][0] * x + h[2][1] * y + h[2][2]
    if abs(s) < 1e-9:
        return None
    return w / s, v / s


def convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    pts = sorted(set(points))
    if len(pts) < 3:
        return pts

    def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def polygon_area(points: list[tuple[float, float]]) -> float:
    n = len(points)
    if n < 3:
        return 0.0
    return abs(
        sum(
            points[i][0] * points[(i + 1) % n][1] - points[(i + 1) % n][0] * points[i][1]
            for i in range(n)
        )
    ) / 2.0


def _court_point(row: dict[str, str]) -> tuple[float, float] | None:
    if row.get('is_smoothed_valid') in ('1', 'true', 'True'):
        cx = parse_float(row.get('smoothed_court_x'))
        cy = parse_float(row.get('smoothed_court_y'))
    else:
        cx = parse_float(row.get('court_x'))
        cy = parse_float(row.get('court_y'))
    if cx is None or cy is None:
        return None
    return cx, cy


def _image_point(row: dict[str, str]) -> tuple[float, float] | None:
    x = parse_float(row.get('smoothed_image_x') or row.get('image_x'))
    y = parse_float(row.get('smoothed_image_y') or row.get('image_y'))
    if x is None or y is None:
        return None
    return x, y


def _own_side_depth(player_id: str, court_y: float) -> float:
    depth = court_y - NET_Y_M if player_id == 'near' else NET_Y_M - court_y
    return max(0.0, min(COURT_LENGTH_M / 2.0, depth))


def _zone_for_depth(depth: float) -> str:
    if depth < FRONT_DEPTH_M:
        return 'front'
    if depth > BACK_DEPTH_M:
        return 'back'
    return 'mid'


def player_metrics(player_id: str, rows: list[dict[str, str]]) -> dict[str, object] | None:
    points: list[tuple[int, float, float, float]] = []
    for row in rows:
        court = _court_point(row)
        if court is None:
            continue
        try:
            frame_id = int(row['frame_id'])
        except (KeyError, ValueError):
            continue
        timestamp = parse_float(row.get('timestamp')) or 0.0
        points.append((frame_id, timestamp, court[0], court[1]))
    if not points:
        return None
    points.sort(key=lambda p: p[0])

    total_distance = 0.0
    for i in range(1, len(points)):
        total_distance += math.hypot(points[i][2] - points[i - 1][2], points[i][3] - points[i - 1][3])
    duration = points[-1][1] - points[0][1]
    avg_speed = total_distance / duration if duration > 0 else 0.0

    court_points = [(p[2], p[3]) for p in points]
    coverage_area = polygon_area(convex_hull(court_points))

    mean_x = sum(p[2] for p in points) / len(points)
    mean_y = sum(p[3] for p in points) / len(points)

    zones = {'front': 0, 'mid': 0, 'back': 0}
    for p in points:
        zones[_zone_for_depth(_own_side_depth(player_id, p[3]))] += 1
    count = len(points)

    return {
        'frames_valid': count,
        'total_distance_m': round(total_distance, 3),
        'avg_speed_m_s': round(avg_speed, 3),
        'coverage_area_m2': round(coverage_area, 3),
        'mean_court_x': round(mean_x, 3),
        'mean_court_y': round(mean_y, 3),
        'front_court_ratio': round(zones['front'] / count, 3),
        'mid_court_ratio': round(zones['mid'] / count, 3),
        'back_court_ratio': round(zones['back'] / count, 3),
    }


def shuttle_image_points(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    points: list[dict[str, object]] = []
    for row in rows:
        # Only real detections; the smoothed columns carry a stale value across
        # invisible stretches.
        if row.get('x') in (None, '') or row.get('y') in (None, ''):
            continue
        x = parse_float(row.get('smoothed_x') or row.get('x'))
        y = parse_float(row.get('smoothed_y') or row.get('y'))
        if x is None or y is None:
            continue
        try:
            frame_id = int(row['frame_id'])
        except (KeyError, ValueError):
            continue
        timestamp = parse_float(row.get('timestamp')) or 0.0
        points.append(
            {
                'frame_id': frame_id,
                'timestamp': timestamp,
                'image_x': x,
                'image_y': y,
            }
        )
    points.sort(key=lambda p: int(p['frame_id']))
    return points


def detect_strikes(
    image_points: list[dict[str, object]],
    turn_angle_deg: float,
    merge_frames: int,
) -> list[dict[str, object]]:
    """Detect shuttle direction reversals in image space.

    Each strike (and each bounce) sharply reverses the shuttle's direction of
    travel in the image. This is robust to the ground-plane projection
    distortion that breaks court-space analysis of an airborne shuttle.
    """
    if len(image_points) < 3:
        return []
    turns: list[dict[str, object]] = []
    for i in range(1, len(image_points) - 1):
        v1 = (
            float(image_points[i]['image_x']) - float(image_points[i - 1]['image_x']),
            float(image_points[i]['image_y']) - float(image_points[i - 1]['image_y']),
        )
        v2 = (
            float(image_points[i + 1]['image_x']) - float(image_points[i]['image_x']),
            float(image_points[i + 1]['image_y']) - float(image_points[i]['image_y']),
        )
        d1 = math.hypot(*v1)
        d2 = math.hypot(*v2)
        if d1 < 1e-6 or d2 < 1e-6:
            continue
        cos_angle = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (d1 * d2)))
        angle = math.degrees(math.acos(cos_angle))
        if angle > turn_angle_deg:
            turns.append(
                {
                    'frame_id': int(image_points[i]['frame_id']),
                    'timestamp': image_points[i]['timestamp'],
                    'angle': angle,
                    'image_x': float(image_points[i]['image_x']),
                    'image_y': float(image_points[i]['image_y']),
                }
            )
    # A single strike produces a couple of adjacent turning frames; merge them.
    events: list[dict[str, object]] = []
    for turn in sorted(turns, key=lambda t: int(t['frame_id'])):
        if events and int(turn['frame_id']) - int(events[-1]['frame_id']) <= merge_frames:
            if turn['angle'] > events[-1]['angle']:
                events[-1] = dict(turn)
        else:
            events.append(dict(turn))
    return events


def _interp(times: list[int], values: list[float], target: int) -> float | None:
    if not times:
        return None
    if target <= times[0]:
        return values[0]
    if target >= times[-1]:
        return values[-1]
    for i in range(len(times) - 1):
        if times[i] <= target <= times[i + 1]:
            span = times[i + 1] - times[i]
            if span <= 0:
                return values[i]
            frac = (target - times[i]) / span
            return values[i] + frac * (values[i + 1] - values[i])
    return None


def build_player_series(
    player_rows_by_id: dict[str, list[dict[str, str]]],
) -> dict[str, dict[str, object]]:
    series: dict[str, dict[str, object]] = {}
    for player_id, rows in player_rows_by_id.items():
        samples: list[tuple[int, float, float, float | None, float | None]] = []
        for row in rows:
            image = _image_point(row)
            if image is None:
                continue
            try:
                frame_id = int(row['frame_id'])
            except (KeyError, ValueError):
                continue
            court = _court_point(row)
            samples.append((frame_id, image[0], image[1], court[0] if court else None, court[1] if court else None))
        samples.sort(key=lambda s: s[0])
        series[player_id] = {
            'frames': [s[0] for s in samples],
            'image_x': [s[1] for s in samples],
            'image_y': [s[2] for s in samples],
            'court_x': [s[3] for s in samples],
            'court_y': [s[4] for s in samples],
        }
    return series


def _side_owner(court_y: float) -> str:
    return 'near' if court_y >= NET_Y_M else 'far'


def analyze_rally_events(
    video_path: str,
    video_stem: str,
    rally_id: str,
    player_series: dict[str, dict[str, object]],
    shuttle_rows: list[dict[str, str]],
    h: list[list[float]] | None,
    turn_angle_deg: float,
    merge_frames: int,
    hit_distance_px: float,
) -> tuple[list[dict[str, object]], dict[str, int], dict[str, int]]:
    events: list[dict[str, object]] = []
    hit_counts: dict[str, int] = defaultdict(int)
    landing_counts: dict[str, int] = defaultdict(int)

    if h is not None:
        strikes = detect_strikes(shuttle_image_points(shuttle_rows), turn_angle_deg, merge_frames)
        for strike in strikes:
            frame_id = int(strike['frame_id'])
            event_type = 'hit'
            court_x: float | None = None
            court_y: float | None = None

            shuttle_court = image_to_court(float(strike['image_x']), float(strike['image_y']), h)
            if shuttle_court is not None and (
                0.0 <= shuttle_court[0] <= COURT_WIDTH_M and 0.0 <= shuttle_court[1] <= COURT_LENGTH_M
            ):
                # The shuttle is near the ground: a bounce on the court. The
                # ground-plane projection is accurate here.
                event_type = 'landing'
                court_x, court_y = shuttle_court
            else:
                # A strike at racket height. Prefer the nearest player's court
                # position as the hit point when attribution is confident;
                # otherwise fall back to the shuttle's own (approximate)
                # projection, which for an airborne shuttle is distorted but
                # bounded.
                best = None
                for pid, series in player_series.items():
                    px = _interp(series['frames'], series['image_x'], frame_id)
                    py = _interp(series['frames'], series['image_y'], frame_id)
                    if px is None or py is None:
                        continue
                    dist = math.hypot(float(strike['image_x']) - px, float(strike['image_y']) - py)
                    if best is None or dist < best[0]:
                        best = (dist, pid)
                if best is not None and best[0] < hit_distance_px:
                    player_id = best[1]
                    cx = _interp(player_series[player_id]['frames'], player_series[player_id]['court_x'], frame_id)
                    cy = _interp(player_series[player_id]['frames'], player_series[player_id]['court_y'], frame_id)
                    if cx is not None and cy is not None:
                        court_x, court_y = cx, cy
                if court_x is None:
                    court_x, court_y = shuttle_court if shuttle_court is not None else (None, None)
                if court_x is None:
                    continue

            if shuttle_court is not None:
                player_id = _side_owner(shuttle_court[1])
            events.append(
                {
                    'frame_id': frame_id,
                    'timestamp': strike['timestamp'],
                    'event_type': event_type,
                    'player_id': player_id,
                    'court_x': round(court_x, 3),
                    'court_y': round(court_y, 3),
                }
            )
            if event_type == 'hit':
                hit_counts[player_id] += 1
            else:
                landing_counts[player_id] += 1

    events.sort(key=lambda e: (int(e['frame_id']), e['event_type']))
    for event in events:
        event['video_path'] = video_path
        event['video_stem'] = video_stem
        event['rally_id'] = rally_id
    return events, dict(hit_counts), dict(landing_counts)


def analyze_tactics(
    player_tracks_csv: Path,
    shuttle_tracks_csv: Path,
    calibration_dir: Path,
    output_dir: Path,
    hit_distance_px: float = 80.0,
    turn_angle_deg: float = 100.0,
    min_event_gap_frames: int = 15,
) -> dict[str, object]:
    output_dir = ensure_dir(output_dir)

    player_rally_groups: dict[tuple[str, str, str], dict[str, list[dict[str, str]]]] = {}
    for row in read_csv_rows(player_tracks_csv):
        key = (row.get('video_path', ''), row.get('video_stem', ''), row.get('rally_id', ''))
        player_rally_groups.setdefault(key, {}).setdefault(row.get('player_id', ''), []).append(row)

    shuttle_rally_groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in read_csv_rows(shuttle_tracks_csv):
        key = (row.get('video_path', ''), row.get('video_stem', ''), row.get('rally_id', ''))
        shuttle_rally_groups.setdefault(key, []).append(row)

    rally_keys = sorted(set(player_rally_groups) | set(shuttle_rally_groups))
    summary_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []

    for key in rally_keys:
        video_path, video_stem, rally_id = key
        players = player_rally_groups.get(key, {})
        h = load_homography(calibration_dir, video_stem)
        player_series = build_player_series(players)

        events, hit_counts, landing_counts = analyze_rally_events(
            video_path,
            video_stem,
            rally_id,
            player_series,
            shuttle_rally_groups.get(key, []),
            h,
            turn_angle_deg,
            min_event_gap_frames,
            hit_distance_px,
        )
        event_rows.extend(events)

        for player_id, rows in players.items():
            metrics = player_metrics(player_id, rows)
            if metrics is None:
                continue
            summary_rows.append(
                {
                    'video_path': video_path,
                    'video_stem': video_stem,
                    'rally_id': rally_id,
                    'player_id': player_id,
                    **metrics,
                    'hit_count': hit_counts.get(player_id, 0),
                    'landing_count': landing_counts.get(player_id, 0),
                }
            )

    summary_csv = output_dir / 'tactics_summary.csv'
    events_csv = output_dir / 'tactics_events.csv'
    write_csv_rows(summary_csv, TACTICS_SUMMARY_FIELDS, summary_rows)
    write_csv_rows(events_csv, TACTICS_EVENT_FIELDS, event_rows)
    return {
        'summary_rows': len(summary_rows),
        'event_rows': len(event_rows),
        'summary_csv': str(summary_csv),
        'events_csv': str(events_csv),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Compute tactical metrics (distance/coverage/position) and shuttle hit/landing events.'
    )
    parser.add_argument('player_tracks_csv', type=Path, help='Player tracks CSV (smoothed preferred).')
    parser.add_argument('shuttle_tracks_csv', type=Path, help='Shuttle tracks CSV (smoothed preferred).')
    parser.add_argument('--calibration-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/tactics'))
    parser.add_argument('--hit-distance-px', type=float, default=80.0)
    parser.add_argument('--turn-angle-deg', type=float, default=100.0)
    parser.add_argument('--min-event-gap-frames', type=int, default=15)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = analyze_tactics(
        args.player_tracks_csv,
        args.shuttle_tracks_csv,
        args.calibration_dir,
        args.output_dir,
        hit_distance_px=args.hit_distance_px,
        turn_angle_deg=args.turn_angle_deg,
        min_event_gap_frames=args.min_event_gap_frames,
    )
    print(f"Wrote {result['summary_rows']} summary rows -> {result['summary_csv']}")
    print(f"Wrote {result['event_rows']} event rows -> {result['events_csv']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

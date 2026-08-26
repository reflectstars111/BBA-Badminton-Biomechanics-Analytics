from __future__ import annotations

import argparse
from pathlib import Path

from common import read_csv_rows, write_csv_rows
from video_preprocess import probe_video

MATCH_FIELDS = [
    'match_id',
    'source',
    'url',
    'tournament',
    'year',
    'discipline',
    'round',
    'player_1',
    'player_2',
    'resolution',
    'fps',
    'camera_type',
    'usable_rallies',
    'notes',
]


def register_local_match(
    video: Path,
    match_id: str,
    tournament: str,
    year: str,
    discipline: str,
    round_: str,
    player_1: str,
    player_2: str,
    camera_type: str,
    csv_path: Path,
    notes: str = '',
) -> dict[str, object]:
    """Probe a local rally/match video and append it to matches.csv."""
    info = probe_video(video, sample_every=1)
    if not info.get('ok'):
        raise SystemExit(f'Cannot read video: {video}')

    rows = read_csv_rows(csv_path)
    if any(row.get('match_id') == match_id for row in rows):
        raise SystemExit(f'match_id already exists in {csv_path}: {match_id}')

    fps = info.get('fps')
    new_row = {
        'match_id': match_id,
        'source': 'local',
        'url': str(video),
        'tournament': tournament,
        'year': str(year),
        'discipline': discipline,
        'round': round_,
        'player_1': player_1,
        'player_2': player_2,
        'resolution': f"{info['width']}x{info['height']}",
        'fps': str(int(round(fps))) if fps else 'unknown',
        'camera_type': camera_type,
        'usable_rallies': '',
        'notes': notes,
    }
    fields = MATCH_FIELDS if rows and list(rows[0].keys()) == MATCH_FIELDS else list(rows[0].keys()) if rows else MATCH_FIELDS
    rows.append(new_row)
    write_csv_rows(csv_path, fields, rows)
    return {
        'match_id': match_id,
        'rows': len(rows),
        'resolution': new_row['resolution'],
        'fps': new_row['fps'],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Probe a local match video and register it in metadata/matches.csv.'
    )
    parser.add_argument('video', type=Path, help='Local match video (mp4).')
    parser.add_argument('--match-id', required=True)
    parser.add_argument('--tournament', required=True)
    parser.add_argument('--year', required=True)
    parser.add_argument('--discipline', required=True, help='MS, WS, MD, WD, XD or multi.')
    parser.add_argument('--round', required=True, help='Final, Semi-final, group stage, etc.')
    parser.add_argument('--player-1', required=True)
    parser.add_argument('--player-2', required=True)
    parser.add_argument('--camera-type', default='broadcast')
    parser.add_argument('--notes', default='')
    parser.add_argument('--csv', type=Path, default=Path('metadata/matches.csv'))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = register_local_match(
        video=args.video,
        match_id=args.match_id,
        tournament=args.tournament,
        year=args.year,
        discipline=args.discipline,
        round_=args.round,
        player_1=args.player_1,
        player_2=args.player_2,
        camera_type=args.camera_type,
        csv_path=args.csv,
        notes=args.notes,
    )
    print(
        f"Registered {result['match_id']} ({result['resolution']}@{result['fps']}fps); "
        f"matches.csv now has {result['rows']} rows"
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

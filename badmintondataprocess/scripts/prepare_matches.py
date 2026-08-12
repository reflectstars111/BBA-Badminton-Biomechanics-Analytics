from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from common import read_csv_rows


REQUIRED_FIELDS = {
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
}

UNKNOWN_VALUES = {'unknown', 'n/a', 'na'}


def validate_matches(csv_path: Path) -> tuple[int, list[dict[str, str]], list[str]]:
    rows = read_csv_rows(csv_path)
    if not rows:
        print('No rows found in matches.csv')
        return 1, [], ['No rows found in matches.csv']

    errors: list[str] = []
    missing_count = 0
    duplicated_match_ids: set[str] = set()
    seen_match_ids: set[str] = set()

    for idx, row in enumerate(rows, start=1):
        missing = [key for key in REQUIRED_FIELDS if not row.get(key)]
        if missing:
            missing_count += 1
            errors.append(f'Row {idx} missing fields: {", ".join(missing)}')

        match_id = (row.get('match_id') or '').strip()
        if match_id:
            if match_id in seen_match_ids:
                duplicated_match_ids.add(match_id)
            seen_match_ids.add(match_id)

        year = (row.get('year') or '').strip()
        if year and not year.isdigit():
            errors.append(f'Row {idx} invalid year: {year}')

        fps = (row.get('fps') or '').strip()
        if fps:
            if fps.lower() in UNKNOWN_VALUES:
                continue
            try:
                fps_value = float(fps)
                if fps_value <= 0:
                    errors.append(f'Row {idx} fps must be > 0: {fps}')
            except ValueError:
                errors.append(f'Row {idx} invalid fps: {fps}')

    if duplicated_match_ids:
        errors.append(
            'Duplicated match_id values: ' + ', '.join(sorted(duplicated_match_ids))
        )

    print(f'Total rows: {len(rows)}')
    print(f'Rows with missing required fields: {missing_count}')
    if errors:
        print(f'Validation errors: {len(errors)}')
        for item in errors:
            print(f'- {item}')
    else:
        print('Validation passed.')

    return (0 if not errors else 2), rows, errors


def print_summary(rows: list[dict[str, str]]) -> None:
    if not rows:
        return

    by_source: dict[str, int] = {}
    by_tournament: dict[str, int] = {}
    fps_values: list[float] = []

    for row in rows:
        source = (row.get('source') or 'UNKNOWN').strip() or 'UNKNOWN'
        tournament = (row.get('tournament') or 'UNKNOWN').strip() or 'UNKNOWN'
        by_source[source] = by_source.get(source, 0) + 1
        by_tournament[tournament] = by_tournament.get(tournament, 0) + 1

        fps = (row.get('fps') or '').strip()
        if fps:
            try:
                fps_values.append(float(fps))
            except ValueError:
                pass

    print('\nSummary by source:')
    for key in sorted(by_source):
        print(f'- {key}: {by_source[key]}')

    print('\nSummary by tournament:')
    for key in sorted(by_tournament):
        print(f'- {key}: {by_tournament[key]}')

    if fps_values:
        print('\nFPS statistics:')
        print(f'- min: {min(fps_values):.2f}')
        print(f'- max: {max(fps_values):.2f}')
        print(f'- mean: {statistics.mean(fps_values):.2f}')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Validate match metadata CSV.')
    parser.add_argument(
        '--csv',
        type=Path,
        default=Path('metadata/matches.csv'),
        help='Path to the matches CSV file.',
    )
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Print basic dataset summary after validation.',
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    exit_code, rows, _ = validate_matches(args.csv)
    if args.summary:
        print_summary(rows)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())

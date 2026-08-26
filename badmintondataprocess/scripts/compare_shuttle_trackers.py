from __future__ import annotations

import argparse
from pathlib import Path

from common import ensure_dir, read_csv_rows, write_csv_rows

COMPARISON_FIELDS = [
    'rally_id',
    'video_stem',
    'frames',
]


def _num(value: str | None) -> float:
    try:
        return float(value or 0)
    except ValueError:
        return 0.0


def load_summary(path: Path) -> dict[str, dict[str, float]]:
    by_stem: dict[str, dict[str, float]] = {}
    for row in read_csv_rows(path):
        by_stem[row['video_stem']] = {
            'track_rows': _num(row.get('track_rows')),
            'visible_rows': _num(row.get('visible_rows')),
            'interpolated_rows': _num(row.get('interpolated_rows')),
        }
    return by_stem


def compare_trackers(
    test_set: Path | None,
    named_summaries: list[tuple[str, Path]],
    output_dir: Path,
) -> dict[str, object]:
    output_dir = ensure_dir(output_dir)

    order: list[tuple[str, str]] = []
    if test_set is not None:
        for row in read_csv_rows(test_set):
            order.append((row['video_stem'], row.get('rally_id', '')))

    summaries = [(label, load_summary(path)) for label, path in named_summaries]
    stems = [s for s, _ in order] if order else sorted(summaries[0][1])

    fieldnames = list(COMPARISON_FIELDS)
    for label, _ in summaries:
        fieldnames += [
            f'{label}_visible_ratio',
            f'{label}_interp_ratio',
            f'{label}_visible_rows',
            f'{label}_interp_rows',
        ]

    rows: list[dict[str, object]] = []
    for stem in stems:
        base = summaries[0][1].get(stem)
        if base is None:
            continue
        row: dict[str, object] = {
            'rally_id': next((r for s, r in order if s == stem), ''),
            'video_stem': stem,
            'frames': int(base['track_rows']),
        }
        for label, data in summaries:
            stats = data.get(stem)
            if stats is None:
                row[f'{label}_visible_ratio'] = ''
                row[f'{label}_interp_ratio'] = ''
                row[f'{label}_visible_rows'] = ''
                row[f'{label}_interp_rows'] = ''
                continue
            total = stats['track_rows'] or 1.0
            row[f'{label}_visible_ratio'] = round(stats['visible_rows'] / total, 3)
            row[f'{label}_interp_ratio'] = round(stats['interpolated_rows'] / total, 3)
            row[f'{label}_visible_rows'] = int(stats['visible_rows'])
            row[f'{label}_interp_rows'] = int(stats['interpolated_rows'])
        rows.append(row)

    summary_row: dict[str, object] = {'rally_id': 'MEAN', 'video_stem': '', 'frames': ''}
    for label, data in summaries:
        vals = [r[f'{label}_visible_ratio'] for r in rows if r.get(f'{label}_visible_ratio') != '']
        ivals = [r[f'{label}_interp_ratio'] for r in rows if r.get(f'{label}_interp_ratio') != '']
        summary_row[f'{label}_visible_ratio'] = round(sum(vals) / len(vals), 3) if vals else ''
        summary_row[f'{label}_interp_ratio'] = round(sum(ivals) / len(ivals), 3) if ivals else ''
        summary_row[f'{label}_visible_rows'] = ''
        summary_row[f'{label}_interp_rows'] = ''
    rows.append(summary_row)

    table_csv = output_dir / 'shuttle_tracker_comparison.csv'
    write_csv_rows(table_csv, fieldnames, rows)

    # Markdown table for quick viewing.
    md_lines = ['| ' + ' | '.join(fieldnames) + ' |', '|' + '|'.join(['---'] * len(fieldnames)) + '|']
    for row in rows:
        md_lines.append('| ' + ' | '.join(str(row.get(col, '')) for col in fieldnames) + ' |')
    md_text = '\n'.join(md_lines)
    (output_dir / 'shuttle_tracker_comparison.md').write_text(md_text, encoding='utf-8')
    print(md_text)
    return {'rows': len(rows) - 1, 'trackers': [label for label, _ in summaries], 'table_csv': str(table_csv)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Compare shuttle trackers (baseline vs TrackNet ablation) per rally.'
    )
    parser.add_argument(
        '--test-set',
        type=Path,
        default=None,
        help='Optional test set CSV (video_stem, rally_id) to order and label rows.',
    )
    parser.add_argument(
        '--compare',
        nargs='+',
        required=True,
        metavar='LABEL=PATH',
        help='One or more named shuttle tracking summary CSVs, e.g. baseline=... tracknet=th0.15=...',
    )
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/tracker_comparison'))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    pairs: list[tuple[str, Path]] = []
    for item in args.compare:
        label, _, path = item.partition('=')
        if not path:
            raise SystemExit(f'Expected LABEL=PATH, got: {item}')
        pairs.append((label, Path(path)))
    compare_trackers(args.test_set, pairs, args.output_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

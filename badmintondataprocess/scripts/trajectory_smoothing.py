from __future__ import annotations

import argparse
import math
from pathlib import Path

from common import read_csv_rows, write_csv_rows


def parse_float(value: str | None) -> float | None:
    if value is None or value == '':
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_int(value: str | None, default: int = 0) -> int:
    if value is None or value == '':
        return default
    try:
        return int(float(value))
    except ValueError:
        return default


def infer_schema(fieldnames: list[str]) -> tuple[str, list[str], list[str]]:
    if {'x', 'y', 'visibility'}.issubset(fieldnames):
        return 'shuttle', ['video_path', 'video_stem', 'rally_id'], ['x', 'y']
    if {'player_id', 'image_x', 'image_y'}.issubset(fieldnames):
        coordinates = ['image_x', 'image_y', 'court_x', 'court_y']
        for explicit_pair in (
            ['body_image_x', 'body_image_y'],
            ['ground_image_x', 'ground_image_y'],
        ):
            if set(explicit_pair).issubset(fieldnames):
                coordinates.extend(explicit_pair)
        return (
            'player',
            ['video_path', 'video_stem', 'rally_id', 'player_id'],
            coordinates,
        )
    raise RuntimeError(f'Unsupported trajectory CSV columns: {fieldnames}')


def group_rows(
    rows: list[dict[str, str]],
    group_keys: list[str],
) -> list[tuple[tuple[str, ...], list[dict[str, str]]]]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = {}
    order: list[tuple[str, ...]] = []
    for row in rows:
        key = tuple(row.get(name, '') for name in group_keys)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(row)
    result: list[tuple[tuple[str, ...], list[dict[str, str]]]] = []
    for key in order:
        items = grouped[key]
        try:
            items.sort(key=lambda row: int(row['frame_id']))
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f'Trajectory group {key!r} has an invalid frame_id') from exc
        frame_ids = [int(row['frame_id']) for row in items]
        if len(frame_ids) != len(set(frame_ids)):
            raise RuntimeError(f'Trajectory group {key!r} has duplicate frame_id values')
        result.append((key, items))
    return result


def row_is_valid(row: dict[str, str], coordinate_columns: list[str], min_confidence: float) -> bool:
    if any(parse_float(row.get(column)) is None for column in coordinate_columns):
        return False
    confidence = parse_float(row.get('confidence'))
    if confidence is not None and confidence < min_confidence:
        return False
    if 'visibility' in row and parse_int(row.get('visibility')) == 0:
        return False
    return True


def interpolate_series(
    values: list[float | None],
    valid_mask: list[bool],
    max_gap_frames: int,
    frame_ids: list[int] | None = None,
    blocked_indices: set[int] | None = None,
) -> tuple[list[float | None], list[bool]]:
    if len(values) != len(valid_mask):
        raise ValueError('values and valid_mask must have the same length')
    positions = frame_ids if frame_ids is not None else list(range(len(values)))
    if len(positions) != len(values):
        raise ValueError('frame_ids and values must have the same length')

    filled = list(values)
    gap_filled = [False] * len(values)
    blocked = blocked_indices or set()
    valid_indices = [index for index, is_valid in enumerate(valid_mask) if is_valid and values[index] is not None]
    if len(valid_indices) < 2:
        return filled, gap_filled

    for start_index, end_index in zip(valid_indices[:-1], valid_indices[1:], strict=True):
        start_frame = positions[start_index]
        end_frame = positions[end_index]
        gap = end_frame - start_frame - 1
        if gap <= 0 or gap > max_gap_frames:
            continue
        if any(index in blocked for index in range(start_index + 1, end_index)):
            continue
        start_value = values[start_index]
        end_value = values[end_index]
        if start_value is None or end_value is None:
            continue
        frame_span = end_frame - start_frame
        for insert_index in range(start_index + 1, end_index):
            insert_frame = positions[insert_index]
            if insert_frame <= start_frame or insert_frame >= end_frame:
                continue
            if filled[insert_index] is None:
                ratio = (insert_frame - start_frame) / float(frame_span)
                filled[insert_index] = start_value + (end_value - start_value) * ratio
                gap_filled[insert_index] = True
    return filled, gap_filled


def large_displacement_gap_indices(
    per_column_values: dict[str, list[float | None]],
    valid_mask: list[bool],
    frame_ids: list[int],
    max_gap_frames: int,
    max_displacement_px: float | None,
) -> set[int]:
    """Return missing row indices whose 2-D interpolation would make a large jump."""
    if max_displacement_px is None:
        return set()
    x_values = per_column_values.get('x')
    y_values = per_column_values.get('y')
    if x_values is None or y_values is None:
        return set()

    valid_indices = [
        index
        for index, is_valid in enumerate(valid_mask)
        if is_valid and x_values[index] is not None and y_values[index] is not None
    ]
    blocked: set[int] = set()
    for start_index, end_index in zip(valid_indices[:-1], valid_indices[1:], strict=True):
        gap = frame_ids[end_index] - frame_ids[start_index] - 1
        if gap <= 0 or gap > max_gap_frames:
            continue
        displacement = math.hypot(
            x_values[end_index] - x_values[start_index],
            y_values[end_index] - y_values[start_index],
        )
        if displacement > max_displacement_px:
            blocked.update(range(start_index + 1, end_index))
    return blocked


def rolling_median(values: list[float | None], window_size: int) -> list[float | None]:
    if window_size <= 1:
        return list(values)
    radius = window_size // 2
    smoothed: list[float | None] = []
    for index in range(len(values)):
        if values[index] is None:
            smoothed.append(None)
            continue
        lower_bound = max(0, index - radius)
        upper_bound = min(len(values), index + radius + 1)
        start = index
        while start > lower_bound and values[start - 1] is not None:
            start -= 1
        end = index + 1
        while end < upper_bound and values[end] is not None:
            end += 1
        window = sorted(value for value in values[start:end] if value is not None)
        if not window:
            smoothed.append(None)
            continue
        mid = len(window) // 2
        if len(window) % 2 == 1:
            smoothed.append(window[mid])
        else:
            smoothed.append((window[mid - 1] + window[mid]) / 2.0)
    return smoothed


def ema_smooth(values: list[float | None], alpha: float) -> list[float | None]:
    if alpha <= 0.0:
        return list(values)
    smoothed: list[float | None] = []
    previous: float | None = None
    for value in values:
        if value is None:
            smoothed.append(None)
            previous = None
            continue
        if previous is None:
            previous = value
        else:
            previous = alpha * value + (1.0 - alpha) * previous
        smoothed.append(previous)
    return smoothed


def smooth_frame_segments(
    values: list[float | None],
    frame_ids: list[int],
    window_size: int,
    ema_alpha: float,
) -> list[float | None]:
    """Smooth contiguous frame runs without carrying state across omitted frames."""
    if len(values) != len(frame_ids):
        raise ValueError('frame_ids and values must have the same length')
    if not values:
        return []

    result: list[float | None] = [None] * len(values)
    start = 0
    for index in range(1, len(values) + 1):
        at_end = index == len(values)
        frame_break = not at_end and frame_ids[index] != frame_ids[index - 1] + 1
        if not at_end and not frame_break:
            continue
        segment = values[start:index]
        median_values = rolling_median(segment, window_size)
        result[start:index] = ema_smooth(median_values, ema_alpha)
        start = index
    return result


def smooth_coordinate_columns(
    rows: list[dict[str, str]],
    coordinate_columns: list[str],
    min_confidence: float,
    max_gap_frames: int,
    window_size: int,
    ema_alpha: float,
    max_interpolation_displacement_px: float | None = None,
) -> tuple[dict[str, list[float | None]], list[bool], list[bool]]:
    frame_ids = [int(row['frame_id']) for row in rows]
    valid_mask = [row_is_valid(row, coordinate_columns, min_confidence) for row in rows]
    per_column_values: dict[str, list[float | None]] = {
        column: [parse_float(row.get(column)) if is_valid else None for row, is_valid in zip(rows, valid_mask, strict=True)]
        for column in coordinate_columns
    }
    blocked_indices = large_displacement_gap_indices(
        per_column_values,
        valid_mask,
        frame_ids,
        max_gap_frames,
        max_interpolation_displacement_px,
    )

    per_column_smoothed: dict[str, list[float | None]] = {}
    combined_gap_filled = [False] * len(rows)
    for column, values in per_column_values.items():
        interpolated, gap_filled = interpolate_series(
            values,
            valid_mask,
            max_gap_frames,
            frame_ids=frame_ids,
            blocked_indices=blocked_indices,
        )
        approved_values = [
            value if source_valid or was_gap_filled else None
            for value, source_valid, was_gap_filled in zip(
                interpolated,
                valid_mask,
                gap_filled,
                strict=True,
            )
        ]
        smoothed = smooth_frame_segments(
            approved_values,
            frame_ids,
            window_size,
            ema_alpha,
        )
        per_column_smoothed[column] = smoothed
        combined_gap_filled = [
            existing or new
            for existing, new in zip(combined_gap_filled, gap_filled, strict=True)
        ]

    smoothed_valid = [
        source_valid or was_gap_filled
        for source_valid, was_gap_filled in zip(valid_mask, combined_gap_filled, strict=True)
    ]
    return per_column_smoothed, combined_gap_filled, smoothed_valid


def round_or_blank(value: float | None, digits: int = 3) -> str:
    if value is None or math.isnan(value):
        return ''
    return str(round(value, digits))


def smooth_trajectory(
    input_csv: Path,
    output_csv: Path,
    summary_csv: Path | None,
    min_confidence: float,
    max_gap_frames: int,
    window_size: int,
    ema_alpha: float,
    max_interpolation_displacement_px: float | None = None,
) -> None:
    rows = read_csv_rows(input_csv)
    if not rows:
        raise RuntimeError(f'Input CSV has no rows: {input_csv}')

    fieldnames = list(rows[0].keys())
    schema_name, group_keys, coordinate_columns = infer_schema(fieldnames)
    grouped_rows = group_rows(rows, group_keys)
    extra_columns = [f'smoothed_{column}' for column in coordinate_columns]
    extra_columns.extend(['is_gap_filled', 'is_smoothed_valid'])
    output_fieldnames = fieldnames + [column for column in extra_columns if column not in fieldnames]

    smoothed_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for key, items in grouped_rows:
        per_column_smoothed, gap_filled_mask, smoothed_valid_mask = smooth_coordinate_columns(
            rows=items,
            coordinate_columns=coordinate_columns,
            min_confidence=min_confidence,
            max_gap_frames=max_gap_frames,
            window_size=window_size,
            ema_alpha=ema_alpha,
            max_interpolation_displacement_px=(
                max_interpolation_displacement_px if schema_name == 'shuttle' else None
            ),
        )

        source_valid_rows = 0
        gap_filled_rows = 0
        smoothed_valid_rows = 0
        for index, row in enumerate(items):
            result_row: dict[str, object] = dict(row)
            is_source_valid = row_is_valid(row, coordinate_columns, min_confidence)
            source_valid_rows += int(is_source_valid)
            gap_filled_rows += int(gap_filled_mask[index])
            smoothed_valid_rows += int(smoothed_valid_mask[index])
            for column in coordinate_columns:
                result_row[f'smoothed_{column}'] = round_or_blank(per_column_smoothed[column][index], digits=3)
            result_row['is_gap_filled'] = int(gap_filled_mask[index])
            result_row['is_smoothed_valid'] = int(smoothed_valid_mask[index])
            smoothed_rows.append(result_row)

        summary_row = {name: value for name, value in zip(group_keys, key, strict=True)}
        summary_row.update(
            {
                'schema': schema_name,
                'rows': len(items),
                'source_valid_rows': source_valid_rows,
                'smoothed_valid_rows': smoothed_valid_rows,
                'gap_filled_rows': gap_filled_rows,
                'min_confidence': round(min_confidence, 3),
                'max_gap_frames': max_gap_frames,
                'max_interpolation_displacement_px': (
                    round(max_interpolation_displacement_px, 3)
                    if schema_name == 'shuttle' and max_interpolation_displacement_px is not None
                    else ''
                ),
                'window_size': window_size,
                'ema_alpha': round(ema_alpha, 3),
            }
        )
        summary_rows.append(summary_row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv_rows(output_csv, output_fieldnames, smoothed_rows)

    if summary_csv is not None:
        summary_fieldnames = group_keys + [
            'schema',
            'rows',
            'source_valid_rows',
            'smoothed_valid_rows',
            'gap_filled_rows',
            'min_confidence',
            'max_gap_frames',
            'max_interpolation_displacement_px',
            'window_size',
            'ema_alpha',
        ]
        write_csv_rows(summary_csv, summary_fieldnames, summary_rows)
    print(f'Smoothed trajectory saved to: {output_csv}')
    if summary_csv is not None:
        print(f'Smoothing summary saved to: {summary_csv}')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Smooth and interpolate trajectory CSV data.')
    parser.add_argument('input', type=Path, help='Input trajectory CSV file.')
    parser.add_argument('output', type=Path, help='Output smoothed CSV file.')
    parser.add_argument(
        '--summary-csv',
        type=Path,
        default=None,
        help='Optional per-group smoothing summary CSV.',
    )
    parser.add_argument(
        '--min-confidence',
        type=float,
        default=0.2,
        help='Minimum confidence to treat a point as a source observation.',
    )
    parser.add_argument(
        '--max-gap-frames',
        type=int,
        default=4,
        help='Maximum gap size to fill with linear interpolation.',
    )
    parser.add_argument(
        '--window-size',
        type=int,
        default=5,
        help='Rolling median window size.',
    )
    parser.add_argument(
        '--max-interpolation-displacement-px',
        type=float,
        default=80.0,
        help='Maximum 2-D endpoint displacement allowed when filling shuttle gaps.',
    )
    parser.add_argument(
        '--ema-alpha',
        type=float,
        default=0.35,
        help='EMA alpha used after median smoothing.',
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    smooth_trajectory(
        input_csv=args.input,
        output_csv=args.output,
        summary_csv=args.summary_csv,
        min_confidence=args.min_confidence,
        max_gap_frames=args.max_gap_frames,
        window_size=args.window_size,
        ema_alpha=args.ema_alpha,
        max_interpolation_displacement_px=args.max_interpolation_displacement_px,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

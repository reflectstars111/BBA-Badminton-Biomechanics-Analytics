from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from common import ensure_dir, read_csv_rows

COURT_WIDTH = 6.10
COURT_LENGTH = 13.40
NET_Y = COURT_LENGTH / 2.0
SHORT_SERVICE_LINE = 1.98
LONG_SERVICE_FROM_BACK = 0.76
CENTER_X = COURT_WIDTH / 2.0


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


def rally_label(row: dict[str, str]) -> str:
    if row.get('rally_id'):
        return f"r{row['rally_id']}"
    stem = row.get('video_stem', '')
    match = re.search(r'rally_(\d+)', stem)
    if match:
        return f"r{match.group(1)}"
    return stem[-12:] if stem else 'unknown'


def save_bar_chart(
    labels: list[str],
    values_a: list[float],
    values_b: list[float] | None,
    title: str,
    ylabel: str,
    legend_a: str,
    legend_b: str | None,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    x_positions = list(range(len(labels)))
    if values_b is None:
        ax.bar(x_positions, values_a, color='#3b82f6')
    else:
        ax.bar([x - 0.2 for x in x_positions], values_a, width=0.4, color='#22c55e', label=legend_a)
        ax.bar([x + 0.2 for x in x_positions], values_b, width=0.4, color='#f59e0b', label=legend_b)
        ax.legend()
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_ratio_chart(summary_rows: list[dict[str, str]], output_path: Path) -> None:
    ordered = []
    for row in summary_rows:
        track_rows = parse_int(row.get('track_rows'))
        visible_rows = parse_int(row.get('visible_rows'))
        ratio = visible_rows / track_rows if track_rows else 0.0
        ordered.append((rally_label(row), ratio))
    ordered.sort(key=lambda item: item[1], reverse=True)

    fig, ax = plt.subplots(figsize=(14, 6))
    labels = [item[0] for item in ordered]
    values = [item[1] for item in ordered]
    ax.bar(range(len(labels)), values, color='#8b5cf6')
    ax.set_title('Shuttle Visible Ratio by Rally')
    ax.set_ylabel('visible_rows / track_rows')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylim(0.0, max(0.05, max(values) * 1.15 if values else 0.05))
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def group_rows_by_video(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row.get('video_stem', ''), []).append(row)
    return grouped


def pick_coordinate(row: dict[str, str], preferred_x: str, preferred_y: str) -> tuple[float | None, float | None]:
    x_value = parse_float(row.get(preferred_x))
    y_value = parse_float(row.get(preferred_y))
    if x_value is not None and y_value is not None:
        return x_value, y_value
    return parse_float(row.get('x')), parse_float(row.get('y'))


def draw_full_court(ax: plt.Axes) -> None:
    ax.add_patch(
        Rectangle((0, 0), COURT_WIDTH, COURT_LENGTH, fill=False, linewidth=2.4, edgecolor='white')
    )
    ax.plot([0, COURT_WIDTH], [NET_Y, NET_Y], color='white', linewidth=2.0)
    ax.plot([0, COURT_WIDTH], [SHORT_SERVICE_LINE, SHORT_SERVICE_LINE], color='white', linewidth=1.2)
    ax.plot(
        [0, COURT_WIDTH],
        [COURT_LENGTH - SHORT_SERVICE_LINE, COURT_LENGTH - SHORT_SERVICE_LINE],
        color='white',
        linewidth=1.2,
    )
    ax.plot([0, COURT_WIDTH], [LONG_SERVICE_FROM_BACK, LONG_SERVICE_FROM_BACK], color='white', linewidth=1.0)
    ax.plot(
        [0, COURT_WIDTH],
        [COURT_LENGTH - LONG_SERVICE_FROM_BACK, COURT_LENGTH - LONG_SERVICE_FROM_BACK],
        color='white',
        linewidth=1.0,
    )
    ax.plot([CENTER_X, CENTER_X], [0, COURT_LENGTH], color='white', linewidth=1.0)


def draw_half_court(ax: plt.Axes) -> None:
    ax.add_patch(
        Rectangle((0, NET_Y), COURT_WIDTH, COURT_LENGTH - NET_Y, fill=False, linewidth=2.4, edgecolor='white')
    )
    ax.plot([0, COURT_WIDTH], [NET_Y, NET_Y], color='white', linewidth=2.0)
    ax.plot(
        [0, COURT_WIDTH],
        [COURT_LENGTH - SHORT_SERVICE_LINE, COURT_LENGTH - SHORT_SERVICE_LINE],
        color='white',
        linewidth=1.2,
    )
    ax.plot(
        [0, COURT_WIDTH],
        [COURT_LENGTH - LONG_SERVICE_FROM_BACK, COURT_LENGTH - LONG_SERVICE_FROM_BACK],
        color='white',
        linewidth=1.0,
    )
    ax.plot([CENTER_X, CENTER_X], [NET_Y, COURT_LENGTH], color='white', linewidth=1.0)


def style_court_axes(ax: plt.Axes, full_court: bool, title: str) -> None:
    ax.set_facecolor('#14532d')
    ax.set_xlim(-0.2, COURT_WIDTH + 0.2)
    if full_court:
        ax.set_ylim(COURT_LENGTH + 0.2, -0.2)
        ax.set_ylabel('Court Length (m)', color='white')
    else:
        ax.set_ylim(COURT_LENGTH + 0.15, NET_Y - 0.15)
        ax.set_ylabel('Half-Court Length (m)', color='white')
    ax.set_aspect('equal')
    ax.set_title(title, color='white', fontsize=14, pad=12)
    ax.set_xlabel('Court Width (m)', color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_visible(False)


def collect_player_points(
    player_track_rows: list[dict[str, str]],
    min_confidence: float,
) -> tuple[list[dict[str, float | str]], dict[str, list[dict[str, float | str]]]]:
    valid_rows: list[dict[str, float | str]] = []
    grouped_by_rally: dict[str, list[dict[str, float | str]]] = {}
    for row in player_track_rows:
        player_id = row.get('player_id', '')
        if player_id not in {'near', 'far'}:
            continue
        if parse_int(row.get('is_smoothed_valid')) != 1:
            continue
        confidence = parse_float(row.get('confidence'))
        if confidence is None or confidence < min_confidence:
            continue
        x_value = parse_float(row.get('smoothed_court_x')) or parse_float(row.get('court_x'))
        y_value = parse_float(row.get('smoothed_court_y')) or parse_float(row.get('court_y'))
        if x_value is None or y_value is None:
            continue
        if not (0.0 <= x_value <= COURT_WIDTH and 0.0 <= y_value <= COURT_LENGTH):
            continue
        point = {
            'player_id': player_id,
            'rally_id': row.get('rally_id', ''),
            'x': x_value,
            'y': y_value,
            'confidence': confidence,
        }
        valid_rows.append(point)
        grouped_by_rally.setdefault(str(point['rally_id']), []).append(point)
    return valid_rows, grouped_by_rally


def collect_player_counts_by_rally(
    player_smoothing_rows: list[dict[str, str]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in player_smoothing_rows:
        rally_id = row.get('rally_id', '')
        counts.setdefault(rally_id, 0)
        counts[rally_id] += parse_int(row.get('smoothed_valid_rows'))
    return counts


def save_player_track_rows_chart(
    player_summary_rows: list[dict[str, str]],
    output_path: Path,
) -> None:
    player_labels = [rally_label(row) for row in player_summary_rows]
    player_track_rows = [parse_int(row.get('track_rows')) for row in player_summary_rows]
    save_bar_chart(
        labels=player_labels,
        values_a=player_track_rows,
        values_b=None,
        title='Player Track Rows by Rally',
        ylabel='rows',
        legend_a='track_rows',
        legend_b=None,
        output_path=output_path,
    )


def save_player_smoothing_gain_chart(
    player_smoothing_rows: list[dict[str, str]],
    output_path: Path,
) -> None:
    grouped_source: dict[str, int] = {}
    grouped_smoothed: dict[str, int] = {}
    for row in player_smoothing_rows:
        rally_id = row.get('rally_id', '')
        grouped_source[rally_id] = grouped_source.get(rally_id, 0) + parse_int(row.get('source_valid_rows'))
        grouped_smoothed[rally_id] = grouped_smoothed.get(rally_id, 0) + parse_int(row.get('smoothed_valid_rows'))
    ordered_rallies = sorted(grouped_source.keys())
    labels = [f'r{rally_id}' for rally_id in ordered_rallies]
    source_valid = [grouped_source[rally_id] for rally_id in ordered_rallies]
    smoothed_valid = [grouped_smoothed[rally_id] for rally_id in ordered_rallies]
    save_bar_chart(
        labels=labels,
        values_a=source_valid,
        values_b=smoothed_valid,
        title='Player Valid Points Before vs After Smoothing',
        ylabel='rows',
        legend_a='source_valid',
        legend_b='smoothed_valid',
        output_path=output_path,
    )


def save_player_overall_topdown(
    player_points: list[dict[str, float | str]],
    output_path: Path,
) -> None:
    near_x = [float(point['x']) for point in player_points if point['player_id'] == 'near']
    near_y = [float(point['y']) for point in player_points if point['player_id'] == 'near']
    far_x = [float(point['x']) for point in player_points if point['player_id'] == 'far']
    far_y = [float(point['y']) for point in player_points if point['player_id'] == 'far']

    fig, ax = plt.subplots(figsize=(7, 14))
    fig.patch.set_facecolor('#0f172a')
    draw_full_court(ax)
    if far_x:
        ax.hexbin(far_x, far_y, gridsize=20, extent=(0, COURT_WIDTH, 0.0, NET_Y), cmap='Blues', mincnt=1, alpha=0.9)
    if near_x:
        ax.hexbin(
            near_x,
            near_y,
            gridsize=20,
            extent=(0, COURT_WIDTH, NET_Y, COURT_LENGTH),
            cmap='Reds',
            mincnt=1,
            alpha=0.9,
        )
    style_court_axes(ax, full_court=True, title='Player Movement Top-Down Heatmap')
    ax.text(
        0.03,
        0.965,
        f'Far: {len(far_x)}',
        transform=ax.transAxes,
        color='white',
        fontsize=11,
        bbox=dict(facecolor='#111827', edgecolor='white', alpha=0.85, boxstyle='round,pad=0.35'),
    )
    ax.text(
        0.03,
        0.035,
        f'Near: {len(near_x)}',
        transform=ax.transAxes,
        color='white',
        fontsize=11,
        bbox=dict(facecolor='#111827', edgecolor='white', alpha=0.85, boxstyle='round,pad=0.35'),
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)


def save_standardized_halfcourt_heatmap(
    player_points: list[dict[str, float | str]],
    player_id: str,
    output_path: Path,
) -> None:
    if player_id == 'near':
        xs = [float(point['x']) for point in player_points if point['player_id'] == 'near' and float(point['y']) >= NET_Y]
        ys = [float(point['y']) for point in player_points if point['player_id'] == 'near' and float(point['y']) >= NET_Y]
        title = 'Near Standardized Half-Court Heatmap'
        cmap = 'Reds'
    else:
        xs = [float(point['x']) for point in player_points if point['player_id'] == 'far' and float(point['y']) <= NET_Y]
        ys = [COURT_LENGTH - float(point['y']) for point in player_points if point['player_id'] == 'far' and float(point['y']) <= NET_Y]
        title = 'Far Standardized Half-Court Heatmap'
        cmap = 'Blues'

    fig, ax = plt.subplots(figsize=(7, 7.8))
    fig.patch.set_facecolor('#0f172a')
    draw_half_court(ax)
    if xs:
        ax.hexbin(xs, ys, gridsize=18, extent=(0, COURT_WIDTH, NET_Y, COURT_LENGTH), cmap=cmap, mincnt=1, alpha=0.92)
    style_court_axes(ax, full_court=False, title=title)
    ax.text(
        0.03,
        0.06,
        f'points={len(xs)}',
        transform=ax.transAxes,
        color='white',
        fontsize=11,
        bbox=dict(facecolor='#111827', edgecolor='white', alpha=0.85, boxstyle='round,pad=0.35'),
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)


def save_standardized_fullcourt_split_compare(
    player_points: list[dict[str, float | str]],
    output_path: Path,
) -> None:
    near_x = [float(point['x']) for point in player_points if point['player_id'] == 'near' and float(point['y']) >= NET_Y]
    near_y = [float(point['y']) for point in player_points if point['player_id'] == 'near' and float(point['y']) >= NET_Y]
    far_x = [float(point['x']) for point in player_points if point['player_id'] == 'far' and float(point['y']) <= NET_Y]
    far_y = [float(point['y']) for point in player_points if point['player_id'] == 'far' and float(point['y']) <= NET_Y]

    fig, ax = plt.subplots(figsize=(7, 14))
    fig.patch.set_facecolor('#0f172a')
    draw_full_court(ax)
    if far_x:
        ax.hexbin(far_x, far_y, gridsize=20, extent=(0, COURT_WIDTH, 0.0, NET_Y), cmap='Blues', mincnt=1, alpha=0.92)
    if near_x:
        ax.hexbin(
            near_x,
            near_y,
            gridsize=20,
            extent=(0, COURT_WIDTH, NET_Y, COURT_LENGTH),
            cmap='Reds',
            mincnt=1,
            alpha=0.92,
        )
    style_court_axes(ax, full_court=True, title='Standardized Full-Court Split Compare')
    ax.text(
        0.03,
        0.965,
        f'Far (upper): {len(far_x)}',
        transform=ax.transAxes,
        color='white',
        fontsize=11,
        bbox=dict(facecolor='#111827', edgecolor='white', alpha=0.85, boxstyle='round,pad=0.35'),
    )
    ax.text(
        0.03,
        0.035,
        f'Near (lower): {len(near_x)}',
        transform=ax.transAxes,
        color='white',
        fontsize=11,
        bbox=dict(facecolor='#111827', edgecolor='white', alpha=0.85, boxstyle='round,pad=0.35'),
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)


def save_single_rally_player_trajectory(
    rally_id: str,
    rally_points: list[dict[str, float | str]],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 14))
    fig.patch.set_facecolor('#0f172a')
    draw_full_court(ax)

    far_points = [(float(point['x']), float(point['y'])) for point in rally_points if point['player_id'] == 'far']
    near_points = [(float(point['x']), float(point['y'])) for point in rally_points if point['player_id'] == 'near']
    for points, color, label in [(far_points, '#60a5fa', 'far'), (near_points, '#f87171', 'near')]:
        if not points:
            continue
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        ax.plot(xs, ys, color=color, linewidth=1.8, alpha=0.95, label=label)
        ax.scatter(xs[0], ys[0], color=color, s=36, marker='o')
        ax.scatter(xs[-1], ys[-1], color=color, s=48, marker='x')
    style_court_axes(ax, full_court=True, title=f'Player Top-Down Trajectory - r{rally_id}')
    ax.legend(facecolor='#111827', edgecolor='white', labelcolor='white', loc='upper right')
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)


def save_shuttle_trajectory_samples(
    shuttle_summary_rows: list[dict[str, str]],
    shuttle_track_rows: list[dict[str, str]],
    output_path: Path,
) -> None:
    grouped_rows = group_rows_by_video(shuttle_track_rows)
    scored_rows = []
    for row in shuttle_summary_rows:
        track_rows = parse_int(row.get('track_rows'))
        visible_rows = parse_int(row.get('visible_rows'))
        ratio = visible_rows / track_rows if track_rows else 0.0
        scored_rows.append((ratio, row))
    scored_rows.sort(key=lambda item: item[0])

    selected = [row for _, row in scored_rows[:3]] + [row for _, row in scored_rows[-3:]]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes_flat = list(axes.flat)
    for axis, summary_row in zip(axes_flat, selected, strict=True):
        video_stem = summary_row.get('video_stem', '')
        points = []
        for row in grouped_rows.get(video_stem, []):
            x_value, y_value = pick_coordinate(row, 'smoothed_x', 'smoothed_y')
            if x_value is None or y_value is None:
                continue
            points.append((x_value, y_value))
        if points:
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            axis.plot(xs, ys, color='#ef4444', linewidth=1.4)
            axis.scatter(xs, ys, s=10, color='#ef4444', alpha=0.5)
            axis.invert_yaxis()
        label = rally_label(summary_row)
        track_rows = parse_int(summary_row.get('track_rows'))
        visible_rows = parse_int(summary_row.get('visible_rows'))
        ratio = visible_rows / track_rows if track_rows else 0.0
        axis.set_title(f'{label} ratio={ratio:.3f}')
        axis.set_xlabel('x')
        axis.set_ylabel('y')
        axis.grid(linestyle='--', alpha=0.25)

    for axis in axes_flat[len(selected):]:
        axis.axis('off')
    fig.suptitle('Shuttle Trajectory Samples')
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_smoothing_gain_chart(
    smoothing_rows: list[dict[str, str]],
    output_path: Path,
) -> None:
    labels = [rally_label(row) for row in smoothing_rows]
    source_valid = [parse_int(row.get('source_valid_rows')) for row in smoothing_rows]
    smoothed_valid = [parse_int(row.get('smoothed_valid_rows')) for row in smoothing_rows]
    save_bar_chart(
        labels=labels,
        values_a=source_valid,
        values_b=smoothed_valid,
        title='Shuttle Valid Points Before vs After Smoothing',
        ylabel='rows',
        legend_a='source_valid',
        legend_b='smoothed_valid',
        output_path=output_path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Convert tracking CSV files into visualization charts.')
    parser.add_argument(
        '--shuttle-summary-csv',
        type=Path,
        required=True,
        help='Per-rally shuttle tracking summary CSV.',
    )
    parser.add_argument(
        '--player-summary-csv',
        type=Path,
        default=None,
        help='Per-rally player tracking summary CSV.',
    )
    parser.add_argument(
        '--shuttle-track-csv',
        type=Path,
        required=True,
        help='Shuttle track CSV. Smoothed columns are used if present.',
    )
    parser.add_argument(
        '--shuttle-smoothing-summary-csv',
        type=Path,
        default=None,
        help='Optional smoothing summary CSV generated by trajectory_smoothing.py.',
    )
    parser.add_argument(
        '--player-track-csv',
        type=Path,
        default=None,
        help='Optional player track CSV. Smoothed court columns are used if present.',
    )
    parser.add_argument(
        '--player-smoothing-summary-csv',
        type=Path,
        default=None,
        help='Optional player smoothing summary CSV generated by trajectory_smoothing.py.',
    )
    parser.add_argument(
        '--player-min-confidence',
        type=float,
        default=0.18,
        help='Minimum confidence used to keep player points in top-down charts.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Directory to save PNG charts.',
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = ensure_dir(args.output_dir)

    shuttle_summary_rows = read_csv_rows(args.shuttle_summary_csv)
    shuttle_track_rows = read_csv_rows(args.shuttle_track_csv)
    if not shuttle_summary_rows or not shuttle_track_rows:
        raise RuntimeError('Shuttle summary and track CSVs must both contain rows.')

    labels = [rally_label(row) for row in shuttle_summary_rows]
    visible_rows = [parse_int(row.get('visible_rows')) for row in shuttle_summary_rows]
    interpolated_rows = [parse_int(row.get('interpolated_rows')) for row in shuttle_summary_rows]
    save_bar_chart(
        labels=labels,
        values_a=visible_rows,
        values_b=interpolated_rows,
        title='Shuttle Visible vs Interpolated Rows',
        ylabel='rows',
        legend_a='visible',
        legend_b='interpolated',
        output_path=output_dir / 'shuttle_visible_vs_interpolated.png',
    )
    save_ratio_chart(shuttle_summary_rows, output_dir / 'shuttle_visible_ratio.png')
    save_shuttle_trajectory_samples(
        shuttle_summary_rows=shuttle_summary_rows,
        shuttle_track_rows=shuttle_track_rows,
        output_path=output_dir / 'shuttle_trajectory_samples.png',
    )

    if args.player_summary_csv is not None:
        player_summary_rows = read_csv_rows(args.player_summary_csv)
        if player_summary_rows:
            save_player_track_rows_chart(
                player_summary_rows=player_summary_rows,
                output_path=output_dir / 'player_track_rows.png',
            )

    if args.shuttle_smoothing_summary_csv is not None:
        smoothing_rows = read_csv_rows(args.shuttle_smoothing_summary_csv)
        if smoothing_rows:
            save_smoothing_gain_chart(
                smoothing_rows=smoothing_rows,
                output_path=output_dir / 'shuttle_smoothing_gain.png',
            )

    if args.player_track_csv is not None:
        player_track_rows = read_csv_rows(args.player_track_csv)
        if player_track_rows:
            player_points, rally_points = collect_player_points(
                player_track_rows=player_track_rows,
                min_confidence=args.player_min_confidence,
            )
            if player_points:
                save_player_overall_topdown(
                    player_points=player_points,
                    output_path=output_dir / 'player_movement_topdown.png',
                )
                save_standardized_halfcourt_heatmap(
                    player_points=player_points,
                    player_id='near',
                    output_path=output_dir / 'player_near_standardized_halfcourt_heatmap.png',
                )
                save_standardized_halfcourt_heatmap(
                    player_points=player_points,
                    player_id='far',
                    output_path=output_dir / 'player_far_standardized_halfcourt_heatmap.png',
                )
                save_standardized_fullcourt_split_compare(
                    player_points=player_points,
                    output_path=output_dir / 'player_standardized_fullcourt_split_compare.png',
                )

                selected_rally_id = None
                if args.player_smoothing_summary_csv is not None:
                    player_smoothing_rows = read_csv_rows(args.player_smoothing_summary_csv)
                    if player_smoothing_rows:
                        player_counts = collect_player_counts_by_rally(player_smoothing_rows)
                        if player_counts:
                            selected_rally_id = max(player_counts.items(), key=lambda item: item[1])[0]
                        save_player_smoothing_gain_chart(
                            player_smoothing_rows=player_smoothing_rows,
                            output_path=output_dir / 'player_smoothing_gain.png',
                        )
                if selected_rally_id is None and rally_points:
                    selected_rally_id = max(rally_points.items(), key=lambda item: len(item[1]))[0]
                if selected_rally_id and selected_rally_id in rally_points:
                    save_single_rally_player_trajectory(
                        rally_id=selected_rally_id,
                        rally_points=rally_points[selected_rally_id],
                        output_path=output_dir / f'player_rally_{selected_rally_id}_topdown_trajectory.png',
                    )

    print(f'Charts saved to: {output_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

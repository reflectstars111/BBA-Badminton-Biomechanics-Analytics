from __future__ import annotations

import argparse
from pathlib import Path

from badminton_data_process.core.io import ensure_dir, read_csv_rows, write_csv_rows
from badminton_data_process.core.schemas import MAIN_VIEW_QUALITY_FIELDS, REJECTED_SEGMENT_FIELDS


def parse_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def group_by_rally(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row.get("rally_id", "000"), []).append(row)
    return grouped


def projection_quality(rows: list[dict[str, str]]) -> dict[str, float]:
    total = 0
    outliers = 0
    absurd_y = 0
    boundary_stuck = 0
    for row in rows:
        x = parse_float(row.get("smoothed_court_x") or row.get("court_x"))
        y = parse_float(row.get("smoothed_court_y") or row.get("court_y"))
        if x is None or y is None:
            continue
        total += 1
        if x < -0.5 or x > 6.6 or y < -0.8 or y > 14.2:
            outliers += 1
        if y < -5.0 or y > 20.0:
            absurd_y += 1
        if abs(x) <= 0.03 or abs(x - 6.10) <= 0.03:
            boundary_stuck += 1
    if total == 0:
        return {
            "projection_outlier_ratio": 1.0,
            "boundary_stuck_ratio": 1.0,
            "absurd_y_ratio": 1.0,
            "quality_score": 0.0,
        }
    out_ratio = outliers / total
    boundary_ratio = boundary_stuck / total
    absurd_ratio = absurd_y / total
    quality = max(0.0, 1.0 - 1.2 * out_ratio - 0.45 * boundary_ratio - 2.0 * absurd_ratio)
    return {
        "projection_outlier_ratio": out_ratio,
        "boundary_stuck_ratio": boundary_ratio,
        "absurd_y_ratio": absurd_ratio,
        "quality_score": quality,
    }


def choose_reject_reason(metrics: dict[str, float], min_quality_score: float) -> str:
    if metrics["absurd_y_ratio"] > 0.02:
        return "court_projection_outlier"
    if metrics["projection_outlier_ratio"] > 0.15:
        return "court_projection_outlier"
    if metrics["boundary_stuck_ratio"] > 0.35:
        return "bad_calibration_or_not_main_view"
    if metrics["quality_score"] < min_quality_score:
        return "low_posthoc_quality_score"
    return ""


def review_main_view_run(
    run_dir: Path,
    output_dir: Path | None = None,
    min_quality_score: float = 0.75,
) -> int:
    run_dir = Path(run_dir)
    output_dir = ensure_dir(output_dir or run_dir / "review")
    rallies = read_csv_rows(run_dir / "rallies.csv")
    player_rows = read_csv_rows(run_dir / "annotations" / "player_tracks_smoothed.csv")
    by_rally = group_by_rally(player_rows)
    quality_rows: list[dict[str, object]] = []
    rejected_rows: list[dict[str, object]] = []

    for index, rally in enumerate(rallies, start=1):
        rally_id = rally.get("rally_id", f"{index:03d}")
        metrics = projection_quality(by_rally.get(rally_id, []))
        reason = choose_reject_reason(metrics, min_quality_score)
        accepted = int(reason == "")
        quality_row = {
            "segment_id": rally_id,
            "rally_id": rally_id,
            "start_frame": rally.get("start_frame", ""),
            "end_frame": rally.get("end_frame", ""),
            "start_time": rally.get("start_time", ""),
            "end_time": rally.get("end_time", ""),
            "duration_seconds": rally.get("duration_seconds", ""),
            "quality_score": round(metrics["quality_score"], 4),
            "main_view_score": "",
            "court_score": "",
            "geometry_score": "",
            "layout_score": "",
            "projection_outlier_ratio": round(metrics["projection_outlier_ratio"], 4),
            "boundary_stuck_ratio": round(metrics["boundary_stuck_ratio"], 4),
            "absurd_y_ratio": round(metrics["absurd_y_ratio"], 4),
            "accepted": accepted,
            "reject_reason": reason,
        }
        quality_rows.append(quality_row)
        if not accepted:
            rejected_rows.append(
                {
                    "segment_id": rally_id,
                    "rally_id": rally_id,
                    "start_frame": rally.get("start_frame", ""),
                    "end_frame": rally.get("end_frame", ""),
                    "start_time": rally.get("start_time", ""),
                    "end_time": rally.get("end_time", ""),
                    "duration_seconds": rally.get("duration_seconds", ""),
                    "reject_reason": reason,
                    "score": round(metrics["quality_score"], 4),
                }
            )

    write_csv_rows(output_dir / "main_view_quality.csv", MAIN_VIEW_QUALITY_FIELDS, quality_rows)
    write_csv_rows(output_dir / "rejected_segments.csv", REJECTED_SEGMENT_FIELDS, rejected_rows)
    print(f"Reviewed rallies: {len(quality_rows)}")
    print(f"Rejected rallies: {len(rejected_rows)}")
    print(f"Review output dir: {output_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Post-hoc review for main-view/rally quality.")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--min-quality-score", type=float, default=0.75)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return review_main_view_run(args.run, args.output_dir, args.min_quality_score)


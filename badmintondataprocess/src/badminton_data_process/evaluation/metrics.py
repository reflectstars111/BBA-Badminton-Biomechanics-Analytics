from __future__ import annotations

from pathlib import Path

from badminton_data_process.core.io import read_csv_rows


def _truthy(value: str | None) -> bool:
    return str(value or "").strip() in {"1", "true", "True", "yes"}


def shuttle_visibility_stats(track_csv: str | Path) -> dict[str, float]:
    rows = read_csv_rows(track_csv)
    total = len(rows)
    visible = sum(1 for row in rows if _truthy(row.get("visibility")))
    interpolated = sum(1 for row in rows if _truthy(row.get("is_interpolated")))
    return {
        "total_rows": float(total),
        "visible_rows": float(visible),
        "interpolated_rows": float(interpolated),
        "visible_ratio": visible / total if total else 0.0,
        "interpolated_ratio": interpolated / total if total else 0.0,
    }


def player_track_stats(track_csv: str | Path) -> dict[str, float]:
    rows = read_csv_rows(track_csv)
    total = len(rows)
    interpolated = sum(1 for row in rows if _truthy(row.get("is_interpolated")))
    valid_court = sum(1 for row in rows if row.get("court_x") not in (None, "") and row.get("court_y") not in (None, ""))
    return {
        "total_rows": float(total),
        "interpolated_rows": float(interpolated),
        "court_projected_rows": float(valid_court),
        "interpolated_ratio": interpolated / total if total else 0.0,
        "court_projection_ratio": valid_court / total if total else 0.0,
    }


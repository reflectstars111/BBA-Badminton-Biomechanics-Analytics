from __future__ import annotations

from pathlib import Path

from badminton_data_process.core.io import read_csv_rows, write_csv_rows
from badminton_data_process.core.schemas import RALLY_FIELDS


def write_manual_review_template(rallies_csv: Path, output_csv: Path) -> None:
    rows = read_csv_rows(rallies_csv)
    review_rows = []
    fieldnames = [*RALLY_FIELDS, "review_status", "corrected_start_time", "corrected_end_time", "review_notes"]
    for row in rows:
        review_rows.append(
            {
                **{field: row.get(field, "") for field in RALLY_FIELDS},
                "review_status": "pending",
                "corrected_start_time": "",
                "corrected_end_time": "",
                "review_notes": "",
            }
        )
    write_csv_rows(output_csv, fieldnames, review_rows)


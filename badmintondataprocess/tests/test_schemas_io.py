from __future__ import annotations

from badminton_data_process.core.io import missing_fields, read_csv_rows, write_csv_rows
from badminton_data_process.core.schemas import MATCH_FIELDS, REQUIRED_MATCH_FIELDS, MatchRecord, dataclass_to_row


def test_schema_row_roundtrip(tmp_path) -> None:
    record = MatchRecord(
        match_id="MS_2024_Test_Final_001",
        source="BWF TV",
        url="https://example.test/video",
        tournament="Test Open",
        year="2024",
        discipline="MS",
        round="Final",
        player_1="A",
        player_2="B",
        resolution="1080p",
        fps="30",
        camera_type="broadcast",
    )
    output = tmp_path / "matches.csv"
    write_csv_rows(output, MATCH_FIELDS, [dataclass_to_row(record)])
    rows = read_csv_rows(output)
    assert rows[0]["match_id"] == "MS_2024_Test_Final_001"
    assert missing_fields(rows[0], REQUIRED_MATCH_FIELDS) == []


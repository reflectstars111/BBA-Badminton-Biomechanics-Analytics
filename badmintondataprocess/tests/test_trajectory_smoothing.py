from __future__ import annotations

from badminton_data_process.core.io import read_csv_rows, write_csv_rows
from badminton_data_process.core.schemas import SHUTTLE_TRACK_FIELDS
from badminton_data_process.smoothing.trajectory import smooth_trajectory


def _shuttle_row(frame_id: int, x: str, y: str) -> dict[str, object]:
    visible = int(bool(x and y))
    return {
        "video_path": "rally.mp4",
        "video_stem": "rally",
        "rally_id": "001",
        "frame_id": frame_id,
        "timestamp": frame_id / 30.0,
        "x": x,
        "y": y,
        "confidence": "0.9" if visible else "",
        "is_interpolated": "0",
        "visibility": visible,
    }


def _smooth(
    tmp_path,
    rows: list[dict[str, object]],
    max_gap_frames: int = 2,
    max_interpolation_displacement_px: float = 80.0,
):
    source = tmp_path / "source.csv"
    output = tmp_path / "smoothed.csv"
    summary = tmp_path / "summary.csv"
    write_csv_rows(source, SHUTTLE_TRACK_FIELDS, rows)
    smooth_trajectory(
        source,
        output,
        summary,
        min_confidence=0.2,
        max_gap_frames=max_gap_frames,
        window_size=1,
        ema_alpha=1.0,
        max_interpolation_displacement_px=max_interpolation_displacement_px,
    )
    return read_csv_rows(output), read_csv_rows(summary)


def test_short_internal_gap_is_filled_and_marked(tmp_path) -> None:
    rows, summary = _smooth(
        tmp_path,
        [
            _shuttle_row(0, "0", "0"),
            _shuttle_row(1, "", ""),
            _shuttle_row(2, "2", "2"),
        ],
    )

    assert [row["frame_id"] for row in rows] == ["0", "1", "2"]
    assert rows[1]["smoothed_x"] == "1.0"
    assert rows[1]["is_gap_filled"] == "1"
    assert rows[1]["is_smoothed_valid"] == "1"
    assert summary[0]["source_valid_rows"] == "2"
    assert summary[0]["smoothed_valid_rows"] == "3"


def test_long_gap_remains_missing_instead_of_becoming_valid(tmp_path) -> None:
    rows, summary = _smooth(
        tmp_path,
        [
            _shuttle_row(0, "0", "0"),
            _shuttle_row(1, "", ""),
            _shuttle_row(2, "", ""),
            _shuttle_row(3, "", ""),
            _shuttle_row(4, "4", "4"),
        ],
        max_gap_frames=2,
    )

    for row in rows[1:4]:
        assert row["smoothed_x"] == ""
        assert row["smoothed_y"] == ""
        assert row["is_gap_filled"] == "0"
        assert row["is_smoothed_valid"] == "0"
    assert summary[0]["smoothed_valid_rows"] == "2"


def test_short_gap_with_large_endpoint_displacement_remains_missing(tmp_path) -> None:
    rows, summary = _smooth(
        tmp_path,
        [
            _shuttle_row(0, "100", "10"),
            _shuttle_row(1, "", ""),
            _shuttle_row(2, "500", "12"),
        ],
        max_gap_frames=2,
        max_interpolation_displacement_px=80.0,
    )

    assert rows[1]["smoothed_x"] == ""
    assert rows[1]["smoothed_y"] == ""
    assert rows[1]["is_gap_filled"] == "0"
    assert rows[1]["is_smoothed_valid"] == "0"
    assert summary[0]["gap_filled_rows"] == "0"


def test_omitted_frames_break_smoothing_state_and_rows_are_sorted(tmp_path) -> None:
    rows, _ = _smooth(
        tmp_path,
        [
            _shuttle_row(10, "100", "100"),
            _shuttle_row(0, "0", "0"),
        ],
    )

    assert [row["frame_id"] for row in rows] == ["0", "10"]
    assert [row["smoothed_x"] for row in rows] == ["0.0", "100.0"]

from __future__ import annotations

import json

from badminton_data_process.core.artifacts import (
    inspect_calibration_json,
    inspect_csv,
    inspect_directory,
    inspect_file,
    inspect_file_set,
    inspect_video,
)
from badminton_data_process.core.io import write_csv_rows
from badminton_data_process.core.schemas import ArtifactStatus


def test_csv_distinguishes_missing_empty_and_valid_artifacts(tmp_path) -> None:
    missing = inspect_csv(tmp_path / "missing.csv", min_rows=1)
    assert missing.status == ArtifactStatus.MISSING

    empty_path = tmp_path / "empty.csv"
    write_csv_rows(empty_path, ["frame_id", "x"], [])
    empty = inspect_csv(
        empty_path,
        min_rows=1,
        required_fields={"frame_id", "x"},
    )
    assert empty.status == ArtifactStatus.EMPTY
    assert empty.details["row_count"] == 0
    allowed_empty = inspect_csv(
        empty_path,
        min_rows=0,
        required_fields={"frame_id", "x"},
    )
    assert allowed_empty.status == ArtifactStatus.VALID
    assert allowed_empty.details["row_count"] == 0

    valid_path = tmp_path / "valid.csv"
    write_csv_rows(valid_path, ["frame_id", "x"], [{"frame_id": 0, "x": 10}])
    valid = inspect_csv(
        valid_path,
        min_rows=1,
        required_fields={"frame_id", "x"},
    )
    assert valid.status == ArtifactStatus.VALID
    assert valid.details["row_count"] == 1


def test_csv_rejects_missing_required_fields(tmp_path) -> None:
    path = tmp_path / "wrong_schema.csv"
    write_csv_rows(path, ["frame_id"], [{"frame_id": 0}])

    report = inspect_csv(path, required_fields={"frame_id", "x"})

    assert report.status == ArtifactStatus.INVALID
    assert report.details["missing_fields"] == ["x"]


def test_file_and_directory_reports_capture_counts(tmp_path) -> None:
    empty_file = tmp_path / "empty.bin"
    empty_file.touch()
    assert inspect_file(empty_file).status == ArtifactStatus.EMPTY

    output_dir = tmp_path / "charts"
    output_dir.mkdir()
    assert inspect_directory(output_dir, pattern="*.png").status == ArtifactStatus.EMPTY
    chart = output_dir / "chart.png"
    chart.write_bytes(b"png")
    directory = inspect_directory(output_dir, pattern="*.png")
    assert directory.status == ArtifactStatus.VALID
    assert directory.details["file_count"] == 1


def test_file_set_rejects_missing_members(tmp_path) -> None:
    existing = tmp_path / "existing.json"
    existing.write_text("{}", encoding="utf-8")

    report = inspect_file_set(
        [existing, tmp_path / "missing.json"],
        name="calibration files",
    )

    assert report.status == ArtifactStatus.MISSING
    assert report.details["file_count"] == 2


def test_calibration_json_requires_validated_v2_contract(tmp_path) -> None:
    valid_path = tmp_path / "valid_calibration.json"
    valid_path.write_text(
        json.dumps(
            {
                "artifact_version": "2.0",
                "validated": True,
                "court_type": "badminton_standard",
                "coordinate_unit": "metre",
                "image_points_tl_tr_br_bl": [[0, 0], [1, 0], [1, 1], [0, 1]],
                "court_points_tl_tr_br_bl": [[0, 0], [1, 0], [1, 1], [0, 1]],
                "homography_image_to_court": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "quality": {"line_support": 0.8},
                "temporal_validation": {"stable_candidate_count": 3},
            }
        ),
        encoding="utf-8",
    )
    assert inspect_calibration_json(valid_path).status == ArtifactStatus.VALID

    invalid_path = tmp_path / "invalid_calibration.json"
    invalid_path.write_text("{}", encoding="utf-8")
    invalid = inspect_calibration_json(invalid_path)
    assert invalid.status == ArtifactStatus.INVALID
    assert "artifact_version" in invalid.details["missing_fields"]


def test_video_requires_a_decodable_frame(tmp_path) -> None:
    invalid_path = tmp_path / "invalid.mp4"
    invalid_path.write_bytes(b"not a video")

    report = inspect_video(invalid_path)

    assert report.status == ArtifactStatus.INVALID

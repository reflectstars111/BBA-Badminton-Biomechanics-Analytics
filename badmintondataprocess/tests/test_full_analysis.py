from __future__ import annotations

from pathlib import Path

from badminton_data_process.core.io import write_csv_rows, write_json
import pytest

from badminton_data_process.core.paths import RunLayout
from badminton_data_process.pipeline.full import (
    build_analysis_summary,
    validate_resume_identity,
)


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(rows[0]) if rows else ["status"]
    write_csv_rows(path, fields, rows)


def test_full_analysis_summary_collects_cleaning_tracking_and_video_outputs(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "full_001"
    demo = run_dir / "outputs" / "demo" / "full.mp4"
    demo.parent.mkdir(parents=True)
    demo.write_bytes(b"video")
    write_json(
        run_dir / "manifest.json",
        {
            "run_id": "full_001",
            "config": {"demo_rendering": {"output_filename": "full.mp4"}},
            "stages": [
                {"name": "main_view", "status": "success", "duration_seconds": 1.2},
                {"name": "demo_rendering", "status": "success", "duration_seconds": 2.3},
            ],
        },
    )
    _write_rows(run_dir / "main_view" / "main_view_segments.csv", [{"segment_id": "001"}])
    _write_rows(
        run_dir / "rally_decisions.csv",
        [
            {"status": "accepted"},
            {"status": "rejected"},
        ],
    )
    _write_rows(
        run_dir / "rallies.csv",
        [{"rally_id": "001", "output_path": str(run_dir / "rallies" / "001.mp4")}],
    )
    _write_rows(
        run_dir / "annotations" / "court_calibration_summary.csv",
        [{"status": "success"}],
    )
    _write_rows(
        run_dir / "annotations" / "player_tracking_summary.csv",
        [{"status": "success", "track_rows": 120, "pose_rows": 116}],
    )
    _write_rows(
        run_dir / "annotations" / "shuttle_tracking_summary.csv",
        [
            {
                "status": "success",
                "track_rows": 60,
                "visible_rows": 52,
                "interpolated_rows": 5,
            }
        ],
    )
    _write_rows(
        run_dir / "annotations" / "shuttle_smoothing_summary.csv",
        [
            {
                "schema": "shuttle",
                "smoothed_valid_rows": 55,
                "gap_filled_rows": 3,
            }
        ],
    )

    summary = build_analysis_summary(run_dir)

    assert summary["status"] == "success"
    assert summary["counts"] == {
        "main_view_segments": 1,
        "rally_candidates_accepted": 1,
        "rally_candidates_rejected": 1,
        "usable_rallies_after_calibration": 1,
        "calibrations_successful": 1,
        "calibrations_rejected": 0,
        "player_track_rows": 120,
        "valid_pose_rows": 116,
        "shuttle_track_rows": 60,
        "shuttle_visible_rows": 52,
        "shuttle_interpolated_rows": 5,
        "shuttle_smoothed_valid_rows": 55,
        "shuttle_gap_filled_rows": 3,
        "biomechanics_kinematics_rows": 0,
        "biomechanics_eligible_rows": 0,
        "biomechanics_rejected_rows": 0,
        "biomechanics_action_candidates": 0,
        "biomechanics_eligible_phase_rows": 0,
    }
    assert summary["outputs"]["analysis_video"] == str(demo)
    assert summary["outputs"]["shuttle_smoothing_summary"] == str(
        run_dir / "annotations" / "shuttle_smoothing_summary.csv"
    )


def test_full_video_gpu_profile_is_automatic_and_explicit_cuda() -> None:
    root = Path(__file__).resolve().parents[1]
    from badminton_data_process.core.config import load_config
    from badminton_data_process.core.config_schema import parse_config

    config = parse_config(
        load_config(root / "configs" / "production" / "full_video_gpu.yaml", root=root)
    )

    assert config.court_calibration.reference_points is None
    assert config.court_calibration.detector == "hybrid"
    assert config.player_tracking.detector == "rtmpose"
    assert config.player_tracking.rtmpose_device == "cuda"
    assert config.shuttle_tracking.model == "tracknet"
    assert config.shuttle_tracking.tracknet_device == "cuda"
    assert config.shuttle_tracking.max_missing_frames == 0
    assert config.smoothing.shuttle_max_interpolation_displacement_px == 80.0
    assert config.demo_rendering.max_rallies is None


def test_full_analysis_refuses_to_resume_with_another_source(tmp_path) -> None:
    layout = RunLayout.create(tmp_path, "existing")
    source_a = tmp_path / "a.mp4"
    source_b = tmp_path / "b.mp4"
    write_json(
        layout.manifest_json,
        {
            "config": {"profile": "gpu"},
            "stages": [{"name": "main_view", "inputs": [str(source_a)]}],
        },
    )

    with pytest.raises(RuntimeError, match="belongs to another source video"):
        validate_resume_identity(
            layout,
            source_b,
            {"profile": "gpu"},
            force=False,
        )


def test_full_analysis_refuses_to_resume_with_another_config(tmp_path) -> None:
    layout = RunLayout.create(tmp_path, "existing")
    source = tmp_path / "a.mp4"
    write_json(
        layout.manifest_json,
        {
            "config": {"profile": "old"},
            "stages": [{"name": "main_view", "inputs": [str(source)]}],
        },
    )

    with pytest.raises(RuntimeError, match="different configuration"):
        validate_resume_identity(
            layout,
            source,
            {"profile": "new"},
            force=False,
        )

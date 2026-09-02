from __future__ import annotations

import json
from pathlib import Path

import pytest

import badminton_data_process.pipeline.run as pipeline_module
from badminton_data_process.core.config import load_yaml
from badminton_data_process.core.io import read_csv_rows, write_csv_rows, write_json
from badminton_data_process.core.run import (
    RunContext,
    StageExecutionError,
    stage_report,
)
from badminton_data_process.core.schemas import (
    ArtifactKind,
    ArtifactReport,
    ArtifactStatus,
    COURT_CALIBRATION_SUMMARY_FIELDS,
    RALLY_FIELDS,
    StageName,
    StageStatus,
)


def _manifest(root: Path, run_id: str) -> dict[str, object]:
    path = root / "runs" / run_id / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _success_calibration(*_args, **_kwargs) -> int:
    summary_csv = Path(_args[3])
    write_csv_rows(
        summary_csv,
        COURT_CALIBRATION_SUMMARY_FIELDS,
        [
            {
                "video_path": "rally_ok.mp4",
                "video_stem": "rally_ok",
                "status": "success",
                "frame_index": 0,
                "json_path": "rally_ok.json",
                "preview_path": "rally_ok.png",
                "message": "",
            }
        ],
    )
    return 0


def _stub_successful_pipeline(monkeypatch, project_dir: Path) -> None:
    config = load_yaml(project_dir / "configs" / "default.yaml")
    monkeypatch.setattr(pipeline_module, "load_config", lambda *_args, **_kwargs: config)
    monkeypatch.setattr(pipeline_module, "segment_rallies", lambda **_kwargs: 0)
    monkeypatch.setattr(pipeline_module, "calibrate_courts", _success_calibration)
    monkeypatch.setattr(pipeline_module, "track_players", lambda **_kwargs: 0)
    monkeypatch.setattr(pipeline_module, "track_shuttle", lambda **_kwargs: 0)
    monkeypatch.setattr(pipeline_module, "smooth_trajectory", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pipeline_module, "visualize_tracking_main", lambda *_args: 0)
    monkeypatch.setattr(pipeline_module, "tactics_main", lambda *_args: 0)
    monkeypatch.setattr(pipeline_module, "analyze_kinematics_csv", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(pipeline_module, "analyze_action_events_csv", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(pipeline_module, "analyze_swing_phases_csv", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(pipeline_module, "analyze_event_descriptors", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(pipeline_module, "classify_action_events_csv", lambda *_args, **_kwargs: {})
    _stub_artifact_inspection(monkeypatch)


def _stub_artifact_inspection(monkeypatch) -> None:
    def valid_artifact(*_args, name: str | None = None, **_kwargs) -> ArtifactReport:
        return ArtifactReport(
            name=name or "test artifact",
            path="test-artifact",
            kind=ArtifactKind.FILE,
            status=ArtifactStatus.VALID,
        )

    monkeypatch.setattr(pipeline_module, "inspect_csv", valid_artifact)
    monkeypatch.setattr(pipeline_module, "inspect_calibration_json", valid_artifact)
    monkeypatch.setattr(pipeline_module, "inspect_directory", valid_artifact)
    monkeypatch.setattr(pipeline_module, "inspect_file", valid_artifact)
    monkeypatch.setattr(pipeline_module, "inspect_file_set", valid_artifact)
    monkeypatch.setattr(pipeline_module, "inspect_video", valid_artifact)
    monkeypatch.setattr(pipeline_module, "analyze_main_view", lambda **_kwargs: 0)


def test_stage_report_records_successful_legacy_exit_code(tmp_path) -> None:
    context = RunContext(tmp_path, "success", {})

    with stage_report(context, StageName.RALLY_SEGMENTATION) as stage:
        stage.accept_legacy(0, operation="test stage")

    report = context.reports[0]
    assert report.status == StageStatus.SUCCESS
    assert report.exit_code == 0
    assert _manifest(tmp_path, "success")["stages"][0]["exit_code"] == 0


def test_stage_report_serializes_artifact_checks(tmp_path) -> None:
    artifact_path = tmp_path / "tracks.csv"
    write_csv_rows(artifact_path, ["frame_id"], [{"frame_id": 0}])
    context = RunContext(tmp_path, "artifact_report", {})

    with stage_report(context, StageName.PLAYER_TRACKING) as stage:
        stage.require_artifact(
            pipeline_module.inspect_csv(
                artifact_path,
                name="player tracks",
                min_rows=1,
                required_fields={"frame_id"},
            )
        )

    artifact = _manifest(tmp_path, "artifact_report")["stages"][0]["artifacts"][0]
    assert artifact["name"] == "player tracks"
    assert artifact["status"] == "valid"
    assert artifact["details"]["row_count"] == 1


def test_empty_required_artifact_preserves_empty_stage_status(tmp_path) -> None:
    artifact_path = tmp_path / "rallies.csv"
    write_csv_rows(artifact_path, RALLY_FIELDS, [])
    context = RunContext(tmp_path, "empty_artifact", {})

    with pytest.raises(StageExecutionError, match="is empty"):
        with stage_report(context, StageName.RALLY_SEGMENTATION) as stage:
            stage.accept_legacy(0)
            stage.require_artifact(
                pipeline_module.inspect_csv(
                    artifact_path,
                    name="rally metadata",
                    min_rows=1,
                    required_fields=RALLY_FIELDS,
                )
            )

    report = context.reports[0]
    assert report.status == StageStatus.EMPTY
    assert report.exit_code == 0
    assert report.artifacts[0].details["row_count"] == 0


def test_quality_rejection_is_distinct_from_empty_and_failed(tmp_path) -> None:
    context = RunContext(tmp_path, "quality_rejection", {})

    with pytest.raises(StageExecutionError, match="no Usable Rally"):
        with stage_report(context, StageName.RALLY_SEGMENTATION) as stage:
            stage.accept_legacy(0)
            stage.reject("no Usable Rally accepted: no_active_play_evidence=1")

    report = context.reports[0]
    assert report.status == StageStatus.REJECTED
    assert report.exit_code == 0


def test_nonzero_legacy_exit_code_is_recorded_and_raised(tmp_path) -> None:
    context = RunContext(tmp_path, "failure", {})

    with pytest.raises(StageExecutionError, match="exit code 2"):
        with stage_report(context, StageName.PLAYER_TRACKING) as stage:
            stage.accept_legacy(2, operation="player tracking")

    report = context.reports[0]
    assert report.status == StageStatus.FAILED
    assert report.exit_code == 2
    assert "player tracking" in report.message
    manifest_stage = _manifest(tmp_path, "failure")["stages"][0]
    assert manifest_stage["status"] == "failed"
    assert manifest_stage["exit_code"] == 2


def test_system_exit_is_recorded_as_failed_stage(tmp_path) -> None:
    context = RunContext(tmp_path, "system_exit", {})

    with pytest.raises(SystemExit):
        with stage_report(context, StageName.RALLY_SEGMENTATION):
            raise SystemExit(3)

    report = context.reports[0]
    assert report.status == StageStatus.FAILED
    assert report.exit_code == 3


def test_resume_retries_failed_stage_instead_of_treating_it_as_completed(tmp_path) -> None:
    context = RunContext(tmp_path, "resume_failure", {})
    with stage_report(context, StageName.RALLY_SEGMENTATION) as stage:
        stage.accept_legacy(0)
    with pytest.raises(StageExecutionError):
        with stage_report(context, StageName.COURT_CALIBRATION) as stage:
            stage.accept_legacy(2)

    resumed = RunContext(tmp_path, "resume_failure", {})
    completed = resumed.resume()

    assert completed == {StageName.RALLY_SEGMENTATION}
    assert [report.name for report in resumed.reports] == [StageName.RALLY_SEGMENTATION]


def test_pipeline_does_not_mark_failed_player_tracker_as_success(
    tmp_path,
    monkeypatch,
) -> None:
    project_dir = Path(__file__).resolve().parents[1]
    _stub_successful_pipeline(monkeypatch, project_dir)
    monkeypatch.setattr(pipeline_module, "track_players", lambda **_kwargs: 2)

    with pytest.raises(StageExecutionError, match="player tracking"):
        pipeline_module.run_pipeline(
            input_video=tmp_path / "match.mp4",
            run_id="player_failure",
            root=tmp_path,
            skip_visualize=True,
            skip_demo=True,
        )

    stages = _manifest(tmp_path, "player_failure")["stages"]
    assert [stage["status"] for stage in stages] == [
        "success",
        "success",
        "success",
        "failed",
    ]
    assert stages[-1]["name"] == "player_tracking"
    assert stages[-1]["exit_code"] == 2


@pytest.mark.parametrize(
    ("attribute", "stage_name", "skip_visualize"),
    [
        ("track_shuttle", "shuttle_tracking", True),
        ("visualize_tracking_main", "visualization", False),
        ("tactics_main", "tactical_analysis", True),
    ],
)
def test_pipeline_propagates_nonzero_exit_codes_from_later_stages(
    tmp_path,
    monkeypatch,
    attribute: str,
    stage_name: str,
    skip_visualize: bool,
) -> None:
    project_dir = Path(__file__).resolve().parents[1]
    _stub_successful_pipeline(monkeypatch, project_dir)
    monkeypatch.setattr(pipeline_module, attribute, lambda *_args, **_kwargs: 2)

    with pytest.raises(StageExecutionError, match="exit code 2"):
        pipeline_module.run_pipeline(
            input_video=tmp_path / "match.mp4",
            run_id=f"{stage_name}_failure",
            root=tmp_path,
            skip_visualize=skip_visualize,
            skip_demo=True,
        )

    stages = _manifest(tmp_path, f"{stage_name}_failure")["stages"]
    assert stages[-1]["name"] == stage_name
    assert stages[-1]["status"] == "failed"
    assert stages[-1]["exit_code"] == 2


def test_pipeline_propagates_rally_nonzero_exit_code(tmp_path, monkeypatch) -> None:
    project_dir = Path(__file__).resolve().parents[1]
    _stub_successful_pipeline(monkeypatch, project_dir)
    captured: dict[str, object] = {}

    def fail_rally(**kwargs) -> int:
        captured.update(kwargs)
        return 2

    monkeypatch.setattr(pipeline_module, "segment_rallies", fail_rally)

    with pytest.raises(StageExecutionError, match="rally segmentation"):
        pipeline_module.run_pipeline(
            input_video=tmp_path / "match.mp4",
            run_id="rally_failure",
            root=tmp_path,
            stop_after="rally",
        )

    stages = _manifest(tmp_path, "rally_failure")["stages"]
    assert len(stages) == 2
    assert stages[0]["name"] == "main_view"
    assert Path(captured["timeline_path"]).name == "main_view_timeline.json"
    assert stages[-1]["status"] == "failed"
    assert stages[-1]["exit_code"] == 2


def test_pipeline_propagates_main_view_nonzero_exit_code(tmp_path, monkeypatch) -> None:
    project_dir = Path(__file__).resolve().parents[1]
    _stub_successful_pipeline(monkeypatch, project_dir)
    monkeypatch.setattr(pipeline_module, "analyze_main_view", lambda **_kwargs: 2)

    with pytest.raises(StageExecutionError, match="main-view analysis"):
        pipeline_module.run_pipeline(
            input_video=tmp_path / "match.mp4",
            run_id="main_view_failure",
            root=tmp_path,
            stop_after="main_view",
        )

    stages = _manifest(tmp_path, "main_view_failure")["stages"]
    assert [(stage["name"], stage["status"]) for stage in stages] == [
        ("main_view", "failed")
    ]
    assert stages[0]["exit_code"] == 2


def test_pipeline_handles_partial_calibration_before_recording_success(
    tmp_path,
    monkeypatch,
) -> None:
    project_dir = Path(__file__).resolve().parents[1]
    config = load_yaml(project_dir / "configs" / "default.yaml")
    monkeypatch.setattr(pipeline_module, "load_config", lambda *_args, **_kwargs: config)
    _stub_artifact_inspection(monkeypatch)

    def segment_rallies(**kwargs) -> int:
        write_csv_rows(
            Path(kwargs["metadata_csv"]),
            RALLY_FIELDS,
            [
                {"rally_id": "001", "output_path": "rally_ok.mp4"},
                {"rally_id": "002", "output_path": "rally_rejected.mp4"},
            ],
        )
        return 0

    def calibrate_courts(*args, **_kwargs) -> int:
        write_csv_rows(
            Path(args[3]),
            COURT_CALIBRATION_SUMMARY_FIELDS,
            [
                {
                    "video_stem": "rally_ok",
                    "status": "success",
                },
                {
                    "video_stem": "rally_rejected",
                    "status": "failed",
                    "message": "line support too low",
                },
            ],
        )
        return 2

    monkeypatch.setattr(pipeline_module, "segment_rallies", segment_rallies)
    monkeypatch.setattr(pipeline_module, "calibrate_courts", calibrate_courts)

    run_dir = pipeline_module.run_pipeline(
        input_video=tmp_path / "match.mp4",
        run_id="partial_calibration",
        root=tmp_path,
        stop_after="calibrate",
    )

    stages = _manifest(tmp_path, "partial_calibration")["stages"]
    assert [stage["status"] for stage in stages] == ["success", "success", "success"]
    assert "rejected 1" in stages[-1]["message"]
    rallies = read_csv_rows(run_dir / "rallies.csv")
    assert [row["output_path"] for row in rallies] == ["rally_ok.mp4"]


def test_pipeline_records_all_calibrations_failed_before_raising(
    tmp_path,
    monkeypatch,
) -> None:
    project_dir = Path(__file__).resolve().parents[1]
    config = load_yaml(project_dir / "configs" / "default.yaml")
    monkeypatch.setattr(pipeline_module, "load_config", lambda *_args, **_kwargs: config)
    _stub_artifact_inspection(monkeypatch)
    monkeypatch.setattr(pipeline_module, "segment_rallies", lambda **_kwargs: 0)

    def calibrate_courts(*args, **_kwargs) -> int:
        write_csv_rows(
            Path(args[3]),
            COURT_CALIBRATION_SUMMARY_FIELDS,
            [{"video_stem": "rally_rejected", "status": "failed"}],
        )
        return 2

    monkeypatch.setattr(pipeline_module, "calibrate_courts", calibrate_courts)

    with pytest.raises(StageExecutionError, match="court calibration"):
        pipeline_module.run_pipeline(
            input_video=tmp_path / "match.mp4",
            run_id="calibration_failure",
            root=tmp_path,
            stop_after="calibrate",
        )

    stages = _manifest(tmp_path, "calibration_failure")["stages"]
    assert [stage["status"] for stage in stages] == ["success", "success", "failed"]
    assert stages[-1]["exit_code"] == 2


def test_pipeline_resume_after_tracking_keeps_player_config_available(
    tmp_path,
    monkeypatch,
) -> None:
    project_dir = Path(__file__).resolve().parents[1]
    _stub_successful_pipeline(monkeypatch, project_dir)

    pipeline_module.run_pipeline(
        input_video=tmp_path / "match.mp4",
        run_id="resume_after_tracking",
        root=tmp_path,
        stop_after="tracking",
        skip_visualize=True,
        skip_demo=True,
    )
    pipeline_module.run_pipeline(
        input_video=tmp_path / "match.mp4",
        run_id="resume_after_tracking",
        root=tmp_path,
        skip_visualize=True,
        skip_demo=True,
    )

    stages = _manifest(tmp_path, "resume_after_tracking")["stages"]
    assert stages[-1]["name"] == "biomechanics_analysis"
    assert stages[-1]["status"] == "success"


def test_legacy_completed_demo_cannot_silently_insert_biomechanics_stage(
    tmp_path,
    monkeypatch,
) -> None:
    project_dir = Path(__file__).resolve().parents[1]
    config = load_yaml(project_dir / "configs" / "default.yaml")
    monkeypatch.setattr(pipeline_module, "load_config", lambda *_args, **_kwargs: config)
    write_json(
        tmp_path / "runs" / "legacy_demo" / "manifest.json",
        {
            "run_id": "legacy_demo",
            "config": config,
            "stages": [
                {"name": "main_view", "status": "success"},
                {"name": "demo_rendering", "status": "success"},
            ],
        },
    )

    with pytest.raises(RuntimeError, match="predates the Biomechanics Analysis stage"):
        pipeline_module.run_pipeline(
            input_video=tmp_path / "match.mp4",
            run_id="legacy_demo",
            root=tmp_path,
        )

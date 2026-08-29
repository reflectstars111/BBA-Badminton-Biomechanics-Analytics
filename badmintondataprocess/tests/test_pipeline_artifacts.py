from __future__ import annotations

import json
from pathlib import Path

import pytest

import badminton_data_process.pipeline.run as pipeline_module
from badminton_data_process.core.config import load_yaml
from badminton_data_process.core.io import write_csv_rows, write_json
from badminton_data_process.core.run import StageExecutionError
from badminton_data_process.core.schemas import (
    COURT_CALIBRATION_SUMMARY_FIELDS,
    MAIN_VIEW_FRAME_FIELDS,
    MAIN_VIEW_QUALITY_FIELDS,
    MAIN_VIEW_SEGMENT_FIELDS,
    PLAYER_TRACK_FIELDS,
    RALLY_DECISION_FIELDS,
    RALLY_FIELDS,
)


def _write_main_view_outputs(output_dir: Path, *, accepted: bool = True) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_row = {field: "" for field in MAIN_VIEW_FRAME_FIELDS}
    frame_row.update({"sample_frame": 0, "timestamp": 0.0, "is_main_view": int(accepted)})
    write_csv_rows(
        output_dir / "main_view_frame_scores.csv",
        MAIN_VIEW_FRAME_FIELDS,
        [frame_row],
    )
    segment_rows = []
    quality_rows = []
    timeline_rows = []
    if accepted:
        segment = {field: "" for field in MAIN_VIEW_SEGMENT_FIELDS}
        segment.update(
            {
                "segment_id": "001",
                "start_frame": 0,
                "end_frame": 30,
                "label": "MAIN_VIEW",
            }
        )
        segment_rows.append(segment)
        quality = {field: "" for field in MAIN_VIEW_QUALITY_FIELDS}
        quality.update({"segment_id": "001", "accepted": 1})
        quality_rows.append(quality)
        timeline_rows.append(
            {
                "segment_id": "001",
                "start_frame": 0,
                "end_frame": 30,
                "label": "MAIN_VIEW",
            }
        )
    write_csv_rows(
        output_dir / "main_view_segments.csv",
        MAIN_VIEW_SEGMENT_FIELDS,
        segment_rows,
    )
    write_csv_rows(
        output_dir / "main_view_quality.csv",
        MAIN_VIEW_QUALITY_FIELDS,
        quality_rows,
    )
    write_csv_rows(output_dir / "rejected_segments.csv", ["segment_id"], [])
    write_json(output_dir / "main_view_timeline.json", timeline_rows)
    return 0


def _config(monkeypatch) -> None:
    project_dir = Path(__file__).resolve().parents[1]
    config = load_yaml(project_dir / "configs" / "default.yaml")
    monkeypatch.setattr(pipeline_module, "load_config", lambda *_args, **_kwargs: config)
    monkeypatch.setattr(
        pipeline_module,
        "analyze_main_view",
        lambda **kwargs: _write_main_view_outputs(Path(kwargs["output_dir"])),
    )


def _write_rally_decision(path: Path, status: str, reason: str) -> None:
    row = {field: "" for field in RALLY_DECISION_FIELDS}
    row.update(
        {
            "candidate_id": "C001",
            "start_frame": 0,
            "end_frame": 30,
            "frame_interval": "[start_frame,end_frame)",
            "status": status,
            "reason": reason,
        }
    )
    write_csv_rows(path, RALLY_DECISION_FIELDS, [row])


def _manifest(root: Path, run_id: str) -> dict[str, object]:
    return json.loads(
        (root / "runs" / run_id / "manifest.json").read_text(encoding="utf-8")
    )


def test_pipeline_honors_configured_runs_dir(tmp_path, monkeypatch) -> None:
    project_dir = Path(__file__).resolve().parents[1]
    config = load_yaml(project_dir / "configs" / "default.yaml")
    config["data"]["runs_dir"] = "research_runs"
    monkeypatch.setattr(pipeline_module, "load_config", lambda *_args, **_kwargs: config)
    monkeypatch.setattr(
        pipeline_module,
        "analyze_main_view",
        lambda **kwargs: _write_main_view_outputs(Path(kwargs["output_dir"])),
    )

    run_dir = pipeline_module.run_pipeline(
        input_video=tmp_path / "match.mp4",
        run_id="custom_layout",
        root=tmp_path,
        stop_after="main_view",
    )

    assert run_dir == tmp_path / "research_runs" / "custom_layout"
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "main_view" / "main_view_timeline.json").exists()


def test_zero_usable_rally_result_is_rejected_not_success(tmp_path, monkeypatch) -> None:
    _config(monkeypatch)

    def segment_rallies(**kwargs) -> int:
        write_csv_rows(Path(kwargs["metadata_csv"]), RALLY_FIELDS, [])
        _write_rally_decision(
            Path(kwargs["decisions_csv"]),
            "rejected",
            "no_active_play_evidence",
        )
        return 0

    monkeypatch.setattr(pipeline_module, "segment_rallies", segment_rallies)

    with pytest.raises(StageExecutionError, match="no Usable Rally accepted"):
        pipeline_module.run_pipeline(
            input_video=tmp_path / "match.mp4",
            run_id="empty_rallies",
            root=tmp_path,
            stop_after="rally",
        )

    stage = _manifest(tmp_path, "empty_rallies")["stages"][1]
    assert stage["status"] == "rejected"
    assert stage["exit_code"] == 0
    assert stage["artifacts"][0]["status"] == "valid"
    assert "no_active_play_evidence=1" in stage["message"]


def test_no_main_view_stops_pipeline_before_rally(tmp_path, monkeypatch) -> None:
    _config(monkeypatch)
    monkeypatch.setattr(
        pipeline_module,
        "analyze_main_view",
        lambda **kwargs: _write_main_view_outputs(
            Path(kwargs["output_dir"]),
            accepted=False,
        ),
    )

    def unexpected_rally(**_kwargs):
        raise AssertionError("rally segmentation must not run without Main View")

    monkeypatch.setattr(pipeline_module, "segment_rallies", unexpected_rally)

    with pytest.raises(StageExecutionError, match="accepted main-view segments.*empty"):
        pipeline_module.run_pipeline(
            input_video=tmp_path / "match.mp4",
            run_id="no_main_view",
            root=tmp_path,
            stop_after="rally",
        )

    stages = _manifest(tmp_path, "no_main_view")["stages"]
    assert [(stage["name"], stage["status"]) for stage in stages] == [
        ("main_view", "empty")
    ]


def test_player_success_code_with_missing_tracks_is_failed(tmp_path, monkeypatch) -> None:
    _config(monkeypatch)

    def segment_rallies(**kwargs) -> int:
        write_csv_rows(
            Path(kwargs["metadata_csv"]),
            RALLY_FIELDS,
            [{"rally_id": "001", "output_path": str(tmp_path / "rally.mp4")}],
        )
        _write_rally_decision(
            Path(kwargs["decisions_csv"]),
            "accepted",
            "active_play_evidence",
        )
        return 0

    def calibrate_courts(*args, **_kwargs) -> int:
        calibration_json = Path(args[1]) / "rally.json"
        calibration_json.write_text(
            json.dumps(
                {
                        "artifact_version": "2.0",
                        "validated": True,
                        "court_type": "badminton_standard",
                        "coordinate_unit": "metre",
                        "image_points_tl_tr_br_bl": [[0, 0], [1, 0], [1, 1], [0, 1]],
                        "court_points_tl_tr_br_bl": [[0, 0], [1, 0], [1, 1], [0, 1]],
                        "homography_image_to_court": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                        "quality": {},
                        "temporal_validation": {"stable_candidate_count": 3},
                }
            ),
            encoding="utf-8",
        )
        write_csv_rows(
            Path(args[3]),
            COURT_CALIBRATION_SUMMARY_FIELDS,
            [
                {
                    "video_stem": "rally",
                    "status": "success",
                    "json_path": str(calibration_json),
                }
            ],
        )
        return 0

    def track_players(**kwargs) -> int:
        write_csv_rows(
            Path(kwargs["summary_csv"]),
            ["video_stem", "status", "track_rows"],
            [{"video_stem": "rally", "status": "success", "track_rows": 0}],
        )
        return 0

    monkeypatch.setattr(pipeline_module, "segment_rallies", segment_rallies)
    monkeypatch.setattr(pipeline_module, "calibrate_courts", calibrate_courts)
    monkeypatch.setattr(pipeline_module, "track_players", track_players)
    monkeypatch.setattr(
        pipeline_module,
        "inspect_video",
        lambda *_args, **_kwargs: pipeline_module.inspect_csv(
            tmp_path / "synthetic_valid.csv",
            name="unused",
        ),
    )
    write_csv_rows(tmp_path / "synthetic_valid.csv", ["ok"], [{"ok": 1}])

    with pytest.raises(StageExecutionError, match="player tracks.*missing"):
        pipeline_module.run_pipeline(
            input_video=tmp_path / "match.mp4",
            run_id="missing_player_tracks",
            root=tmp_path,
            stop_after="tracking",
        )

    stages = _manifest(tmp_path, "missing_player_tracks")["stages"]
    assert stages[-1]["name"] == "player_tracking"
    assert stages[-1]["status"] == "failed"
    assert stages[-1]["exit_code"] == 0
    assert stages[-1]["artifacts"][0]["status"] == "missing"

from __future__ import annotations

from pathlib import Path

import badminton_data_process.webui.adapter as adapter_module
from badminton_data_process.core.config import load_config
from badminton_data_process.core.config_schema import parse_config
from badminton_data_process.core.io import write_json
from badminton_data_process.webui.adapter import (
    MODE_CONFIGS,
    RunSpecification,
    WebAnalysisRequest,
    read_run_state,
    resolve_mode_config,
    submit_run,
    submit_web_analysis,
)


def test_webui_submission_uses_pipeline_interface_without_reorchestration(tmp_path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_pipeline(**kwargs):
        captured.update(kwargs)
        return tmp_path / "runs" / "ui_run"

    monkeypatch.setattr(adapter_module, "run_pipeline", fake_run_pipeline)
    spec = RunSpecification(
        input_video=tmp_path / "match.mp4",
        run_id="ui_run",
        root=tmp_path,
        stop_after="tracking",
        skip_demo=True,
    )
    result = submit_run(spec)

    assert result == tmp_path / "runs" / "ui_run"
    assert captured["input_video"] == spec.input_video
    assert captured["run_id"] == "ui_run"
    assert captured["stop_after"] == "tracking"
    assert captured["skip_demo"] is True


def test_run_state_preserves_manifest_status_and_history(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "failed_run"
    run_dir.mkdir(parents=True)
    marker = run_dir / "keep.me"
    marker.write_text("historical artifact", encoding="utf-8")
    write_json(
        run_dir / "manifest.json",
        {
            "run_id": "failed_run",
            "run_dir": str(run_dir),
            "stages": [
                {
                    "name": "court_calibration",
                    "status": "rejected",
                    "artifacts": [
                        {"name": "candidate", "path": str(marker), "status": "valid"}
                    ],
                }
            ],
        },
    )

    state = read_run_state(tmp_path, "failed_run")

    assert state["status"] == "rejected"
    assert state["artifacts"][0]["path"] == str(marker)
    assert marker.read_text(encoding="utf-8") == "historical artifact"


def test_all_webui_mode_combinations_resolve_reviewed_configs() -> None:
    root = Path(__file__).resolve().parents[1]

    resolved = {
        key: resolve_mode_config(root, key[0], key[1]).relative_to(root)
        for key in MODE_CONFIGS
    }

    assert resolved == MODE_CONFIGS
    assert "lindan" not in str(resolved[("uncut", "low")]).lower()
    parsed = {
        key: parse_config(load_config(root / path, root=root))
        for key, path in resolved.items()
    }
    assert parsed[("uncut", "low")].court_calibration.detector == "hough_low_angle"
    assert parsed[("clipped", "low")].court_calibration.max_out_of_bounds_ratio == 0.25


def test_webui_full_submission_builds_report_without_ui_orchestration(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "runs" / "web_001"
    run_dir.mkdir(parents=True)
    config = tmp_path / "configs" / "production" / "full_video_gpu.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("demo_rendering:\n  enabled: true\n", encoding="utf-8")
    monkeypatch.setitem(MODE_CONFIGS, ("uncut", "overhead"), config.relative_to(tmp_path))
    captured: dict[str, object] = {}

    def fake_full(input_video, **kwargs):
        captured.update({"input_video": input_video, **kwargs})
        return run_dir, {"status": "success"}

    monkeypatch.setattr(adapter_module, "run_full_analysis", fake_full)
    monkeypatch.setattr(adapter_module, "build_web_report", lambda path: {"run_dir": str(path)})
    request = WebAnalysisRequest(
        input_video=tmp_path / "upload.mp4",
        input_kind="uncut",
        view_kind="overhead",
        run_id="web_001",
        root=tmp_path,
    )

    result_dir, summary, report = submit_web_analysis(request)

    assert result_dir == run_dir
    assert summary["status"] == "success"
    assert report["run_dir"] == str(run_dir)
    assert captured["config_path"] == config
    assert captured["run_id"] == "web_001"


def test_webui_manual_corners_are_forwarded_as_reviewed_config_override(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "runs" / "manual"
    run_dir.mkdir(parents=True)
    config = tmp_path / "configs" / "webui" / "low.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("court_calibration:\n  detector: hough_low_angle\n", encoding="utf-8")
    monkeypatch.setitem(MODE_CONFIGS, ("clipped", "low"), config.relative_to(tmp_path))
    captured: dict[str, object] = {}

    def fake_full(input_video, **kwargs):
        captured.update({"input_video": input_video, **kwargs})
        return run_dir, {"status": "success"}

    monkeypatch.setattr(adapter_module, "run_full_analysis", fake_full)
    monkeypatch.setattr(adapter_module, "build_web_report", lambda path: {"run_dir": str(path)})
    points = [0.1, 0.2, 0.8, 0.2, 1.2, 0.9, -0.2, 0.9]

    submit_web_analysis(
        WebAnalysisRequest(
            input_video=tmp_path / "upload.mp4",
            input_kind="clipped",
            view_kind="low",
            run_id="manual",
            root=tmp_path,
            manual_reference_points=points,
        )
    )

    assert captured["config_overrides"] == {
        "court_calibration": {
            "reference_points": points,
            "max_out_of_bounds_ratio": 1.0,
        }
    }

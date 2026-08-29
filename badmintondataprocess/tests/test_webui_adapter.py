from __future__ import annotations

from pathlib import Path

import badminton_data_process.webui.adapter as adapter_module
from badminton_data_process.core.io import write_json
from badminton_data_process.webui.adapter import RunSpecification, read_run_state, submit_run


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

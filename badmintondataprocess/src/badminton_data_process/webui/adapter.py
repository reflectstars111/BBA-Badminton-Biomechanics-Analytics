from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from badminton_data_process.core.io import read_json
from badminton_data_process.core.paths import RunLayout
from badminton_data_process.pipeline.run import run_pipeline


@dataclass(frozen=True, slots=True)
class RunSpecification:
    """UI-neutral input with the same semantics as ``bdp pipeline run``."""

    input_video: Path
    run_id: str
    root: Path
    config_path: Path | None = None
    stop_after: str | None = None
    skip_visualize: bool = False
    skip_demo: bool = False
    force: bool = False
    runs_dir: Path | None = None


def submit_run(specification: RunSpecification) -> Path:
    """Call the single research pipeline Interface; no UI-side orchestration."""

    return run_pipeline(
        input_video=specification.input_video,
        run_id=specification.run_id,
        config_path=specification.config_path,
        root=specification.root,
        stop_after=specification.stop_after,
        skip_visualize=specification.skip_visualize,
        skip_demo=specification.skip_demo,
        force=specification.force,
        runs_dir=specification.runs_dir,
    )


def read_run_state(root: Path, run_id: str, runs_dir: Path | None = None) -> dict[str, Any]:
    """Read status and browseable artifacts without mutating historical runs."""

    layout = RunLayout.create(root, run_id, runs_dir or "runs")
    if not layout.manifest_json.is_file():
        return {
            "run_id": run_id,
            "run_dir": str(layout.run_dir),
            "status": "missing",
            "capability": "near_only; dual-side observation is experimental",
            "stages": [],
            "artifacts": [],
        }
    manifest = read_json(layout.manifest_json)
    stages = list(manifest.get("stages", []))
    artifacts = [
        artifact
        for stage in stages
        for artifact in stage.get("artifacts", [])
    ]
    status = stages[-1].get("status", "pending") if stages else "pending"
    return {
        "run_id": manifest.get("run_id", run_id),
        "run_dir": manifest.get("run_dir", str(layout.run_dir)),
        "status": status,
        "capability": "near_only; dual-side observation is experimental",
        "stages": stages,
        "artifacts": artifacts,
    }

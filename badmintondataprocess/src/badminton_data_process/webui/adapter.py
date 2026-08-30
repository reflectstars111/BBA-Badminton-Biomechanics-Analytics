from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from badminton_data_process.core.io import read_json
from badminton_data_process.core.paths import RunLayout
from badminton_data_process.pipeline.full import run_full_analysis
from badminton_data_process.pipeline.run import run_pipeline
from badminton_data_process.webui.reporting import build_web_report


INPUT_KINDS = ("uncut", "clipped")
VIEW_KINDS = ("overhead", "low")
MODE_CONFIGS = {
    ("uncut", "overhead"): Path("configs/production/full_video_gpu.yaml"),
    ("uncut", "low"): Path("configs/webui/low_angle_gpu.yaml"),
    ("clipped", "overhead"): Path("configs/webui/clipped_overhead_gpu.yaml"),
    ("clipped", "low"): Path("configs/webui/clipped_low_angle_gpu.yaml"),
}


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


@dataclass(frozen=True, slots=True)
class WebAnalysisRequest:
    input_video: Path
    input_kind: str
    view_kind: str
    run_id: str
    root: Path
    runs_dir: Path | None = None
    manual_reference_points: list[float] | None = None


def resolve_mode_config(root: Path, input_kind: str, view_kind: str) -> Path:
    """Map explicit user choices to a reviewed pipeline profile."""

    if input_kind not in INPUT_KINDS:
        raise ValueError(f"Unsupported input kind: {input_kind!r}")
    if view_kind not in VIEW_KINDS:
        raise ValueError(f"Unsupported view kind: {view_kind!r}")
    config_path = Path(root) / MODE_CONFIGS[(input_kind, view_kind)]
    if not config_path.is_file():
        raise FileNotFoundError(f"WebUI mode configuration does not exist: {config_path}")
    return config_path


def submit_web_analysis(
    request: WebAnalysisRequest,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Run the one-click workflow and build its detailed UI-neutral report."""

    config_path = resolve_mode_config(request.root, request.input_kind, request.view_kind)
    overrides: dict[str, Any] | None = None
    if request.manual_reference_points is not None:
        if len(request.manual_reference_points) != 8:
            raise ValueError("manual court reference must contain four normalized x,y pairs")
        out_of_bounds = any(
            value < 0.0 or value > 1.0
            for value in request.manual_reference_points
        )
        overrides = {
            "court_calibration": {
                "reference_points": request.manual_reference_points,
                "max_out_of_bounds_ratio": 0.25 if out_of_bounds else 0.0,
            }
        }
    run_dir, summary = run_full_analysis(
        request.input_video,
        run_id=request.run_id,
        config_path=config_path,
        root=request.root,
        runs_dir=request.runs_dir,
        config_overrides=overrides,
        progress_callback=progress_callback,
    )
    report = build_web_report(run_dir)
    return run_dir, summary, report


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

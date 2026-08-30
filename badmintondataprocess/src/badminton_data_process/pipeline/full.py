from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from badminton_data_process.core.config import load_config
from badminton_data_process.core.config_schema import PipelineConfig, parse_config
from badminton_data_process.core.io import read_csv_rows, read_json, write_json
from badminton_data_process.core.paths import RunLayout, discover_project_root
from badminton_data_process.core.run import make_run_id
from badminton_data_process.pipeline.run import run_pipeline


DEFAULT_FULL_CONFIG = Path("configs/production/full_video_gpu.yaml")


@dataclass(frozen=True, slots=True)
class FullAnalysisRequest:
    project_root: Path
    input_video: Path
    config_path: Path
    config: dict[str, Any]
    parsed_config: PipelineConfig


def _resolve_project_path(path: str | Path, project_root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else project_root / candidate


def _require_cuda_torch(component: str) -> None:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001 - turn dependency failures into preflight errors
        raise RuntimeError(f"{component} requires CUDA PyTorch, but torch cannot import: {exc}") from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"{component} requires CUDA, but torch.cuda.is_available() is false. "
            "Run this workflow from the good-badminton Conda environment."
        )


def validate_full_analysis_environment(
    input_video: Path,
    cfg: PipelineConfig,
    project_root: Path,
) -> Path:
    """Fail before a long run when its source, models, or GPU runtime are unusable."""

    resolved_input = input_video.expanduser().resolve()
    if not resolved_input.is_file():
        raise FileNotFoundError(f"Input video does not exist: {resolved_input}")

    try:
        import cv2
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"OpenCV cannot import: {exc}") from exc
    capture = cv2.VideoCapture(str(resolved_input))
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"Input video cannot be decoded: {resolved_input}")

    player = cfg.player_tracking
    if player.detector == "rtmpose":
        try:
            import rtmlib  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "RTMPose is enabled, but rtmlib cannot import. Install it in the "
                "good-badminton Conda environment."
            ) from exc
        if player.rtmpose_device == "cuda":
            _require_cuda_torch("RTMPose")
            if player.rtmpose_backend == "onnxruntime":
                try:
                    import onnxruntime as ort
                except Exception as exc:  # noqa: BLE001
                    raise RuntimeError(
                        "RTMPose CUDA requires onnxruntime-gpu, but onnxruntime cannot import"
                    ) from exc
                if "CUDAExecutionProvider" not in ort.get_available_providers():
                    raise RuntimeError(
                        "RTMPose CUDA requires ONNX Runtime CUDAExecutionProvider. "
                        "Remove CPU-only onnxruntime and install onnxruntime-gpu."
                    )

    shuttle = cfg.shuttle_tracking
    if shuttle.model == "tracknet":
        weights = _resolve_project_path(shuttle.tracknet_weights, project_root)
        if not weights.is_file():
            raise FileNotFoundError(f"TrackNet weights do not exist: {weights}")
        if shuttle.tracknet_device == "cuda":
            _require_cuda_torch("TrackNet")

    return resolved_input


def prepare_full_analysis(
    input_video: Path,
    *,
    config_path: Path | None = None,
    root: Path | None = None,
    config_overrides: dict[str, Any] | None = None,
) -> FullAnalysisRequest:
    """Resolve and validate everything needed before creating a run directory."""

    project_root = (root or discover_project_root()).resolve()
    selected_config = config_path or project_root / DEFAULT_FULL_CONFIG
    if not selected_config.is_absolute():
        selected_config = project_root / selected_config
    selected_config = selected_config.resolve()
    config = load_config(selected_config, overrides=config_overrides, root=project_root)
    cfg = parse_config(config)
    resolved_input = validate_full_analysis_environment(input_video, cfg, project_root)
    return FullAnalysisRequest(
        project_root=project_root,
        input_video=resolved_input,
        config_path=selected_config,
        config=config,
        parsed_config=cfg,
    )


def _sum_int(rows: list[dict[str, str]], field: str) -> int:
    total = 0
    for row in rows:
        try:
            total += int(float(row.get(field, "0") or 0))
        except (TypeError, ValueError):
            continue
    return total


def validate_resume_identity(
    layout: RunLayout,
    input_video: Path,
    config: dict[str, Any],
    *,
    force: bool,
) -> None:
    """Prevent a named run from mixing artifacts from another source/config."""

    if force or not layout.manifest_json.is_file():
        return
    manifest = read_json(layout.manifest_json)
    if manifest.get("config") != config:
        raise RuntimeError(
            f"Run {layout.run_id!r} was created with a different configuration. "
            "Choose a new --run-id or pass --force to rerun every stage."
        )
    main_view_stage = next(
        (stage for stage in manifest.get("stages", []) if stage.get("name") == "main_view"),
        None,
    )
    recorded_inputs = list(main_view_stage.get("inputs", [])) if main_view_stage else []
    if not recorded_inputs:
        raise RuntimeError(
            f"Run {layout.run_id!r} has no recorded source identity. "
            "Choose a new --run-id or pass --force."
        )
    recorded_source = Path(recorded_inputs[0]).resolve()
    if recorded_source != input_video.resolve():
        raise RuntimeError(
            f"Run {layout.run_id!r} belongs to another source video: {recorded_source}. "
            "Choose a new --run-id or pass --force to rerun every stage."
        )


def build_analysis_summary(run_dir: Path) -> dict[str, Any]:
    """Build a compact hand-off artifact for users and downstream automation."""

    manifest_path = run_dir / "manifest.json"
    manifest = read_json(manifest_path)
    config = manifest.get("config", {})
    demo_name = (
        config.get("demo_rendering", {}).get("output_filename")
        or "badminton_analysis_demo.mp4"
    )
    main_view_rows = read_csv_rows(run_dir / "main_view" / "main_view_segments.csv")
    decisions = read_csv_rows(run_dir / "rally_decisions.csv")
    rallies = read_csv_rows(run_dir / "rallies.csv")
    calibrations = read_csv_rows(run_dir / "annotations" / "court_calibration_summary.csv")
    players = read_csv_rows(run_dir / "annotations" / "player_tracking_summary.csv")
    shuttles = read_csv_rows(run_dir / "annotations" / "shuttle_tracking_summary.csv")
    shuttle_smoothing_path = run_dir / "annotations" / "shuttle_smoothing_summary.csv"
    shuttle_smoothing = read_csv_rows(shuttle_smoothing_path)
    demo_path = run_dir / "outputs" / "demo" / str(demo_name)
    stages = list(manifest.get("stages", []))
    stage_success = bool(stages) and all(stage.get("status") == "success" for stage in stages)

    return {
        "run_id": manifest.get("run_id", run_dir.name),
        "status": "success" if stage_success and demo_path.is_file() else "incomplete",
        "run_dir": str(run_dir),
        "counts": {
            "main_view_segments": len(main_view_rows),
            "rally_candidates_accepted": sum(row.get("status") == "accepted" for row in decisions),
            "rally_candidates_rejected": sum(row.get("status") == "rejected" for row in decisions),
            "usable_rallies_after_calibration": len(rallies),
            "calibrations_successful": sum(row.get("status") == "success" for row in calibrations),
            "calibrations_rejected": sum(row.get("status") != "success" for row in calibrations),
            "player_track_rows": _sum_int(players, "track_rows"),
            "valid_pose_rows": _sum_int(players, "pose_rows"),
            "shuttle_track_rows": _sum_int(shuttles, "track_rows"),
            "shuttle_visible_rows": _sum_int(shuttles, "visible_rows"),
            "shuttle_interpolated_rows": _sum_int(shuttles, "interpolated_rows"),
            "shuttle_smoothed_valid_rows": _sum_int(
                shuttle_smoothing, "smoothed_valid_rows"
            ),
            "shuttle_gap_filled_rows": _sum_int(shuttle_smoothing, "gap_filled_rows"),
        },
        "stages": [
            {
                "name": stage.get("name", ""),
                "status": stage.get("status", ""),
                "duration_seconds": stage.get("duration_seconds", 0.0),
                "message": stage.get("message", ""),
            }
            for stage in stages
        ],
        "outputs": {
            "analysis_video": str(demo_path),
            "cleaned_rallies_csv": str(run_dir / "rallies.csv"),
            "cleaned_rallies_dir": str(run_dir / "rallies"),
            "cleaned_rally_videos": [row.get("output_path", "") for row in rallies],
            "court_calibrations": str(run_dir / "annotations" / "court_calibration"),
            "player_tracks": str(run_dir / "annotations" / "player_tracks_smoothed.csv"),
            "shuttle_tracks": str(run_dir / "annotations" / "shuttle_tracks_smoothed.csv"),
            "shuttle_smoothing_summary": str(shuttle_smoothing_path),
            "tracking_charts": str(run_dir / "outputs" / "tracking_charts"),
            "tactical_summary": str(run_dir / "outputs" / "tactics" / "tactics_summary.csv"),
            "manifest": str(manifest_path),
        },
    }


def run_full_analysis(
    input_video: Path,
    *,
    run_id: str | None = None,
    config_path: Path | None = None,
    root: Path | None = None,
    runs_dir: Path | None = None,
    force: bool = False,
    config_overrides: dict[str, Any] | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> tuple[Path, dict[str, Any]]:
    request = prepare_full_analysis(
        input_video,
        config_path=config_path,
        root=root,
        config_overrides=config_overrides,
    )
    resolved_run_id = run_id or make_run_id("full_analysis")
    layout = RunLayout.create(
        request.project_root,
        resolved_run_id,
        runs_dir if runs_dir is not None else request.parsed_config.data.runs_dir,
    )
    validate_resume_identity(
        layout,
        request.input_video,
        request.config,
        force=force,
    )

    run_dir = run_pipeline(
        input_video=request.input_video,
        run_id=resolved_run_id,
        config_path=request.config_path,
        root=request.project_root,
        force=force,
        runs_dir=runs_dir,
        config_overrides=config_overrides,
        progress_callback=progress_callback,
    )
    summary = build_analysis_summary(run_dir)
    write_json(run_dir / "analysis_summary.json", summary)
    return run_dir, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "One-command raw broadcast workflow: clean main views and rallies, calibrate "
            "courts, run GPU pose/TrackNet, analyse trajectories, and render one video."
        )
    )
    parser.add_argument("input", type=Path, help="Uncleaned source broadcast video.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--runs-dir", type=Path, default=None)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate the input, configuration, models, and GPU runtime without starting a run.",
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.preflight_only:
        request = prepare_full_analysis(args.input, config_path=args.config)
        cfg = request.parsed_config
        print("Full analysis preflight: success")
        print(f"Input video: {request.input_video}")
        print(f"Configuration: {request.config_path}")
        print(
            "Player tracking: "
            f"{cfg.player_tracking.detector} / {cfg.player_tracking.rtmpose_device}"
        )
        print(
            "Shuttle tracking: "
            f"{cfg.shuttle_tracking.model} / {cfg.shuttle_tracking.tracknet_device}"
        )
        return 0
    run_dir, summary = run_full_analysis(
        args.input,
        run_id=args.run_id,
        config_path=args.config,
        runs_dir=args.runs_dir,
        force=args.force,
    )
    print(f"Full analysis status: {summary['status']}")
    print(f"Run directory: {run_dir}")
    print(f"Analysis video: {summary['outputs']['analysis_video']}")
    print(f"Analysis summary: {run_dir / 'analysis_summary.json'}")
    return 0 if summary["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())

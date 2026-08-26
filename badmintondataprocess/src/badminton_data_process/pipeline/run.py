from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from badminton_data_process.calibration.court import calibrate_courts
from badminton_data_process.core.config import load_config
from badminton_data_process.core.config_schema import parse_config
from badminton_data_process.core.io import ensure_dir
from badminton_data_process.core.paths import discover_project_root
from badminton_data_process.core.run import RunContext, make_run_id, stage_report
from badminton_data_process.core.schemas import StageName
from badminton_data_process.rally.segmentation import segment_rallies
from badminton_data_process.smoothing.trajectory import smooth_trajectory
from badminton_data_process.tracking.player.tracking import track_players
from badminton_data_process.tracking.shuttle.tracking import track_shuttle
from badminton_data_process.visualization.tracking import main as visualize_tracking_main


def run_pipeline(
    input_video: Path,
    run_id: str | None = None,
    config_path: Path | None = None,
    root: Path | None = None,
    stop_after: str | None = None,
    skip_visualize: bool = False,
    force: bool = False,
) -> Path:
    project_root = root or discover_project_root()
    config = load_config(config_path, root=project_root)
    cfg = parse_config(config)
    context = RunContext(project_root, run_id or make_run_id("pipeline"), config)
    context.ensure()
    completed: set[StageName] = set() if force else context.resume()

    run_dir = context.run_dir
    annotations_dir = ensure_dir(run_dir / "annotations")
    outputs_dir = ensure_dir(run_dir / "outputs")
    rallies_dir = ensure_dir(run_dir / "rallies")
    calibration_dir = ensure_dir(annotations_dir / "court_calibration")
    calibration_preview_dir = ensure_dir(outputs_dir / "court_calibration_debug")

    rallies_csv = run_dir / "rallies.csv"
    calibration_summary_csv = annotations_dir / "court_calibration_summary.csv"
    player_csv = annotations_dir / "player_tracks.csv"
    player_summary_csv = annotations_dir / "player_tracking_summary.csv"
    shuttle_csv = annotations_dir / "shuttle_tracks.csv"
    shuttle_summary_csv = annotations_dir / "shuttle_tracking_summary.csv"
    player_smoothed_csv = annotations_dir / "player_tracks_smoothed.csv"
    shuttle_smoothed_csv = annotations_dir / "shuttle_tracks_smoothed.csv"
    player_smoothing_summary = annotations_dir / "player_smoothing_summary.csv"
    shuttle_smoothing_summary = annotations_dir / "shuttle_smoothing_summary.csv"

    if StageName.RALLY_SEGMENTATION not in completed:
        rally_cfg = cfg.rally_segmentation
        with stage_report(
            context,
            StageName.RALLY_SEGMENTATION,
            inputs=[str(input_video)],
            outputs=[str(rallies_dir), str(rallies_csv)],
            parameters=asdict(rally_cfg),
        ):
            segment_rallies(
                input_path=input_video,
                output_dir=rallies_dir,
                metadata_csv=rallies_csv,
                sample_every=rally_cfg.sample_every,
                min_rally_seconds=rally_cfg.min_rally_seconds,
                max_rally_seconds=rally_cfg.max_rally_seconds,
                max_gap_seconds=rally_cfg.max_gap_seconds,
                min_motion_score=rally_cfg.min_motion_score,
                max_motion_score=rally_cfg.max_motion_score,
                min_center_green_ratio=rally_cfg.min_center_green_ratio,
                min_bottom_green_ratio=rally_cfg.min_bottom_green_ratio,
                min_line_ratio=rally_cfg.min_line_ratio,
                min_top_green_ratio=rally_cfg.min_top_green_ratio,
                min_middle_green_ratio=rally_cfg.min_middle_green_ratio,
                max_left_right_green_diff=rally_cfg.max_left_right_green_diff,
                min_top_dark_ratio=rally_cfg.min_top_dark_ratio,
                min_middle_edge_ratio=rally_cfg.min_middle_edge_ratio,
                pad_before_seconds=rally_cfg.pad_before_seconds,
                pad_after_seconds=rally_cfg.pad_after_seconds,
                max_pre_context_seconds=rally_cfg.max_pre_context_seconds,
                max_post_context_seconds=rally_cfg.max_post_context_seconds,
                allowed_context_drop_samples=rally_cfg.allowed_context_drop_samples,
                overwrite=True,
                scoreboard_score_roi=(
                    tuple(rally_cfg.scoreboard_score_roi)
                    if rally_cfg.scoreboard_score_roi
                    else None
                ),
                scoreboard_context_roi=(
                    tuple(rally_cfg.scoreboard_context_roi)
                    if rally_cfg.scoreboard_context_roi
                    else None
                ),
                scoreboard_max_lag_seconds=rally_cfg.scoreboard_max_lag_seconds,
            )
    if stop_after == "rally":
        context.write_manifest()
        return run_dir

    if StageName.COURT_CALIBRATION not in completed:
        calibration_cfg = cfg.court_calibration
        with stage_report(
            context,
            StageName.COURT_CALIBRATION,
            inputs=[str(rallies_csv)],
            outputs=[str(calibration_dir), str(calibration_summary_csv)],
            parameters=asdict(calibration_cfg),
        ):
            calibrate_courts(
                rallies_csv,
                calibration_dir,
                calibration_preview_dir,
                calibration_summary_csv,
                reference_points=calibration_cfg.reference_points,
                min_line_support=calibration_cfg.min_line_support,
            )
    if stop_after == "calibrate":
        context.write_manifest()
        return run_dir

    if StageName.PLAYER_TRACKING not in completed:
        player_cfg = cfg.player_tracking
        with stage_report(
            context,
            StageName.PLAYER_TRACKING,
            inputs=[str(rallies_csv), str(calibration_dir)],
            outputs=[str(player_csv), str(player_summary_csv)],
            parameters=asdict(player_cfg),
        ):
            track_players(
                input_path=rallies_csv,
                calibration_dir=calibration_dir,
                output_csv=player_csv,
                summary_csv=player_summary_csv,
                debug_dir=ensure_dir(outputs_dir / "player_tracking_debug"),
                detector=player_cfg.detector,
                yolo_model_name=player_cfg.yolo_model,
                yolo_confidence=player_cfg.yolo_confidence,
                yolo_image_size=player_cfg.yolo_image_size,
                near_max_track_distance=player_cfg.near_max_track_distance,
                far_max_track_distance=player_cfg.far_max_track_distance,
                near_max_missing_frames=player_cfg.near_max_missing_frames,
                far_max_missing_frames=player_cfg.far_max_missing_frames,
                role_half_tolerance=player_cfg.role_half_tolerance,
            )

    if StageName.SHUTTLE_TRACKING not in completed:
        shuttle_cfg = cfg.shuttle_tracking
        with stage_report(
            context,
            StageName.SHUTTLE_TRACKING,
            inputs=[str(rallies_csv), str(calibration_dir)],
            outputs=[str(shuttle_csv), str(shuttle_summary_csv)],
            parameters=asdict(shuttle_cfg),
        ):
            track_shuttle(
                input_path=rallies_csv,
                calibration_dir=calibration_dir,
                output_csv=shuttle_csv,
                summary_csv=shuttle_summary_csv,
                debug_dir=ensure_dir(outputs_dir / "shuttle_tracking_debug"),
                diff_threshold=shuttle_cfg.diff_threshold,
                max_jump=shuttle_cfg.max_jump,
                max_missing_frames=shuttle_cfg.max_missing_frames,
                min_brightness=shuttle_cfg.min_brightness,
                min_candidate_area=shuttle_cfg.min_candidate_area,
                max_candidate_area=shuttle_cfg.max_candidate_area,
                max_candidate_size=shuttle_cfg.max_candidate_size,
                direction_weight=shuttle_cfg.direction_weight,
                speed_weight=shuttle_cfg.speed_weight,
                model=shuttle_cfg.model,
                tracknet_weights=shuttle_cfg.tracknet_weights,
            )
    if stop_after == "tracking":
        context.write_manifest()
        return run_dir

    if StageName.TRAJECTORY_SMOOTHING not in completed:
        smooth_cfg = cfg.smoothing
        with stage_report(
            context,
            StageName.TRAJECTORY_SMOOTHING,
            inputs=[str(player_csv), str(shuttle_csv)],
            outputs=[str(player_smoothed_csv), str(shuttle_smoothed_csv)],
            parameters=asdict(smooth_cfg),
        ):
            smooth_trajectory(
                player_csv,
                player_smoothed_csv,
                player_smoothing_summary,
                smooth_cfg.min_confidence,
                smooth_cfg.max_gap_frames,
                smooth_cfg.window_size,
                smooth_cfg.ema_alpha,
            )
            smooth_trajectory(
                shuttle_csv,
                shuttle_smoothed_csv,
                shuttle_smoothing_summary,
                smooth_cfg.min_confidence,
                smooth_cfg.max_gap_frames,
                smooth_cfg.window_size,
                smooth_cfg.ema_alpha,
            )

    if not skip_visualize and StageName.VISUALIZATION not in completed:
        chart_dir = ensure_dir(outputs_dir / "tracking_charts")
        with stage_report(
            context,
            StageName.VISUALIZATION,
            inputs=[str(shuttle_summary_csv), str(shuttle_smoothed_csv), str(player_smoothed_csv)],
            outputs=[str(chart_dir)],
            parameters={},
        ):
            visualize_tracking_main(
                [
                    "--shuttle-summary-csv",
                    str(shuttle_summary_csv),
                    "--player-summary-csv",
                    str(player_summary_csv),
                    "--shuttle-track-csv",
                    str(shuttle_smoothed_csv),
                    "--shuttle-smoothing-summary-csv",
                    str(shuttle_smoothing_summary),
                    "--player-track-csv",
                    str(player_smoothed_csv),
                    "--player-smoothing-summary-csv",
                    str(player_smoothing_summary),
                    "--output-dir",
                    str(chart_dir),
                ]
            )

    context.write_manifest()
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the end-to-end badminton data pipeline.")
    parser.add_argument("input", type=Path, help="Raw match video path.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--stop-after", choices=["rally", "calibrate", "tracking"], default=None)
    parser.add_argument("--skip-visualize", action="store_true")
    parser.add_argument("--force", action="store_true", help="Ignore existing manifest and re-run all stages.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = run_pipeline(
        input_video=args.input,
        run_id=args.run_id,
        config_path=args.config,
        stop_after=args.stop_after,
        skip_visualize=args.skip_visualize,
        force=args.force,
    )
    print(f"Pipeline run directory: {run_dir}")
    return 0

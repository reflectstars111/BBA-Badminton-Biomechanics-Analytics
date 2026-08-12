from __future__ import annotations

import argparse
from pathlib import Path

from badminton_data_process.calibration.court import calibrate_courts
from badminton_data_process.core.config import load_config
from badminton_data_process.core.io import ensure_dir
from badminton_data_process.core.paths import discover_project_root
from badminton_data_process.core.run import RunContext, make_run_id, stage_report
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
) -> Path:
    project_root = root or discover_project_root()
    config = load_config(config_path, root=project_root)
    context = RunContext(project_root, run_id or make_run_id("pipeline"), config)
    context.ensure()

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

    rally_cfg = config.get("rally_segmentation", {})
    with stage_report(
        context,
        "rally_segmentation",
        inputs=[str(input_video)],
        outputs=[str(rallies_dir), str(rallies_csv)],
        parameters=rally_cfg,
    ):
        segment_rallies(
            input_path=input_video,
            output_dir=rallies_dir,
            metadata_csv=rallies_csv,
            sample_every=int(rally_cfg.get("sample_every", 15)),
            min_rally_seconds=float(rally_cfg.get("min_rally_seconds", 4.0)),
            max_rally_seconds=float(rally_cfg.get("max_rally_seconds", 45.0)),
            max_gap_seconds=float(rally_cfg.get("max_gap_seconds", 3.0)),
            min_motion_score=float(rally_cfg.get("min_motion_score", 0.01)),
            max_motion_score=float(rally_cfg.get("max_motion_score", 0.16)),
            min_center_green_ratio=float(rally_cfg.get("min_center_green_ratio", 0.22)),
            min_bottom_green_ratio=float(rally_cfg.get("min_bottom_green_ratio", 0.36)),
            min_line_ratio=float(rally_cfg.get("min_line_ratio", 0.09)),
            min_top_green_ratio=float(rally_cfg.get("min_top_green_ratio", 0.05)),
            min_middle_green_ratio=float(rally_cfg.get("min_middle_green_ratio", 0.20)),
            max_left_right_green_diff=float(rally_cfg.get("max_left_right_green_diff", 0.16)),
            min_top_dark_ratio=float(rally_cfg.get("min_top_dark_ratio", 0.80)),
            min_middle_edge_ratio=float(rally_cfg.get("min_middle_edge_ratio", 0.16)),
            pad_before_seconds=float(rally_cfg.get("pad_before_seconds", 0.4)),
            pad_after_seconds=float(rally_cfg.get("pad_after_seconds", 0.6)),
            max_pre_context_seconds=float(rally_cfg.get("max_pre_context_seconds", 2.2)),
            max_post_context_seconds=float(rally_cfg.get("max_post_context_seconds", 1.4)),
            allowed_context_drop_samples=int(rally_cfg.get("allowed_context_drop_samples", 1)),
            overwrite=True,
            scoreboard_score_roi=(
                tuple(rally_cfg["scoreboard_score_roi"])
                if rally_cfg.get("scoreboard_score_roi")
                else None
            ),
            scoreboard_context_roi=(
                tuple(rally_cfg["scoreboard_context_roi"])
                if rally_cfg.get("scoreboard_context_roi")
                else None
            ),
            scoreboard_max_lag_seconds=float(
                rally_cfg.get("scoreboard_max_lag_seconds", 8.0)
            ),
        )
    if stop_after == "rally":
        context.write_manifest()
        return run_dir

    with stage_report(
        context,
        "court_calibration",
        inputs=[str(rallies_csv)],
        outputs=[str(calibration_dir), str(calibration_summary_csv)],
        parameters=config.get("court_calibration", {}),
    ):
        calibrate_courts(rallies_csv, calibration_dir, calibration_preview_dir, calibration_summary_csv)
    if stop_after == "calibrate":
        context.write_manifest()
        return run_dir

    player_cfg = config.get("player_tracking", {})
    with stage_report(
        context,
        "player_tracking",
        inputs=[str(rallies_csv), str(calibration_dir)],
        outputs=[str(player_csv), str(player_summary_csv)],
        parameters=player_cfg,
    ):
        track_players(
            input_path=rallies_csv,
            calibration_dir=calibration_dir,
            output_csv=player_csv,
            summary_csv=player_summary_csv,
            debug_dir=ensure_dir(outputs_dir / "player_tracking_debug"),
            detector=str(player_cfg.get("detector", "heuristic")),
            yolo_model_name=str(player_cfg.get("yolo_model", "yolov8n.pt")),
            yolo_confidence=float(player_cfg.get("yolo_confidence", 0.12)),
            yolo_image_size=int(player_cfg.get("yolo_image_size", 1280)),
            near_max_track_distance=float(player_cfg.get("near_max_track_distance", 120.0)),
            far_max_track_distance=float(player_cfg.get("far_max_track_distance", 170.0)),
            near_max_missing_frames=int(player_cfg.get("near_max_missing_frames", 4)),
            far_max_missing_frames=int(player_cfg.get("far_max_missing_frames", 10)),
            role_half_tolerance=float(player_cfg.get("role_half_tolerance", 48.0)),
        )

    shuttle_cfg = config.get("shuttle_tracking", {})
    with stage_report(
        context,
        "shuttle_tracking",
        inputs=[str(rallies_csv), str(calibration_dir)],
        outputs=[str(shuttle_csv), str(shuttle_summary_csv)],
        parameters=shuttle_cfg,
    ):
        track_shuttle(
            input_path=rallies_csv,
            calibration_dir=calibration_dir,
            output_csv=shuttle_csv,
            summary_csv=shuttle_summary_csv,
            debug_dir=ensure_dir(outputs_dir / "shuttle_tracking_debug"),
            diff_threshold=int(shuttle_cfg.get("diff_threshold", 18)),
            max_jump=float(shuttle_cfg.get("max_jump", 80.0)),
            max_missing_frames=int(shuttle_cfg.get("max_missing_frames", 3)),
            min_brightness=int(shuttle_cfg.get("min_brightness", 165)),
            min_candidate_area=float(shuttle_cfg.get("min_candidate_area", 1.0)),
            max_candidate_area=float(shuttle_cfg.get("max_candidate_area", 55.0)),
            max_candidate_size=int(shuttle_cfg.get("max_candidate_size", 14)),
            direction_weight=float(shuttle_cfg.get("direction_weight", 24.0)),
            speed_weight=float(shuttle_cfg.get("speed_weight", 0.35)),
        )
    if stop_after == "tracking":
        context.write_manifest()
        return run_dir

    smooth_cfg = config.get("smoothing", {})
    with stage_report(
        context,
        "trajectory_smoothing",
        inputs=[str(player_csv), str(shuttle_csv)],
        outputs=[str(player_smoothed_csv), str(shuttle_smoothed_csv)],
        parameters=smooth_cfg,
    ):
        smooth_trajectory(
            player_csv,
            player_smoothed_csv,
            player_smoothing_summary,
            float(smooth_cfg.get("min_confidence", 0.2)),
            int(smooth_cfg.get("max_gap_frames", 4)),
            int(smooth_cfg.get("window_size", 5)),
            float(smooth_cfg.get("ema_alpha", 0.35)),
        )
        smooth_trajectory(
            shuttle_csv,
            shuttle_smoothed_csv,
            shuttle_smoothing_summary,
            float(smooth_cfg.get("min_confidence", 0.2)),
            int(smooth_cfg.get("max_gap_frames", 4)),
            int(smooth_cfg.get("window_size", 5)),
            float(smooth_cfg.get("ema_alpha", 0.35)),
        )

    if not skip_visualize:
        chart_dir = ensure_dir(outputs_dir / "tracking_charts")
        with stage_report(
            context,
            "visualization",
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = run_pipeline(
        input_video=args.input,
        run_id=args.run_id,
        config_path=args.config,
        stop_after=args.stop_after,
        skip_visualize=args.skip_visualize,
    )
    print(f"Pipeline run directory: {run_dir}")
    return 0

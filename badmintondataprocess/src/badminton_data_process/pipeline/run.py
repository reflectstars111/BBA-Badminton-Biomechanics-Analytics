from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from badminton_data_process.calibration.court import calibrate_courts
from badminton_data_process.core.artifacts import (
    inspect_calibration_json,
    inspect_csv,
    inspect_directory,
    inspect_file,
    inspect_file_set,
    inspect_video,
)
from badminton_data_process.core.config import load_config
from badminton_data_process.core.config_schema import parse_config
from badminton_data_process.core.io import ensure_dir, read_csv_rows, write_csv_rows
from badminton_data_process.core.paths import RunLayout, discover_project_root
from badminton_data_process.core.run import RunContext, make_run_id, stage_report
from badminton_data_process.core.schemas import (
    COURT_CALIBRATION_SUMMARY_FIELDS,
    MAIN_VIEW_FRAME_FIELDS,
    MAIN_VIEW_QUALITY_FIELDS,
    MAIN_VIEW_SEGMENT_FIELDS,
    PLAYER_TRACK_FIELDS,
    RALLY_DECISION_FIELDS,
    RALLY_FIELDS,
    SHUTTLE_TRACK_FIELDS,
    StageName,
)
from badminton_data_process.main_view.analyze import analyze_main_view
from badminton_data_process.media.export import export_browser_video
from badminton_data_process.rally.segmentation import (
    segment_rallies_with_timeline as segment_rallies,
)
from badminton_data_process.smoothing.trajectory import smooth_trajectory
from badminton_data_process.tactics.analyze import main as tactics_main
from badminton_data_process.tracking.player.tracking import track_players
from badminton_data_process.tracking.player.pose import PoseRuntimeConfig
from badminton_data_process.tracking.shuttle.tracking import track_shuttle
from badminton_data_process.visualization.tracking import main as visualize_tracking_main
from badminton_data_process.visualization.demo import render_demo


PLAYER_SUMMARY_REQUIRED_FIELDS = {"video_stem", "status", "track_rows"}
SHUTTLE_SUMMARY_REQUIRED_FIELDS = {
    "video_stem",
    "status",
    "track_rows",
    "visible_rows",
}
SMOOTHING_SUMMARY_REQUIRED_FIELDS = {
    "schema",
    "rows",
    "source_valid_rows",
    "smoothed_valid_rows",
}
TACTICS_SUMMARY_REQUIRED_FIELDS = {
    "rally_id",
    "player_id",
    "analysis_mode",
    "event_eligibility",
    "event_reject_reason",
    "frames_valid",
    "movement_eligibility",
    "movement_reject_reason",
    "distance_steps",
}
TACTICS_EVENT_REQUIRED_FIELDS = {
    "rally_id",
    "frame_id",
    "event_type",
    "player_id",
    "image_x",
    "image_y",
    "event_eligibility",
}


def _artifact_path(value: str, project_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _analysis_mode_for_roles(roles: list[str]) -> str:
    return "near_only" if set(roles) == {"near"} else "experimental_two_player"


def run_pipeline(
    input_video: Path,
    run_id: str | None = None,
    config_path: Path | None = None,
    root: Path | None = None,
    stop_after: str | None = None,
    skip_visualize: bool = False,
    skip_demo: bool = False,
    force: bool = False,
    runs_dir: Path | None = None,
) -> Path:
    project_root = root or discover_project_root()
    config = load_config(config_path, root=project_root)
    cfg = parse_config(config)
    resolved_run_id = run_id or make_run_id("pipeline")
    layout = RunLayout.create(
        project_root,
        resolved_run_id,
        runs_dir if runs_dir is not None else cfg.data.runs_dir,
    )
    context = RunContext(project_root, resolved_run_id, config, layout=layout)
    context.ensure()
    completed: set[StageName] = set() if force else context.resume()
    if completed and StageName.MAIN_VIEW not in completed:
        raise RuntimeError(
            "Run manifest predates the mandatory Main View stage; use a new run_id "
            "or rerun explicitly with --force instead of reusing unconstrained rallies"
        )

    run_dir = layout.run_dir
    ensure_dir(layout.annotations_dir)
    ensure_dir(layout.outputs_dir)
    main_view_dir = ensure_dir(layout.main_view_dir)
    rallies_dir = ensure_dir(layout.rallies_dir)
    calibration_dir = ensure_dir(layout.court_calibration_dir)
    calibration_preview_dir = ensure_dir(layout.court_calibration_debug_dir)

    main_view_frame_scores_csv = layout.main_view_frame_scores_csv
    main_view_segments_csv = layout.main_view_segments_csv
    main_view_quality_csv = layout.main_view_quality_csv
    main_view_timeline_json = layout.main_view_timeline_json
    rallies_csv = layout.rallies_csv
    rally_decisions_csv = layout.rally_decisions_csv
    calibration_summary_csv = layout.court_calibration_summary_csv
    player_csv = layout.player_tracks_csv
    player_summary_csv = layout.player_tracking_summary_csv
    shuttle_csv = layout.shuttle_tracks_csv
    shuttle_summary_csv = layout.shuttle_tracking_summary_csv
    player_smoothed_csv = layout.player_tracks_smoothed_csv
    shuttle_smoothed_csv = layout.shuttle_tracks_smoothed_csv
    player_smoothing_summary = layout.player_smoothing_summary_csv
    shuttle_smoothing_summary = layout.shuttle_smoothing_summary_csv
    tactics_dir = ensure_dir(layout.tactics_dir)
    tactics_summary_csv = layout.tactics_summary_csv
    tactics_events_csv = layout.tactics_events_csv
    ensure_dir(layout.demo_dir)

    if StageName.MAIN_VIEW not in completed:
        main_view_cfg = cfg.main_view
        with stage_report(
            context,
            StageName.MAIN_VIEW,
            inputs=[str(input_video)],
            outputs=[
                str(main_view_frame_scores_csv),
                str(main_view_segments_csv),
                str(main_view_quality_csv),
                str(main_view_timeline_json),
            ],
            parameters=asdict(main_view_cfg),
        ) as stage:
            stage.accept_legacy(
                analyze_main_view(
                    input_video=input_video,
                    output_dir=main_view_dir,
                    sample_every=main_view_cfg.sample_every,
                    threshold=main_view_cfg.threshold,
                    min_segment_seconds=main_view_cfg.min_segment_seconds,
                    max_gap_seconds=main_view_cfg.max_gap_seconds,
                ),
                operation="main-view analysis",
            )
            stage.require_artifact(
                inspect_csv(
                    main_view_frame_scores_csv,
                    name="main-view frame scores",
                    min_rows=1,
                    required_fields=MAIN_VIEW_FRAME_FIELDS,
                )
            )
            stage.require_artifact(
                inspect_csv(
                    main_view_segments_csv,
                    name="accepted main-view segments",
                    min_rows=1,
                    required_fields=MAIN_VIEW_SEGMENT_FIELDS,
                )
            )
            stage.require_artifact(
                inspect_csv(
                    main_view_quality_csv,
                    name="main-view quality",
                    min_rows=1,
                    required_fields=MAIN_VIEW_QUALITY_FIELDS,
                )
            )
            stage.require_artifact(
                inspect_file(main_view_timeline_json, name="main-view timeline")
            )
    if stop_after == "main_view":
        context.write_manifest()
        return run_dir

    if StageName.RALLY_SEGMENTATION not in completed:
        rally_cfg = cfg.rally_segmentation
        with stage_report(
            context,
            StageName.RALLY_SEGMENTATION,
            inputs=[str(input_video), str(main_view_timeline_json)],
            outputs=[str(rallies_dir), str(rallies_csv), str(rally_decisions_csv)],
            parameters=asdict(rally_cfg),
        ) as stage:
            stage.accept_legacy(
                segment_rallies(
                    input_path=input_video,
                    timeline_path=main_view_timeline_json,
                    output_dir=rallies_dir,
                    metadata_csv=rallies_csv,
                    decisions_csv=rally_decisions_csv,
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
                    pre_context_seconds=rally_cfg.pre_context_seconds,
                    post_context_seconds=rally_cfg.post_context_seconds,
                    min_active_samples=rally_cfg.min_active_samples,
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
                ),
                operation="rally segmentation",
            )
            stage.require_artifact(
                inspect_csv(
                    rally_decisions_csv,
                    name="Usable Rally decisions",
                    min_rows=1,
                    required_fields=RALLY_DECISION_FIELDS,
                )
            )
            decision_rows = read_csv_rows(rally_decisions_csv)
            if decision_rows and not any(
                row.get("status") == "accepted" for row in decision_rows
            ):
                reason_counts: dict[str, int] = {}
                for row in decision_rows:
                    reason = row.get("reason") or "unspecified"
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
                reasons = ", ".join(
                    f"{reason}={count}" for reason, count in sorted(reason_counts.items())
                )
                stage.reject(f"no Usable Rally accepted: {reasons}")
            stage.require_artifact(
                inspect_csv(
                    rallies_csv,
                    name="rally metadata",
                    min_rows=1,
                    required_fields=RALLY_FIELDS,
                )
            )
            rally_rows = read_csv_rows(rallies_csv)
            for row in rally_rows:
                rally_id = row.get("rally_id", "unknown")
                stage.require_artifact(
                    inspect_video(
                        _artifact_path(row.get("output_path", ""), project_root),
                        name=f"rally video {rally_id}",
                    )
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
        ) as stage:
            calibration_result = calibrate_courts(
                rallies_csv,
                calibration_dir,
                calibration_preview_dir,
                calibration_summary_csv,
                reference_points=calibration_cfg.reference_points,
                min_line_support=calibration_cfg.min_line_support,
                detector=calibration_cfg.detector,
                min_area_ratio=calibration_cfg.min_area_ratio,
                max_condition_number=calibration_cfg.max_condition_number,
                max_reprojection_error_px=calibration_cfg.max_reprojection_error_px,
                stability_corner_rmse_ratio=calibration_cfg.stability_corner_rmse_ratio,
                min_stable_candidates=calibration_cfg.min_stable_candidates,
            )
            stage.require_artifact(
                inspect_csv(
                    calibration_summary_csv,
                    name="court calibration summary",
                    min_rows=1,
                    required_fields=COURT_CALIBRATION_SUMMARY_FIELDS,
                )
            )
            summary_rows = read_csv_rows(calibration_summary_csv)
            failed_stems = {
                row["video_stem"] for row in summary_rows if row["status"] != "success"
            }
            if calibration_result == 0:
                if failed_stems:
                    raise RuntimeError(
                        "court calibration returned success but its summary contains "
                        f"failed clips: {sorted(failed_stems)}"
                    )
                stage.accept_legacy(0, operation="court calibration")
            elif calibration_result == 2 and failed_stems:
                if len(failed_stems) == len(summary_rows):
                    stage.accept_legacy(calibration_result, operation="court calibration")
                rally_rows = read_csv_rows(rallies_csv)
                fieldnames = list(rally_rows[0]) if rally_rows else RALLY_FIELDS
                kept = [
                    row
                    for row in rally_rows
                    if Path(row["output_path"]).stem not in failed_stems
                ]
                if not kept:
                    raise RuntimeError(
                        "court calibration rejected every rally in the metadata"
                    )
                write_csv_rows(rallies_csv, fieldnames, kept)
                stage.require_artifact(
                    inspect_csv(
                        rallies_csv,
                        name="accepted rally metadata",
                        min_rows=1,
                        required_fields=RALLY_FIELDS,
                    )
                )
                message = (
                    f"court calibration accepted {len(summary_rows) - len(failed_stems)} "
                    f"clip(s) and rejected {len(failed_stems)}: {sorted(failed_stems)}"
                )
                print(message)
                stage.complete(message=message)
            else:
                stage.accept_legacy(calibration_result, operation="court calibration")
            successful_json_paths = [
                _artifact_path(row.get("json_path", ""), project_root)
                for row in summary_rows
                if row["status"] == "success"
            ]
            stage.require_artifact(
                inspect_file_set(successful_json_paths, name="court calibration JSON files")
            )
            for calibration_json in successful_json_paths:
                stage.require_artifact(
                    inspect_calibration_json(
                        calibration_json,
                        name=f"validated calibration {calibration_json.stem}",
                    )
                )
    if stop_after == "calibrate":
        context.write_manifest()
        return run_dir

    player_cfg = cfg.player_tracking
    if StageName.PLAYER_TRACKING not in completed:
        with stage_report(
            context,
            StageName.PLAYER_TRACKING,
            inputs=[str(rallies_csv), str(calibration_dir)],
            outputs=[str(player_csv), str(player_summary_csv)],
            parameters=asdict(player_cfg),
        ) as stage:
            stage.accept_legacy(
                track_players(
                    input_path=rallies_csv,
                    calibration_dir=calibration_dir,
                    output_csv=player_csv,
                    summary_csv=player_summary_csv,
                    debug_dir=ensure_dir(layout.player_tracking_debug_dir),
                    detector=player_cfg.detector,
                    yolo_model_name=player_cfg.yolo_model,
                    yolo_confidence=player_cfg.yolo_confidence,
                    yolo_image_size=player_cfg.yolo_image_size,
                    near_max_track_distance=player_cfg.near_max_track_distance,
                    far_max_track_distance=player_cfg.far_max_track_distance,
                    near_max_missing_frames=player_cfg.near_max_missing_frames,
                    far_max_missing_frames=player_cfg.far_max_missing_frames,
                    role_half_tolerance=player_cfg.role_half_tolerance,
                    player_roles=tuple(player_cfg.roles),
                    pose_config=PoseRuntimeConfig(
                        model_name=player_cfg.pose_model,
                        keypoint_threshold=player_cfg.pose_keypoint_confidence,
                        min_valid_keypoints=player_cfg.pose_min_keypoints,
                        rtmpose_mode=player_cfg.rtmpose_mode,
                        rtmpose_backend=player_cfg.rtmpose_backend,
                        rtmpose_device=player_cfg.rtmpose_device,
                        rtmpose_detector_model=player_cfg.rtmpose_detector_model,
                        rtmpose_pose_model=player_cfg.rtmpose_pose_model,
                        rtmpose_detector_input_size=tuple(player_cfg.rtmpose_detector_input_size),
                        rtmpose_pose_input_size=tuple(player_cfg.rtmpose_pose_input_size),
                    ),
                ),
                operation="player tracking",
            )
            stage.require_artifact(
                inspect_csv(
                    player_csv,
                    name="player tracks",
                    min_rows=1,
                    required_fields=PLAYER_TRACK_FIELDS,
                )
            )
            stage.require_artifact(
                inspect_csv(
                    player_summary_csv,
                    name="player tracking summary",
                    min_rows=1,
                    required_fields=PLAYER_SUMMARY_REQUIRED_FIELDS,
                )
            )

    if StageName.SHUTTLE_TRACKING not in completed:
        shuttle_cfg = cfg.shuttle_tracking
        with stage_report(
            context,
            StageName.SHUTTLE_TRACKING,
            inputs=[str(rallies_csv), str(calibration_dir)],
            outputs=[str(shuttle_csv), str(shuttle_summary_csv)],
            parameters=asdict(shuttle_cfg),
        ) as stage:
            stage.accept_legacy(
                track_shuttle(
                    input_path=rallies_csv,
                    calibration_dir=calibration_dir,
                    output_csv=shuttle_csv,
                    summary_csv=shuttle_summary_csv,
                    debug_dir=ensure_dir(layout.shuttle_tracking_debug_dir),
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
                    tracknet_device=shuttle_cfg.tracknet_device,
                    tracknet_vis_threshold=shuttle_cfg.tracknet_vis_threshold,
                ),
                operation="shuttle tracking",
            )
            stage.require_artifact(
                inspect_csv(
                    shuttle_csv,
                    name="shuttle tracks",
                    min_rows=1,
                    required_fields=SHUTTLE_TRACK_FIELDS,
                )
            )
            stage.require_artifact(
                inspect_csv(
                    shuttle_summary_csv,
                    name="shuttle tracking summary",
                    min_rows=1,
                    required_fields=SHUTTLE_SUMMARY_REQUIRED_FIELDS,
                )
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
            outputs=[
                str(player_smoothed_csv),
                str(shuttle_smoothed_csv),
                str(player_smoothing_summary),
                str(shuttle_smoothing_summary),
            ],
            parameters=asdict(smooth_cfg),
        ) as stage:
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
                smooth_cfg.shuttle_max_interpolation_displacement_px,
            )
            stage.require_artifact(
                inspect_csv(
                    player_smoothed_csv,
                    name="smoothed player tracks",
                    min_rows=1,
                    required_fields=PLAYER_TRACK_FIELDS,
                )
            )
            stage.require_artifact(
                inspect_csv(
                    shuttle_smoothed_csv,
                    name="smoothed shuttle tracks",
                    min_rows=1,
                    required_fields=SHUTTLE_TRACK_FIELDS,
                )
            )
            stage.require_artifact(
                inspect_csv(
                    player_smoothing_summary,
                    name="player smoothing summary",
                    min_rows=1,
                    required_fields=SMOOTHING_SUMMARY_REQUIRED_FIELDS,
                )
            )
            stage.require_artifact(
                inspect_csv(
                    shuttle_smoothing_summary,
                    name="shuttle smoothing summary",
                    min_rows=1,
                    required_fields=SMOOTHING_SUMMARY_REQUIRED_FIELDS,
                )
            )

    if not skip_visualize and StageName.VISUALIZATION not in completed:
        chart_dir = ensure_dir(layout.tracking_charts_dir)
        with stage_report(
            context,
            StageName.VISUALIZATION,
            inputs=[str(shuttle_summary_csv), str(shuttle_smoothed_csv), str(player_smoothed_csv)],
            outputs=[str(chart_dir)],
            parameters={},
        ) as stage:
            stage.accept_legacy(
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
                ),
                operation="tracking visualization",
            )
            stage.require_artifact(
                inspect_directory(
                    chart_dir,
                    name="tracking charts",
                    pattern="*.png",
                    min_files=1,
                )
            )

    if StageName.TACTICAL_ANALYSIS not in completed:
        tactics_cfg = cfg.tactical_analysis
        with stage_report(
            context,
            StageName.TACTICAL_ANALYSIS,
            inputs=[str(player_smoothed_csv), str(shuttle_smoothed_csv), str(calibration_dir)],
            outputs=[str(tactics_summary_csv), str(tactics_events_csv)],
            parameters=asdict(tactics_cfg),
        ) as stage:
            stage.accept_legacy(
                tactics_main(
                    [
                        str(player_smoothed_csv),
                        str(shuttle_smoothed_csv),
                        "--calibration-dir",
                        str(calibration_dir),
                        "--output-dir",
                        str(tactics_dir),
                        "--hit-distance-px",
                        str(tactics_cfg.hit_distance_px),
                        "--turn-angle-deg",
                        str(tactics_cfg.turn_angle_deg),
                        "--min-event-gap-frames",
                        str(tactics_cfg.min_event_gap_frames),
                        "--analysis-mode",
                        _analysis_mode_for_roles(player_cfg.roles),
                    ]
                ),
                operation="tactical analysis",
            )
            stage.require_artifact(
                inspect_csv(
                    tactics_summary_csv,
                    name="tactics summary",
                    min_rows=1,
                    required_fields=TACTICS_SUMMARY_REQUIRED_FIELDS,
                )
            )
            stage.require_artifact(
                inspect_csv(
                    tactics_events_csv,
                    name="tactics events",
                    min_rows=0,
                    required_fields=TACTICS_EVENT_REQUIRED_FIELDS,
                )
            )

    demo_cfg = cfg.demo_rendering
    demo_output = layout.demo_output(demo_cfg.output_filename)
    demo_intermediate = layout.demo_intermediate_output(demo_cfg.output_filename)
    if demo_cfg.enabled and not skip_demo and StageName.DEMO_RENDERING not in completed:
        with stage_report(
            context,
            StageName.DEMO_RENDERING,
            inputs=[
                str(rallies_csv),
                str(player_smoothed_csv),
                str(shuttle_smoothed_csv),
                str(calibration_dir),
                str(tactics_events_csv),
                str(tactics_summary_csv),
            ],
            outputs=[str(demo_intermediate), str(demo_output)],
            parameters=asdict(demo_cfg),
        ) as stage:
            render_target = demo_intermediate if demo_cfg.browser_compatible else demo_output
            render_demo(
                rallies_csv=rallies_csv,
                player_tracks_csv=player_smoothed_csv,
                shuttle_tracks_csv=shuttle_smoothed_csv,
                calibration_dir=calibration_dir,
                tactics_events_csv=tactics_events_csv,
                tactics_summary_csv=tactics_summary_csv,
                output_video=render_target,
                max_rallies=demo_cfg.max_rallies,
                trail_length=demo_cfg.trail_length,
                event_hold_frames=demo_cfg.event_hold_frames,
                show_topdown=demo_cfg.show_topdown,
                show_stats=demo_cfg.show_stats,
                codec=demo_cfg.codec,
            )
            if demo_cfg.browser_compatible:
                stage.require_artifact(
                    inspect_video(render_target, name="OpenCV diagnostic demo video")
                )
                export_result = export_browser_video(
                    render_target,
                    demo_output,
                    preserve_audio=demo_cfg.preserve_audio,
                )
            else:
                export_result = None
            final_report = inspect_video(demo_output, name="browser-compatible diagnostic demo video")
            if export_result is not None:
                final_report.details.update(export_result.artifact_details())
            stage.require_artifact(
                final_report
            )

    context.write_manifest()
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the end-to-end badminton data pipeline.")
    parser.add_argument("input", type=Path, help="Raw match video path.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help="Override data.runs_dir; relative paths are resolved from the project root.",
    )
    parser.add_argument(
        "--stop-after",
        choices=["main_view", "rally", "calibrate", "tracking"],
        default=None,
    )
    parser.add_argument("--skip-visualize", action="store_true")
    parser.add_argument("--skip-demo", action="store_true")
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
        skip_demo=args.skip_demo,
        force=args.force,
        runs_dir=args.runs_dir,
    )
    print(f"Pipeline run directory: {run_dir}")
    return 0

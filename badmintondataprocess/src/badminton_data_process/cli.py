from __future__ import annotations

import argparse
import sys

from badminton_data_process.calibration.court import main as calibrate_main
from badminton_data_process.core.verify import main as verify_main
from badminton_data_process.evaluation.compare import main as compare_trackers_main
from badminton_data_process.main_view.analyze import main as main_view_analyze_main
from badminton_data_process.main_view.export import main as main_view_export_main
from badminton_data_process.metadata.download import main as download_main
from badminton_data_process.metadata.matches import main as metadata_validate_main
from badminton_data_process.metadata.register import main as metadata_register_main
from badminton_data_process.pipeline.batch import main as pipeline_batch_main
from badminton_data_process.pipeline.full import main as full_analysis_main
from badminton_data_process.pipeline.run import main as pipeline_run_main
from badminton_data_process.rally.segmentation import main as rally_segment_main
from badminton_data_process.review.main_view import main as review_main_view_main
from badminton_data_process.smoothing.trajectory import main as smooth_main
from badminton_data_process.tactics.analyze import main as tactics_main
from badminton_data_process.tracking.player.tracking import main as track_players_main
from badminton_data_process.tracking.shuttle.tracking import main as track_shuttle_main
from badminton_data_process.video.preprocess import main as preprocess_main
from badminton_data_process.visualization.tracking import main as visualize_main
from badminton_data_process.visualization.demo import main as render_demo_main
from badminton_data_process.webui.app import main as webui_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bdp",
        description="Badminton data process unified CLI.",
    )
    parser.add_argument(
        "command",
        nargs="*",
        help=(
            "Commands: metadata validate/register-local, download, preprocess, rally segment, "
            "main-view analyze/export, review main-view, calibrate, track players, "
            "track shuttle, smooth, visualize, tactics analyze, compare trackers, "
            "render demo, analyze, pipeline run/batch, webui, verify"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        build_parser().print_help()
        return 0

    if argv[:2] == ["metadata", "validate"]:
        return metadata_validate_main(argv[2:])
    if argv[:2] == ["metadata", "register-local"]:
        return metadata_register_main(argv[2:])
    if argv[0] == "download":
        return download_main(argv[1:])
    if argv[0] == "preprocess":
        return preprocess_main(argv[1:])
    if argv[:2] == ["main-view", "analyze"]:
        return main_view_analyze_main(argv[2:])
    if argv[:2] == ["main-view", "export"]:
        return main_view_export_main(argv[2:])
    if argv[:2] == ["rally", "segment"]:
        return rally_segment_main(argv[2:])
    if argv[:2] == ["review", "main-view"]:
        return review_main_view_main(argv[2:])
    if argv[0] == "calibrate":
        return calibrate_main(argv[1:])
    if argv[:2] == ["track", "players"]:
        return track_players_main(argv[2:])
    if argv[:2] == ["track", "shuttle"]:
        return track_shuttle_main(argv[2:])
    if argv[0] == "smooth":
        return smooth_main(argv[1:])
    if argv[0] == "visualize":
        return visualize_main(argv[1:])
    if argv[:2] == ["tactics", "analyze"]:
        return tactics_main(argv[2:])
    if argv[:2] == ["render", "demo"]:
        return render_demo_main(argv[2:])
    if argv[:2] == ["compare", "trackers"]:
        return compare_trackers_main(argv[2:])
    if argv[:2] == ["pipeline", "batch"]:
        return pipeline_batch_main(argv[2:])
    if argv[:2] == ["pipeline", "run"]:
        return pipeline_run_main(argv[2:])
    if argv[0] == "analyze":
        return full_analysis_main(argv[1:])
    if argv[0] == "verify":
        return verify_main(argv[1:])
    if argv[0] == "webui":
        return webui_main(argv[1:])

    print(f"Unknown command: {' '.join(argv)}", file=sys.stderr)
    build_parser().print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

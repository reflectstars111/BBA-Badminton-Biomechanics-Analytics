from __future__ import annotations

from pathlib import Path

import pytest

from badminton_data_process.core.io import write_csv_rows, write_json
from badminton_data_process.webui.app import (
    action_event_table,
    build_app,
    duration_label,
    estimate_remaining_seconds,
    pipeline_progress,
    player_overview_table,
    player_rally_table,
    progress_html,
    quality_markdown,
    shuttle_rally_table,
    swing_phase_table,
)
from badminton_data_process.webui.reporting import build_web_report
from badminton_data_process.webui.styles import WEBUI_CSS


def test_report_tables_and_stage_codes_pin_a_light_high_contrast_palette() -> None:
    """Dark browser preferences must not leak into the white report surface."""
    expected_rules = (
        ".report-table {",
        "--table-even-background-fill: #ffffff;",
        "--table-odd-background-fill: #f5f6f3;",
        "--body-text-color: #1b211f;",
        ".stage-log .md :not(pre) > code {",
        "background: #eef0ed !important;",
        "color: #17201c !important;",
    )

    for rule in expected_rules:
        assert rule in WEBUI_CSS

    config = build_app().get_config_file()
    report_tables = [
        component
        for component in config["components"]
        if component["type"] == "dataframe"
    ]
    assert len(report_tables) == 5
    assert all(
        component["props"].get("elem_classes") == ["report-table"]
        for component in report_tables
    )


def _player_row(player_id: str, frame: int, timestamp: float, court_y: float) -> dict[str, object]:
    return {
        "video_stem": "rally_001",
        "rally_id": "001",
        "frame_id": frame,
        "timestamp": timestamp,
        "player_id": player_id,
        "bbox_y1": 0,
        "bbox_y2": 240,
        "body_image_y": 100,
        "ground_image_y": 200,
        "pose_valid": 1,
        "smoothed_court_x": 3.0,
        "smoothed_court_y": court_y,
        "is_smoothed_valid": 1,
    }


def _shuttle_row(frame: int, timestamp: float, x: float) -> dict[str, object]:
    return {
        "video_stem": "rally_001",
        "rally_id": "001",
        "frame_id": frame,
        "timestamp": timestamp,
        "smoothed_x": x,
        "smoothed_y": 20,
        "is_smoothed_valid": 1,
        "is_gap_filled": 0,
    }


def test_web_report_exposes_detailed_trustworthy_match_and_rally_metrics(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "web_report"
    player_path = run_dir / "annotations" / "player_tracks_smoothed.csv"
    shuttle_path = run_dir / "annotations" / "shuttle_tracks_smoothed.csv"
    players = [
        _player_row("near", 0, 0.0, 8.0),
        _player_row("near", 1, 0.1, 8.02),
        _player_row("far", 0, 0.0, 4.0),
        _player_row("far", 1, 0.1, 3.98),
    ]
    shuttles = [_shuttle_row(0, 0.0, 10.0), _shuttle_row(1, 0.1, 20.0)]
    write_csv_rows(player_path, list(players[0]), players)
    write_csv_rows(shuttle_path, list(shuttles[0]), shuttles)
    write_csv_rows(
        run_dir / "rallies.csv",
        ["video_stem", "rally_id", "duration_seconds"],
        [{"video_stem": "rally_001", "rally_id": "001", "duration_seconds": 0.2}],
    )
    write_json(
        run_dir / "analysis_summary.json",
        {
            "run_id": "web_report",
            "status": "success",
            "outputs": {"analysis_video": str(run_dir / "outputs" / "demo.mp4")},
        },
    )

    report = build_web_report(run_dir)

    assert report["match"]["usable_rallies"] == 1
    assert report["match"]["analyzed_duration_seconds"] == pytest.approx(0.2)
    assert {row["player_id"] for row in report["players"]} == {"near", "far"}
    near = next(row for row in report["players"] if row["player_id"] == "near")
    assert near["total_distance_m"] == pytest.approx(0.02)
    assert near["average_speed_m_s"] == pytest.approx(0.2)
    assert near["tracking_coverage_ratio"] == pytest.approx(1.0)
    assert near["pose_valid_ratio"] == pytest.approx(1.0)
    assert near["average_body_center_height_ratio"] == pytest.approx(0.417, abs=0.001)
    assert report["shuttle"]["average_image_speed_px_s"] == pytest.approx(100.0)
    assert report["quality"]["metric_contract"].startswith("Player movement")
    assert report["development"]["bone_action_detail"].startswith("二维动作分析基础版已上线")
    assert (run_dir / "webui_report.json").is_file()

    assert player_overview_table(report)[0][0] == "far"
    assert len(player_rally_table(report)) == 2
    assert shuttle_rally_table(report)[0][3] == "100.0"
    assert "骨骼动作细节分析" in quality_markdown(report)
    assert action_event_table(report) == []
    assert swing_phase_table(report) == []


def test_pipeline_progress_reports_current_stage_from_manifest() -> None:
    state = {
        "stages": [
            {"name": "main_view", "status": "success"},
            {"name": "rally_segmentation", "status": "success"},
        ]
    }

    progress = pipeline_progress(state, running=True)

    assert progress == {
        "completed": 2,
        "total": 10,
        "percent": 20,
        "current_stage": "白色边线与球场标定",
        "state_label": "正在分析",
        "failed": False,
    }
    html = progress_html(state, running=True)
    assert 'value="20"' in html
    assert "已完成 2 / 10 阶段" in html


def test_pipeline_progress_distinguishes_finalizing_complete_and_failure() -> None:
    successful_stages = [
        {"name": name, "status": "success"}
        for name in (
            "main_view", "rally_segmentation", "court_calibration", "player_tracking",
            "shuttle_tracking", "trajectory_smoothing", "visualization",
            "tactical_analysis", "biomechanics_analysis", "demo_rendering",
        )
    ]
    assert pipeline_progress({"stages": successful_stages}, running=True)["percent"] == 99
    finished = pipeline_progress({"stages": successful_stages}, running=False)
    assert finished["percent"] == 100
    assert finished["current_stage"] == "全部阶段已完成"

    failed = pipeline_progress(
        {
            "stages": [
                {"name": "main_view", "status": "success"},
                {"name": "rally_segmentation", "status": "failed"},
            ]
        },
        running=False,
    )
    assert failed["percent"] == 10
    assert failed["current_stage"] == "有效回合切分"
    assert failed["state_label"] == "分析失败"


def test_progress_estimates_remaining_time_from_live_stage_units() -> None:
    state = {
        "stages": [
            {"name": "main_view", "status": "success", "duration_seconds": 10.0},
            {"name": "rally_segmentation", "status": "success", "duration_seconds": 20.0},
        ]
    }

    live_progress = {
        "stage": "player_tracking",
        "completed_units": 500,
        "total_units": 1000,
        "stage_elapsed_seconds": 120.0,
    }
    remaining = estimate_remaining_seconds(state, live_progress)

    assert remaining == pytest.approx(416.04, rel=1e-3)
    assert duration_label(remaining, approximate=True) == "约 7 分钟"
    html = progress_html(
        state,
        running=True,
        elapsed_seconds=155.0,
        live_progress=live_progress,
    )
    assert "预计剩余" in html
    assert "约 7 分钟" in html
    assert "500 / 1000 帧（50.0%）" in html
    assert "已运行" in html
    assert "3 分钟" in html


def test_progress_waits_for_runtime_evidence_before_showing_eta() -> None:
    html = progress_html({"stages": []}, running=True, elapsed_seconds=4.0)

    assert estimate_remaining_seconds({"stages": []}) is None
    assert "正在测量当前阶段" in html
